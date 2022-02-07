using LinearAlgebra
using Random

export parameter_update, one_step_EM_estimate_parameter, initial_parameters, parameter_tie,
reset_prob

"""
Parameter update wrapper
"""
function parameter_update(pc::StructType, train_x::FairDataset; pseudocount)

    if pc isa LatentStructType
        one_step_EM_estimate_parameter(pc.pc, train_x.data; pseudocount=pseudocount)
    elseif pc isa NonLatentStructType
        estimate_parameters!(pc.pc, train_x.data; pseudocount=pseudocount)
    else
        error("")
    end

    if !(pc isa TwoNB)
        parameter_tie(pc.pc)
    end
end

"""
One Expectation Maximization step
"""
function one_step_EM_estimate_parameter(pc::ProbCircuit, train_x::DataFrame; pseudocount::Float64=1.0)
    estimate_parameters_em!(pc, train_x; pseudocount=pseudocount) 
end


"""
Initialize the parameters with algorithm `alg`
"""
function random_parameters(pc::ProbCircuit)
    foreach(pc) do n
        if is⋁gate(n) 
            if num_children(n) == 1
                n.log_probs .= 0.0
            else
                thetas = LinearAlgebra.normalize(rand(Float64, num_children(n)), 1.0)
                n.log_probs .= log.(thetas)
            end
        end
    end
end

"""
Initialize parameters
`para_alg` is one of `[estimate, rand, uni, prior-subop, prior-rand, non, prior-latent]`
"""
function initial_parameters(fairpc::StructType, fairdata::FairDataset; para_alg, pseudocount)
    pc = fairpc.pc
    data = fairdata.data    
    if para_alg == "estimate"
        @assert fairpc isa NonLatentStructType
        estimate_parameters!(pc, data; pseudocount=pseudocount)
    
    elseif para_alg == "rand"
        random_parameters(pc)
    
    elseif para_alg == "uni"
        uniform_parameters(pc)
    
    elseif para_alg == "prior-subop" || para_alg == "prior-rand"
        @assert fairpc isa LatentStructType
        maxiters = para_alg == "prior-subop" ? 100 : 0
        
        S = fairpc.S
        @assert S == fairdata.S
        D = fairpc.D
        @assert D == fairdata.D
        
        # evaluate 4 sub circuits separately
        sub_circuits = non_sensitive_sub_circuits(fairpc)
        dict_tmp = [Dict(D=>1, S=>1), Dict(D=>1, S=>0), Dict(D=>0, S=>1), Dict(D=>0, S=>0)]
        sub_datas = map(dict_tmp) do given
            data_given_constraint(data; given=given, type=Int8, keep_var=true, keep_Df=false)
        end
        for (sub_circuit, sub_data) in zip(sub_circuits, sub_datas)
            random_parameters(sub_circuit)
            for i in 1: maxiters
                one_step_EM_estimate_parameter(sub_circuit, sub_data; pseudocount=pseudocount)
            end
        end

        # set CPT between Df, D, and S with prior knowledge
        flow_cnts = num_examples.(sub_datas)
        root_thetas = LinearAlgebra.normalize(flow_cnts .+ pseudocount, 1.0)
        D_counts = [[flow_cnts[1], 0], [flow_cnts[2], 0], [0, flow_cnts[3]], [0, flow_cnts[4]]]
        D_thetas = map(x->LinearAlgebra.normalize(x .+ pseudocount), D_counts)
        pc.log_probs .= log.(root_thetas)
        for (n, theta) in zip(nodes_D_sum(fairpc), D_thetas)
            n.log_probs .= log.(theta)
        end

    elseif para_alg == "non"
    elseif para_alg == "prior-latent"
        pc = fairpc.pc
        @assert fairpc isa LatNB
        random_parameters(pc)
        num = num_examples(data)
        S = fairdata.S
        D = fairdata.D
        count = Dict()
        map([Dict(D=>1, S=>1), Dict(D=>1, S=>0),
                        Dict(D=>0, S=>1), Dict(D=>0, S=>0)]) do given
            count[given] = num_examples(data_given_constraint(data; given=given, type=Int8, keep_var=true, keep_Df=false))
        end
        num_S = count[Dict(D=>0, S=>1)] + count[Dict(D=>1, S=>1)]
        num_D = count[Dict(D=>1, S=>1)] + count[Dict(D=>1, S=>0)]
        num_move = (count[Dict(D=>1, S=>0)] * num_S- count[Dict(D=>1, S=>1)] * (num - num_S)) / num
        @assert isapprox((count[Dict(D=>1, S=>0)] - num_move) / (num - num_S), (count[Dict(D=>1, S=>1)] + num_move) / num_S; atol=1e-6) "$num_move $count"
        @assert num_move >= 0 "$num_move"
        new_count = Dict()
        Df = D + 1
        new_count[Dict(D=>1, S=>0, Df=>1)] = count[Dict(D=>1, S=>0)] - num_move
        new_count[Dict(D=>1, S=>0, Df=>0)] = num_move
        new_count[Dict(D=>0, S=>0, Df=>1)] = 0
        new_count[Dict(D=>0, S=>0, Df=>0)] = count[Dict(D=>0, S=>0)]
        new_count[Dict(D=>1, S=>1, Df=>1)] = count[Dict(D=>1, S=>1)]
        new_count[Dict(D=>1, S=>1, Df=>0)] = 0
        new_count[Dict(D=>0, S=>1, Df=>1)] = num_move
        new_count[Dict(D=>0, S=>1, Df=>0)] = count[Dict(D=>0, S=>1)] - num_move

        num_Df = new_count[Dict(D=>1, S=>1, Df=>1)] + new_count[Dict(D=>0, S=>1, Df=>1)] + new_count[Dict(D=>1, S=>0, Df=>1)] + new_count[Dict(D=>0, S=>0, Df=>1)]
        PD = []
        for df in [1, 0]
            for s in [1, 0]
                push!(PD, new_count[Dict(D=>1, S=>s, Df=>df)] / (new_count[Dict(D=>1, S=>s, Df=>df)] + new_count[Dict(D=>0, S=>s, Df=>df)]))
            end
        end

        # reset paras
        pS = num_S / num
        pDf = num_Df / num
        @assert pS >= 0 && pS <= 1 && pDf >= 0 && pDf <= 1
        pc.log_probs .= log.([pDf * pS, pDf * (1 - pS), (1 - pDf) * pS, (1 - pDf) * (1 - pS)])

        for (pn, p) in zip(nodes_D_sum(fairpc), PD)
            pn.log_probs .= log.([p, 1 - p])
        end
   
    end

    @assert fairpc.pc == pc
    if !(fairpc isa TwoNB)
        parameter_tie(pc)
    end
end

"""
Pamater tie between Df and S, Df is independant of S
"""
function parameter_tie(pc::ProbCircuit)
    thetas = exp.(pc.log_probs)
    @assert length(thetas) == 4
    p1, p2 = thetas[1] + thetas[2], thetas[1] + thetas[3]
    thetas2 = [p1 * p2, p1 * (1 - p2), (1 - p1) * p2, (1 - p1) * (1 - p2)]
    pc.log_probs .= log.(thetas2)
end

"""
Reset parameters of circuit `fairpc` given probabilities `prob`
"""
function reset_prob(fairpc::StructType, prob::Dict;init_X=true, prior=0.5)
    # set pDf, pS
    pDf, pS = prob["Df"], prob["S"]
    pc_root = [pDf * pS, pDf * (1 - pS), (1 - pDf) * pS, (1 - pDf) * (1 - pS)]
    fairpc.pc.log_probs .= log.(pc_root)

    # set pD
    map(zip(nodes_D_sum(fairpc), [prob["D|Df,S"], prob["D|Df,notS"], prob["D|notDf,S"], prob["D|notDf,notS"]])) do (pn, p)
        pn.log_probs .= log.([p, 1 - p])
    end

    # rand set pX
    cnt = 1
    if init_X
        map(non_sensitive_sub_circuits(fairpc)) do pc
            if cnt == 1 || cnt == 2
                prior_vec = [prior, 0.0]
            else
                prior_vec = [0.0, prior]
            end
            cnt += 1
            foreach(pc) do n
                if is⋁gate(n) && num_children(n) == 1
                    n.log_probs .= 0.0
                elseif is⋁gate(n)
                    @assert num_children(n) == 2
                    p = rand(num_children(n)) .+ prior_vec
                    theta = LinearAlgebra.normalize(p, 1)
                    println(theta)
                    n.log_probs .= log.(theta)
                end
            end
        end
    end
    nothing
end

"""
Return CPT between Df, S, and D
"""
function cpt(fairpc::StructType)
    pc = fairpc.pc
    results = Dict()
    for k in CPT_HEADER
        results[k] = 0.0
    end

    if fairpc isa TwoNB
        PS = exp(pc.log_probs[1])
        sums = right_sums(pc)
        @assert length(sums) == 2
        PD1 = exp(sums[1].log_probs[1])
        PD2 = exp(sums[2].log_probs[1])
        results["P(S)"] = PS
        results["P(D)"] = PS * PD1 + (1 - PS) * PD2
        return results
    end
    
    thetas = exp.(pc.log_probs)
    p1, p2 = thetas[1] + thetas[2], thetas[1] + thetas[3]

    if fairpc isa LatentStructType
        log_PDs = map(nodes_D_sum(fairpc)) do pn
            pn.log_probs[1]
        end
        PD = sum(exp.(pc.log_probs) .* exp.(log_PDs))
        results["P(Df)"] = p1
        results["P(S)"] = p2
        results["P(D)"] = PD
        for (key, value) in zip(["P(D|Df,S)", "P(D|Df,notS)", "P(D|notDf,S)", "P(D|notDf,notS)"], exp.(log_PDs))
            results[key] = value
        end
    elseif fairpc isa NlatPC
        results["P(D)"] = p1
        results["P(S)"] = p2
    else
        error("")
    end
    results
end