export initial_structure, structure_update
"""
Initialize structure given dataset
"""
function initial_structure(T::Type{<:StructType}, train_x::FairDataset; pseudocount)
    # init sub circuits
    data = data_without_latent(train_x)
    i2v, sub_train = remove_variables(train_x.header, data; SV=train_x.SV, remove_D=true)
    pcs, vtree = init_sub_circuits(T, sub_train, i2v; pseudocount=pseudocount)

    # add top layer 
    S = train_x.S
    @assert !(S in i2v)
    Df = train_x.is_latent ? Var(train_x.Df) : nothing
    pc, vtree = add_top_layer(T, pcs, vtree; S=Var(train_x.S), D=Var(train_x.D), Df=Df)
    fairpc = T(pc, vtree, train_x.S, train_x.D)
    fairpc
end

"""
Initialize FairPc structure from other structures
"""
function initial_structure(::Type{FairPC}, fairpc::StructType; init_alg)
    @assert fairpc isa NlatPC
    pc0 = fairpc.pc
    vtree = fairpc.vtree.right
    vtree.parent = nothing
    S = fairpc.S
    D = fairpc.D
    old_p = exp.(pc0.log_probs)

    # data
    pcs = non_sensitive_sub_circuits(fairpc)
    pc2, vtree2 = add_top_layer(FairPC, pcs, vtree;S=Var(S), D=Var(D), Df=Var(D+1))

    # reset_prob
    PS = old_p[1] + old_p[3]
    PD = old_p[1] + old_p[2]
    if init_alg == "prior-subop"
        prob=Dict("Df"=>PD, "S"=>PS, "D|Df,S"=>1.0, "D|Df,notS"=>1.0, "D|notDf,S"=>0.0, "D|notDf,notS"=>0.0)
    elseif init_alg == "rand"
        prob=Dict("Df"=>rand(), "S"=>rand(), "D|Df,S"=>rand(), "D|Df,notS"=>rand(),
        "D|notDf,S"=>rand(), "D|notDf,notS"=>rand())
    end
    fairpc = FairPC(pc2, vtree2, S, D)
    reset_prob(fairpc, prob;init_X=false)
    fairpc
end

"""
The  sub circuits about X or D / X
"""
function init_sub_circuits(T::Type{<:StructType}, sub_train::DataFrame, i2v; pseudocount::Float64)
    vtree = nothing
    pcs = Vector{StructProbCircuit}()

    for i in 1 : 4
        pc = nothing
        
        clt = learn_chow_liu_tree(sub_train)
        if isnothing(vtree) # all clts decorate on same vtree
            vtree = learn_vtree_from_clt(clt, vtree_mode="balanced")
        end

        if T in [FairPC, NlatPC]     
            lc = compile_sdd_from_clt(clt, vtree)            
        else #T in [LatNB, TwoNB]
            @assert T in [LatNB, TwoNB]
            lc = fully_factorized_circuit(StructLogicCircuit, vtree).children[1]
            # TODO rm bias terms
        end
        pc = ProbCircuit(lc)
        estimate_parameters!(pc, sub_train; pseudocount=pseudocount)
        
        replace_index_with_variables!(pc, i2v)
        push!(pcs, pc)
    end
    replace_index_with_variables!(vtree, i2v)
    pcs, vtree
end

"""
Add decision variable `D`, faired decision variable `Df`,
and sensitive variable `S` to circuit `pc` and vtree `vtree`
"""
function add_top_layer(::Type{<:LatentStructType}, pcs::Vector{<:StructProbCircuit}, vtree::PlainVtree; S::Var, D::Var, Df::Var)
    @assert num_variables(vtree) + 2 == D
    @assert D + 1 == Df
    @assert !(S in variables(vtree))
    @assert length(pcs) == 4

    # vtree
    v_D, v_Df, v_S = map(x -> PlainVtreeLeafNode(x), [D, Df, S])
    v_left, v_right = map(x -> PlainVtreeInnerNode(x...), [(v_Df, v_S), (v_D, vtree)])
    v_root = PlainVtreeInnerNode(v_left, v_right)
    vtree_new = v_root

    # SDD
    T = StructProbCircuit
    pos_n_D = compile(T, v_D, var2lit(D))
    neg_n_D = compile(T, v_D, - var2lit(D))
    pos_n_Df = compile(T, v_Df, var2lit(Df))
    neg_n_Df = compile(T, v_Df, - var2lit(Df))
    pos_n_S = compile(T, v_S, var2lit(S))
    neg_n_S = compile(T, v_S, - var2lit(S))

    ## right
    D_ors = map(i -> summate([pos_n_D, neg_n_D]; use_vtree=v_D), 1 : 4)
    foreach(x -> x.log_probs .= log(0.25), D_ors)
    right_ands = map(c -> multiply([c[1], c[2]]; use_vtree=v_right), zip(D_ors, pcs))
    right_ors = map(i -> summate([i]; use_vtree=v_right), right_ands)
    foreach(x -> x.log_probs .= 0.0, right_ors)

    ## left
    childrens = [[pos_n_Df, pos_n_S], [pos_n_Df, neg_n_S],
                 [neg_n_Df, pos_n_S], [neg_n_Df, neg_n_S]]
    left_ands = map(c -> multiply(c; use_vtree=v_left), childrens)
    left_ors = map(and -> summate([and]; use_vtree=v_left), left_ands)
    foreach(x -> x.log_probs .= 0.0, left_ors)

    ## root
    ands = map(or -> multiply(collect(or); use_vtree=v_root), zip(left_ors, right_ors))
    pc_root = summate(ands; use_vtree=v_root)
    pc_new = pc_root
    pc_root.log_probs .= log(0.25)

    return pc_new, vtree_new
end

function add_top_layer(::Type{NlatPC}, pcs::Vector{StructProbCircuit}, vtree::PlainVtree;S, D, Df=nothing)
    # @assert num_variables(vtree) + 2 == Df
    @assert !(S in variables(vtree))
    @assert length(pcs) == 4
    @assert Df == nothing

    # vtree
    v_D, v_S = map(x -> PlainVtreeLeafNode(x), [D, S])
    v_left = PlainVtreeInnerNode(v_D, v_S)
    v_root = PlainVtreeInnerNode(v_left, vtree)
    vtree_new = v_root

    # SDD
    T = StructProbCircuit
    pos_n_D = compile(T, v_D, var2lit(D))
    neg_n_D = compile(T, v_D, - var2lit(D))
    pos_n_S = compile(T, v_S, var2lit(S))
    neg_n_S = compile(T, v_S, - var2lit(S))

    ## right
    right_ors = pcs

    ## left
    childrens = [[pos_n_D, pos_n_S], [pos_n_D, neg_n_S],
                 [neg_n_D, pos_n_S], [neg_n_D, neg_n_S]]
    left_ands = map(c -> multiply(c; use_vtree=v_left), childrens)
    left_ors = map(and -> summate([and]; use_vtree=v_left), left_ands)
    @assert left_ors isa Vector
    foreach(x -> x.log_probs .= 0.0, left_ors)

    ## root
    ands = map(or -> multiply(collect(or); use_vtree=v_root), zip(left_ors, right_ors))
    pc_root = summate(ands; use_vtree=v_root)
    pc_new = pc_root
    pc_root.log_probs .= log(0.25)

    return pc_new, vtree_new
end

function add_top_layer(::Type{TwoNB}, pcs::Vector{StructProbCircuit}, vtree::PlainVtree;S, D, Df=nothing)
    # sanity check
    @assert !(S in variables(vtree))
    @assert length(pcs) == 4

    # vtree
    v_D, v_S = map(x -> PlainVtreeLeafNode(x), [D, S])
    v_right = PlainVtreeInnerNode(v_D, vtree)
    v_root = PlainVtreeInnerNode(v_S, v_right)
    vtree_new = v_root

    # SDD
    T = StructProbCircuit
    pos_n_D = compile(T, v_D, var2lit(D))
    neg_n_D = compile(T, v_D, - var2lit(D))
    pos_n_S = compile(T, v_S, var2lit(S))
    neg_n_S = compile(T, v_S, - var2lit(S))
    lits = [pos_n_D, neg_n_D, pos_n_S, neg_n_S]

     ## right
    D_ands = map(c -> multiply([c[1], c[2]]; use_vtree=v_right), zip([pos_n_D, neg_n_D, pos_n_D, neg_n_D], pcs))
    D_ors = map(i -> summate([D_ands[i], D_ands[i+1]]; use_vtree=v_right), [1, 3])
    foreach(x -> x.log_probs .= log(0.5), D_ors)

    ## left
    ands = map(c -> multiply(c; use_vtree=v_root), [[pos_n_S, D_ors[1]], [neg_n_S, D_ors[2]]])
    pc_root = summate(ands, use_vtree=v_root)
    pc_root.log_probs .= log(0.5)
    pc_new = pc_root

    
    return pc_new, vtree_new
end

"""
Replace the index in circuit `clt` and vtree `vtree` with variables,
according to mapping `i2v`
"""
function replace_index_with_variables!(vtree::PlainVtree, i2v::Vector{Var})
    # vtree
    foreach(vtree) do n
        if n isa PlainVtreeLeafNode
            n.var = i2v[n.var]
        else
            n.variables = variables(n.left) ∪ variables(n.right)
        end
    end
end

function replace_index_with_variables!(clt::ProbCircuit, i2v::Vector{Var})
    # (p)sdd
    foreach(clt) do n
        if isliteralgate(n)
            l = n.literal
            v = i2v[lit2var(l)]
            n.literal = l > 0 ? var2lit(v) : - var2lit(v)
        end
    end
end

using ProbabilisticCircuits: eFlow, vMI
"""
One structure update step 
"""
function split_step(fairpc::StructType, train_x::DataFrame; pick_edge="eFlow", pick_var="vMI", split_depth=1, pseudocount=1.0, sanity_check=true)
    pc = fairpc.pc
    unsplittable_sums(fpc::NlatPC) = [nodes_DS_sum(fpc); [fpc.pc]]
    unsplittable_sums(pc::StructType) = error("No structure learing when PC isa $(typeof(pc))")

    candidates, scope = split_candidates(pc)
    candidates = filter(x -> !(x[1] in unsplittable_sums(fairpc)), candidates)
    if isempty(candidates)
        return missing, true
    end

    values, flows, node2id = satisfies_flows(pc, train_x; weights = nothing)

    if isgpu(values)
        values = to_cpu(values)
    end
    if isgpu(flows)
        flows = to_cpu(flows)
    end
    
    if pick_edge == "eFlow"
        edge, flow = eFlow(values, flows, candidates, node2id)
    elseif pick_edge == "eRand"
        edge = eRand(candidates)
    else
        error("Heuristics $pick_edge to pick edge is undefined.")
    end

    or, and = edge
    vars = Var.(collect(scope[and]))

    if pick_var == "vMI"
        var, score = vMI(values, flows, edge, vars, train_x, node2id)
    elseif pick_var == "vRand"
        var = vRand(vars)
    end

    pc_new, _ = split(pc, edge, var; depth=split_depth, sanity_check=sanity_check)

    estimate_parameters!(pc_new, train_x; pseudocount=pseudocount)
    fairpc.pc = pc_new
    fairpc, false
end

function structure_update(pc::StructType, train_x::FairDataset; kwargs...)
    pc, stop = split_step(pc, train_x.data; kwargs...)
end

function sanity_check_pc(pc)
    foreach(pc) do n
        if is⋁gate(n)
            @assert(all(.!is⋁gate.(children(n))))
        elseif is⋀gate(n)
            @assert(all(.!is⋀gate.(children(n))))
        end
    end
end