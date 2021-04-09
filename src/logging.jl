using Plots
using Statistics
using Logging
using Suppressor
export OPTS, TRAIN_HEADER, CPT_HEADER, PREDICT_HEADER, PREDICT_EXAMPLE_HEADER, arg2str,
log_dataset_property, read_dict_from_csv, save_as_csv, plot_dict, scatter_dict4, 
dict_init, dict_append!, dict_push!, dict_avg, log_init, log_per_iter, reload_learned_pc

"""
Useful constants for logging
"""
OPTS = (
    DEFAULT_LOG_OPT=Dict("valid_x"=>nothing, "test_x"=>nothing, "outdir"=>"", "save"=>1, "print"=>""),
    DEFAULT_SPECIFIC_OPT=Dict("CPT"=>nothing))
TRAIN_HEADER = ["iter", "time", "total_time", "circuit_size", "cross_entropy_D", "cross_entropy_Df", "train_discrimination", "valid_discrimination", "test_discrimination", "train_mll", "valid_mll", "test_mll", "struct?", "stop?"]
CPT_HEADER = ["P(D)", "P(Df)", "P(S)", "P(D|Df,S)", "P(D|Df,notS)", "P(D|notDf,S)", "P(D|notDf,notS)"]
PREDICT_HEADER = ["dataset", "variable", "TP", "TN", "FP", "FN", "accuracy"]
PREDICT_EXAMPLE_HEADER = ["S train_x", "D train_x", "P(Df|e) train_x", "P(D|e) train_x", "S valid_x", "D valid_x", "P(Df|e) valid_x", "P(D|e) valid_x", "S test_x", "D test_x", "P(Df|e) test_x", "P(D|e) test_x"]

"""
"""
function arg2str(args)
    out = @capture_out foreach(x -> println(x[1], " => ", x[2]), args)
end

"""
"""
function log_dataset_property(train_x, valid_x, test_x)
    function dataset_property(train_x)
        m = train_x.data
        N = num_examples(train_x.data)
        results = Dict()
        D = Bool.(m[:, train_x.D])
        S = Bool.(m[:, train_x.S])
        results["P(D)"] = sum(D) / N
        results["P(S)"] = sum(S) / N
        PDS = sum(D .& S) / N
        PDnotS = sum(D .& .!S) / N
        results["P(D|S)"] = PDS / results["P(S)"]
        results["P(D|notS)"] = PDnotS / (1 - results["P(S)"])
        results
    end

    r1 = dataset_property(train_x)
    r2 = dataset_property(valid_x)
    r3 = dataset_property(test_x)

    # @info "Dataset name: $(train_x.name)\n"
    # @info "Num variables : $(num_features(train_x.data)) $(train_x.header)"
    # @info "Sensitive variable: $(train_x.SV)"
    # @info "Dataset size: Train $(num_examples(train_x.data)), 
    #     Valid $(num_examples(valid_x.data)), Test $(num_examples(test_x.data))"
    # @info "Train set: \n$(arg2str(r1))"
    # @info "Valid set: \n$(arg2str(r2))"
    # @info "Test set: \n$(arg2str(r3))"
end

"""
Read CSV file, return a dictionary
"""
function read_dict_from_csv(filename; labels=nothing, delim=",")
    dataframe = CSV.read(filename; header=true, delim=delim)
    header = map(x -> String(x), names(dataframe))
    result = Dict()
    if isnothing(labels)
        labels = header
    end
    for label in labels
        @assert label in header
        result[label] = convert(Vector, dataframe[!, Symbol(label)])
    end
    result
end

"""
Save dictionary as csv
"""
function save_as_csv(dict; filename, header=keys(dict), diff_length=false)
    if diff_length
        maxlength = maximum([length(dict[key]) for key in header])
        tmp_dict = dict_init(header; length=maxlength)
        for key in header
            tmp_dict[key][1:length(dict[key])] .= dict[key]
        end
    else
        tmp_dict = dict
    end
    table = DataFrame(;[Symbol(x) => tmp_dict[x] for x in header]...)
    CSV.write(filename, table; )
    table
end

"""
Plot dictionary
"""
function plot_dict(dict; opts, debug=false, seriestype = :line)

    # gr(size=(1000,1000),legend=false,markerstrokewidth=0,markersize=30)


    ys = dict[opts["ys"][1]]
    xs = !haskey(opts, "xs") ? (1 : length(ys)) : dict[opts["xs"]]
    x = nothing
    if xs[1] isa Vector
        x = xs[1]
    else
        x = xs
    end
    xlabel = !haskey(opts, "xs") ? nothing : opts["xs"]
    p = plot(x, ys,  seriestype=seriestype, xlabel=xlabel, markersize=3, markershape=:circle, markerstrokecolor=:auto, markeralpha = 0.2, label=opts["ys"][1], title=opts["title"])

    if debug
        println(xs)
        println(ys)
    end

    for i in 2: length(opts["ys"])
        if xs[1] isa Vector
            x = xs[i]
        else
            x = xs
        end
        y = opts["ys"][i]
        p = plot!(x, dict[y], seriestype=seriestype, markersize=3, markershape=:circle, markerstrokecolor=:auto, markeralpha = 0.2, label=y)
        if debug
            println(dict[y])
        end
    end

    if haskey(opts, "filename")
        savefig(p, opts["filename"])
    end
    if debug
        # @info "Success"
    end
    # @info "Plot $opts"
end

function scatter_dict4(dict; opts, debug=false, seriestype = :line)

    # gr(size=(1000,1000),legend=false,markerstrokewidth=0,markersize=30)
    xs = !haskey(opts, "xs") ? (1 : length(ys)) : dict[opts["xs"]]
    ylabel = opts["ylabel"]
    xlabel = !haskey(opts, "xs") ? nothing : opts["xs"]
    ps = []
    @info xs
    for i in 1 : length(opts["ys"])
        @info i 
        x = xs[i]
        y = opts["ys"][i]
        push!(ps,plot(x, dict[y], seriestype=seriestype, xlabel=xlabel, ylabel=ylabel,
            markersize=2, markershape=:circle, markerstrokecolor=:auto, markeralpha = 0.5, label=y, title=opts["title"]))
    end
    @info "start"

    p = plot(ps[1], ps[2], ps[3], ps[4], layout=(2,2), title=opts["title"])
    if haskey(opts, "filename")
        savefig(p, opts["filename"])
    end
    if debug
        # @info "Success"
    end
    @info "Plot $opts Success!"
end

"""
Initialize dictionary with empty contents
"""
function dict_init(header;length=0)
    results = Dict()
    for x in header
        results[x] = Vector{Union{Any,Missing}}(missing, length)
    end
    results
end

"""
"""
function dict_append!(dst::Dict, src::Dict...)
    for dict in src
        for (k, v) in dict
            append!(dst[k], v)
        end
    end
    dst
end

function dict_push!(dst::Dict, src::Dict...)
    for dict in src
        for (k, v) in dict
            push!(dst[k], v)
        end
    end
    dst
end

function dict_avg(src; ignore=["fold", "seed"],
    avg=["test_mll", "accuracy(Df)", "TP(Df)", "TN(Df)", "FP(Df)", "FN(Df)", "accuracy(D)", "TP(D)", "TN(D)", "FP(D)", "FN(D)"],
    same=["dataset", "sensitive_variable", "missing_perct", "batch_size", "init_alg"])
    avg_std = ["$x-std" for x in avg]
    results = dict_init([avg; avg_std; same])

    for x in same
        for y in src[x]
            @assert y==src[x][1] "$y $(src[x])"
        end
        push!(results[x], src[x][1])
    end

    for x in avg
        if !isempty(src[x])
            push!(results[x], mean(src[x]))
            push!(results["$x-std"], std(src[x]))
        else
            push!(results[x], missing)
        end
    end
    results

end

"""
Initialize log dict
"""
function log_init(;opts, specific_opts=true) # TODO later
    results = dict_init(TRAIN_HEADER)
    if issomething(specific_opts)
        merge!(results, dict_init(CPT_HEADER))
    end
    results
end

"""
Every log step
"""
function log_per_iter(pc::StructType, data::FairDataset, results;
    opts, time, iter=nothing, save_pdf=true, save_sub=true)
    # from kwargs
    continue_flag = true
    vtree = pc.vtree
    if isnothing(iter)
        iter = length(results["iter"]) + 1
    end

    push!(results["iter"], iter)
    push!(results["time"], time)
    push!(results["struct?"], opts["learn_mode"] != "para")
    push!(results["circuit_size"], num_nodes(pc.pc))
    if length(results["total_time"]) == 0
        push!(results["total_time"], time)
    else
        push!(results["total_time"], time + results["total_time"][end])
    end
    push!(results["stop?"], missing)

    # cross entropy
    p = prediction_per_example(pc, data)
    if pc isa LatentStructType
        cedf = cross_entropy(p["D"], p["P(Df|e)"])
    else
        cedf = cross_entropy(p["D"], p["P(D|e)"])
    end
    ced = cross_entropy(p["D"], p["P(D|e)"])
    push!(results["cross_entropy_D"], ced)
    push!(results["cross_entropy_Df"], cedf)

    if pc isa LatentStructType
        decision = Bool.(p["P(Df|e)"] .> 0.5)
    else
        decision = Bool.(p["P(D|e)"] .> 0.5)
    end
    # discrimination score
    if !haskey(opts, "missing") || !opts["missing"]
        for data_str in ["train", "valid", "test"]
            p = prediction_per_example(pc, opts["$(data_str)_x"]) 
            prob = pc isa LatentStructType ? p["P(Df|e)"] : p["P(D|e)"]
            s_label = Bool.(opts["$(data_str)_x"].data[:, pc.S])
            @assert pc.S == data.S
            push!(results["$(data_str)_discrimination"], discrimination_score(prob, s_label))
        end
    else
        for data_str in ["train", "valid", "test"]
            push!(results["$(data_str)_discrimination"], 0.0)
        end
    end
    # ll
    if pc isa LatentStructType
        train_ll = marginal_log_likelihood_per_instance(pc.pc, opts["train_x"].data)
        valid_ll = marginal_log_likelihood_per_instance(pc.pc, opts["valid_x"].data)
        test_ll = marginal_log_likelihood_per_instance(pc.pc, opts["test_x"].data)
    else
        @assert pc isa NonLatentStructType
        train_ll = log_likelihood_per_instance(pc.pc, opts["train_x"].data)
        valid_ll = log_likelihood_per_instance(pc.pc, opts["valid_x"].data)
        test_ll = log_likelihood_per_instance(pc.pc, opts["test_x"].data)
    end

    push!(results["train_mll"], mean(train_ll))
    push!(results["valid_mll"], mean(valid_ll))
    push!(results["test_mll"], mean(test_ll))

    # CPT
    CPT = cpt(pc)
    for (key, value) in CPT
        push!(results[key], value)
    end


    # save circuit
    save_last = opts["learn_mode"] == "para"
    name = data.name
    if save_sub
        continue_flag = save_with_best_valid_ll(results, pc;outdir=opts["outdir"], patience=opts["patience"], save_last=save_last, name=name)
        # save_with_best_ce(results, pc; outdir=opts["outdir"],name=name)
        # save_with_best_dis_score(results, pc; outdir=opts["outdir"], name=name)
    else
        save_pc(results, pc; outdir=opts["outdir"], name=name)
        continue_flag = true
    end

    # save to csv, pdf
    if !continue_flag || (haskey(opts, "save") && (iter % opts["save"] == 0))
        save_as_csv(results; filename=joinpath(opts["outdir"], "progress.csv"), header=[TRAIN_HEADER; CPT_HEADER])

        if save_pdf
            ys = ["train_mll"]
            if issomething(opts["valid_x"])
                push!(ys, "valid_mll")
            end
            if issomething(opts["test_x"])
                push!(ys, "test_mll")
            end

            plot_opts = Dict("xs"=>"iter", "ys"=>ys, "filename"=>joinpath(opts["outdir"], "train-curve.pdf"),
                "title"=>"Marginal log likelihood wrt iterations")
            plot_dict(results; opts=plot_opts)

            plot_opts = Dict("xs"=>"iter", "ys"=>CPT_HEADER, "filename"=>joinpath(opts["outdir"], "probability-curve.pdf"),
                "title"=>"CPT wrt iterations")
            plot_dict(results; opts=plot_opts)

            plot_opts = Dict("xs"=>"iter", "ys"=>["cross_entropy_D", "cross_entropy_Df"], "filename"=>joinpath(opts["outdir"], "cross-entropy-curve.pdf"),
                "title"=>"Cross entropy wrt iterations")
            plot_dict(results; opts=plot_opts)

            ys = ["$(data_str)_discrimination" for data_str in ["train", "valid", "test"]]
            plot_opts = Dict("xs"=>"iter", "ys"=>ys, "filename"=>joinpath(opts["outdir"], "discrimination-score.pdf"),
                "title"=>"Discrimination score wrt iterations")
            plot_dict(results; opts=plot_opts)
        end
    end

    # print for debug
    if haskey(opts, "print")
        println("Marginall log likelihood is $(mean(train_ll))")
    end

    return continue_flag
end

# function save_force(results::Dict, pc::ProbΔ, vtree, name; outdir, dirname)
#     results[dirname] = Dict()
#     results[dirname]["pc"] = pc
#     results[dirname]["vtree"] = vtree
#     results[dirname]["id"] = length(results["iter"])
#     if !isdir(joinpath(outdir, dirname))
#         mkpath(joinpath(outdir, dirname))
#     end
#     save_circuit(joinpath(outdir, dirname, "$name.psdd"), pc, vtree)
#     save(vtree, joinpath(outdir, dirname, "$name.vtree"))
# end

function save_with_best_dis_score(results::Dict, pc::StructType; outdir, name)
    save_with_best(results, "discrimination", "min-dis", pc; f=(x -> findmin(abs.(x))), outdir=outdir, name=name)
end

function save_with_best_valid_ll(results::Dict, pc::StructType; outdir, patience, save_last=true, name)
    continue_flag = true
    f = nothing
    cur_iter = length(results["iter"])
    if  save_last# save the last one
        f = x -> (x[end], length(x))
    else # save the best after min_iters
        f = findmax
    end

    best_id, cur_id = save_with_best(results, "valid_mll", "max-ll", pc; f=f, outdir=outdir, stop=true, name=name)

    @assert cur_id == cur_iter
    if cur_id - best_id > patience
        continue_flag = false
    end

    if cur_iter > patience
        best_v = results["valid_mll"][best_id]
        cur_v = results["valid_mll"][cur_id]
        tmp_v = results["valid_mll"][cur_id - patience]
        if abs(cur_v - tmp_v) < 1e-4
            continue_flag = false
        end
    end
    return continue_flag
end

function save_with_best_ce(results::Dict, pc::StructType; outdir, name)
    save_with_best(results, "cross_entropy_Df", "min-ce", pc; f=findmax, outdir=outdir, name=name)
end

function save_with_best(results, key, dirname, fairpc::StructType; f=findmax, outdir="", stop=false, name)
    pc = fairpc.pc
    vtree = fairpc.vtree
    if !haskey(results, dirname)
        results[dirname] = Dict()
        last_best_id = 0
        if !isdir(joinpath(outdir, dirname))
            mkpath(joinpath(outdir, dirname))
        end
    else
        last_best_id = results[dirname]["id"]
    end

    v, best_id = f(results[key])
    cur_id = length(results[key])

    if last_best_id != best_id
        @assert best_id == cur_id "$last_best_id $best_id $cur_id"
        if stop
            results["stop?"] .= missing
            results["stop?"][cur_id] = true
        end
        results[dirname]["pc"] = pc
        results[dirname]["vtree"] = vtree
        results[dirname]["id"] = best_id
        save_circuit(joinpath(outdir, dirname, "$name.psdd"), pc, vtree)
        save_vtree(vtree, joinpath(outdir, dirname, "$name.vtree"))
    end
    best_id, cur_id
end

function save_pc(results::Dict, pc::StructType; outdir, name)
    if !isdir(outdir)
        mkpath(outdir)
    end
    save_circuit(joinpath(outdir, "$name.psdd"), pc.pc, pc.vtree)
    save(pc.vtree, joinpath(outdir, "$name.vtree"))
end
function reload_learned_pc(results, key; opts, name)
    _, _, id = results[key]
    pc, vtree = load_struct_prob_circuit(joinpath(opts["outdir"], key, "$name.psdd"), joinpath(opts["outdir"], key, "$name.vtree"))
    # @info "Pick the circuit with best $key"
    # @info id
    pc, vtree
end
