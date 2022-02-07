using StatsFuns
using Statistics: mean
export predict_wrapper, cross_entropy, predict_all_circuits, predict_summary, prediction_per_example, 
ROC_curve, f1_score, discrimination_score, discrimination_score_from_predicted_label

"""
Predict Df, D given circuit `pc` and data `data`
"""
function predict_summary(predict_prob::Dict, threshold::Float64, data_str::String)
    header = ["dict-Df", "dict-D"]
    results = dict_init(header)
   
    function predict(P_n, actual_label)
        predicted = P_n .>= threshold
        dict = confusion_matrix(actual_label, predicted)
        dict["accuracy"] = accuracy(dict)
        dict
    end

    results["dict-D"] = predict(predict_prob["P(D|e) $data_str"], predict_prob["D $data_str"])
    
    if all(predict_prob["P(Df|e) $data_str"][1] isa Missing)
        results["dict-Df"] = deepcopy(results["dict-D"])
    else
        results["dict-Df"] = predict(predict_prob["P(Df|e) $data_str"], predict_prob["D $data_str"])
    end

    results
end

"""
Wrapper of prediction pipeline
"""
function predict_wrapper(pc::StructType, train_x::FairDataset, valid_x::FairDataset, test_x::FairDataset; outdir)
    if train_x.name == "synthetic" # fair label
        predict_prob = prediction_per_example(pc, train_x, valid_x, test_x; dir=outdir, use_fair_label=true)
        results = dict_init(PREDICT_HEADER)
        for (i, data_str) in enumerate(["train_x", "valid_x", "test_x"])
            p = predict_summary(predict_prob, 0.5, data_str)
            dict_push!(results, p["dict-Df"], p["dict-D"],
                Dict("dataset"=>data_str), Dict("dataset"=>data_str),
                Dict("variable"=>"Df"), Dict("variable"=>"D"))
        end
        save_as_csv(results; filename=joinpath(outdir, "predict-summary-fair-label.csv"), header=PREDICT_HEADER)
    end

    # proxy label
    predict_prob = prediction_per_example(pc, train_x, valid_x, test_x; dir=outdir)
    results = dict_init(PREDICT_HEADER)
    for (i, data_str) in enumerate(["train_x", "valid_x", "test_x"])
        p = predict_summary(predict_prob, 0.5, data_str)
        dict_push!(results, p["dict-Df"], p["dict-D"],
            Dict("dataset"=>data_str), Dict("dataset"=>data_str),
            Dict("variable"=>"Df"), Dict("variable"=>"D"))
    end
    save_as_csv(results; filename=joinpath(outdir, "predict-summary-proxy-label.csv"), header=PREDICT_HEADER)
end

"""
Plot ROC curve
"""
function ROC_curve(predict_prob::Dict; dir)
    ROC_Df, ROC_D = [], []
    δd = 0.01
    for threshold in 0 : δd : 1.0
        p = predict_summary(predict_prob, threshold, "valid_x")
        dict_Df = p["dict-Df"]
        dict_D = p["dict-D"]
        push!(ROC_Df, (threshold, accuracy(dict_Df), true_positive(dict_Df), false_positive(dict_Df), dict_Df))
        push!(ROC_D, (threshold, accuracy(dict_D), true_positive(dict_D), false_positive(dict_D), dict_D))
    end
    sort!(ROC_Df, by = x -> x[4])
    sort!(ROC_D, by = x -> x[4])
    plot_ROC(map(x -> x[3], ROC_D), map(x -> x[4], ROC_D), dir; filename="ROC-D.pdf", δd=δd)
    plot_ROC(map(x -> x[3], ROC_Df), map(x -> x[4], ROC_Df), dir; filename="ROC-Df.pdf", δd=δd)
end

function plot_ROC(TPrate, FPrate, dir; filename=".pdf", δd)
    results = Dict("False positive rate"=>FPrate, "True positive rate"=>TPrate)
    plot_opts = Dict("xs"=>"False positive rate", "ys"=>["True positive rate"], "filename"=>joinpath(dir, "ROC-$filename"), "title"=>"ROC")
    plot_dict(results; opts=plot_opts)

    results = Dict("False positive rate"=>FPrate, "Accumulate true positive rate"=>integral(FPrate, TPrate))
    plot_opts = Dict("xs"=>"False positive rate", "ys"=>["Accumulate true positive rate"], "filename"=>joinpath(dir, "AUC-$filename"), "title"=>"AUC")
    plot_dict(results; opts=plot_opts)

end

"""
Save prediction results for every example
"""
function prediction_per_example(fairpc::StructType, fairdata::FairDataset; use_fair_label=false)
    @inline get_node_id(id::⋁NodeIds) = id.node_id
    @inline get_node_id(id::⋀NodeIds) = @assert false

    results = Dict()
    header = fairdata.header
    SV = fairdata.SV
    data = fairdata.data

    if fairdata.name == "synthetic" && use_fair_label
        actual_label = fairdata.true_label
    else
        actual_label = copy(fairdata.D_label)
    end
    
    S = fairpc.S
    sensitive_label = data[:, S]

    data = reset_end_missing(data)
    if fairpc isa LatentStructType
        data = reset_end_two_missing(data)
    end

    _, flows, node2id = marginal_flows(fairpc.pc, data)
    # compute_exp_flows(fairpc.pc, data)

    # P_D
    D = get_node_id(node2id[node_D(fairpc)])
    n_D = get_node_id(node2id[node_not_D(fairpc)])
    P_D = exp.(flows[:, D])
    # @assert all(flows[:, D] .<= 0.0)
    @assert all(P_D .+ exp.(flows[:, n_D]) .≈ 1.0)
    P_D = min.(1.0, P_D)


    if fairpc isa LatentStructType
        Df = get_node_id(node2id[node_Df(fairpc)])
        n_Df = get_node_id(node2id[node_not_Df(fairpc)])
        P_Df = exp.(flows[:, Df])
        @assert all(P_Df .+ exp.(flows[:, n_Df]) .≈ 1.0)
        # @assert all(flows[:, Df] .<= 0.0)
        P_Df = min.(1.0, P_Df)    
    else
        P_Df = Vector{Missing}(missing, length(P_D))
    end

    results["P(Df|e)"] = P_Df
    results["P(D|e)"] = P_D
    results["D"] = Int8.(actual_label)
    results["S"] = Int8.(sensitive_label)
    return results
end

function prediction_per_example(fairpc::StructType, train_x::FairDataset, valid_x::FairDataset, test_x::FairDataset; dir=nothing, use_fair_label=false)
    header = train_x.header
    SV = train_x.SV

    dict_header = PREDICT_EXAMPLE_HEADER
    length = maximum(num_examples.([train_x.data, valid_x.data, test_x.data]))
    results = dict_init(dict_header; length=length)

    for (fairdata, data_str) in zip([train_x, valid_x, test_x], ["train_x", "valid_x", "test_x"])
        data = fairdata.data
        run_results = prediction_per_example(fairpc, fairdata, use_fair_label=use_fair_label)
        results["P(Df|e) $data_str"] = run_results["P(Df|e)"]
        results["P(D|e) $data_str"] = run_results["P(D|e)"]
        results["D $data_str"] = run_results["D"]
        results["S $data_str"] = run_results["S"]
    end

    if !isnothing(dir)
        if use_fair_label
            filename =  "predict-per-example-fair-label.csv"
        else
            filename =  "predict-per-example-proxy-label.csv"
        end
        save_as_csv(results; filename=joinpath(dir, filename), header=dict_header, diff_length=true)
    end
    return results
end

function integral(x::Vector, y::Vector)
    cnt = [0.0]
    @assert length(x) == length(y)
    for i in 1 : (length(x) - 1)
        push!(cnt, (y[i] + y[i+1]) * (x[i+1] - x[i]) / 2)
    end
    accumulate(+, cnt)
end

"""
Cross entropy between actual label and predicted label
"""
function cross_entropy(actual, predicted)
    actual = Float64.(actual)
    predicted = Float64.(predicted)
    @assert length(actual) == length(predicted)
    @assert all(predicted .> 0)
    @assert all(predicted .<= 1)
    CE = - sum(xlogy.(actual, predicted) + xlogy.(1 .- actual, 1 .- predicted)) / length(actual)
end

"""
Predictions for all learned circuits and save resutls to files
"""
function predict_all_circuits(T, result_circuits, log_opts, train_x, valid_x, test_x)
    for (key, value) in result_circuits
        dir = joinpath(log_opts["outdir"], key)
        (pc, vtree) = value
        if !isdir(dir)
            mkpath(dir)
        end
        run_fairpc = T(pc, vtree, train_x.S, train_x.D)
        predict_wrapper(run_fairpc, train_x, valid_x, test_x; outdir=dir)
    end
end

"""
Return confusion matrix given predicted label and actual label
"""
function confusion_matrix(actual, predicted)
    actual = Bool.(actual)
    predicted = Bool.(predicted)
    TP = sum(actual .& predicted)
    TN = sum(.!actual .& .!predicted)
    FP = sum(.!actual .& predicted)
    FN = sum(actual .& .!predicted)
    Dict{Any, Any}("TP"=>TP, "TN"=>TN, "FP"=>FP, "FN"=>FN)
end

function discrimination_score(prob::Vector{<:AbstractFloat}, S_label::Union{Vector{Bool}, BitVector})
    @assert length(prob) == length(S_label)
    mean(prob[S_label]) - mean(prob[.!S_label])
end

function discrimination_score_from_predicted_label(prob::Vector{<:AbstractFloat}, S_label::Union{Vector{Bool}, BitVector})
    @assert length(prob) == length(S_label)
    predicted = Bool.(prob .> 0.5)
    sum(predicted .& S_label) / sum(S_label) - sum(predicted .& .!S_label) / sum(.!S_label)
end

accuracy(dict::Dict) = (dict["TP"] + dict["TN"]) / (dict["TP"] + dict["TN"] + dict["FP"] + dict["FN"])
true_positive(dict::Dict) = (dict["TP"]) / (dict["TP"] + dict["FN"])
false_positive(dict::Dict) = (dict["FP"]) / (dict["FP"] + dict["TN"])
true_negative(dict::Dict) = (dict["TN"]) / (dict["FP"] + dict["TN"])
false_negative(dict::Dict) = (dict["FN"]) / (dict["TP"] + dict["FN"])
f1_score(dict::Dict) = begin
    tp = true_positive(dict)
    fp = false_positive(dict)
    fn = false_negative(dict)
    tp / (tp + 0.5 * (fp + fn))
end