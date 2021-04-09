using ArgParse
using CSV
using DataFrames
using Random
using MLBase

function parse_cmd_line()
    s = ArgParseSettings()
    @add_arg_table! s begin
    "dataset"
        help = "Dataset name"
        arg_type = String
        required = true
    "--input"
        help = "Input file name"
        arg_type = String
    "--outdir"
        help = "Output directory to splited datas"
        arg_type = String
    "--perct"
        help = "Percentage of split"
        default = [0.75, 0.10, 0.15]
        arg_type = Vector{Float64}
    "--seed"
        help = "Random seed"
        default = 1337
        arg_type = Int64
    "--fold"
        help = "k-fold cross validation"
        default = 10
        arg_type = Int64
    end
    return parse_args(s)
end
dataset2decision = Dict("adult"=>"target", "compas"=>"ScoreText_", "german"=>"loan_status", "synthetic"=>"D")

function main()
    args = parse_cmd_line()
    name = args["dataset"]
    Random.seed!(args["seed"])

    fold = args["fold"]
    outpath = "./data/splited_data/10-cv/$name/"
    if name == "synthetic"
        num_X = basename(args["input"])[1:2]
        outpath = "./data/splited_data/10-cv/synthetic/$num_X/"
    end

    if !isdir(outpath)
        mkpath(outpath)
    end

    # read data
    dataframe = CSV.read(args["input"];header=true, type=Bool, truestrings=["1"], falsestrings=["0"], strict=true)
    decision = dataframe[!, dataset2decision[name]]

    # split to 10 fold
    N = size(dataframe)[1]
    train_ids = collect(StratifiedKfold(decision, fold))
    test_ids = map(train_ids) do train_id
        [i for i in 1 : N if !(i in train_id)]
    end

    # valid ids for each fold
    train_ids2 = []
    valid_ids = []
    for fold_i in 1 : fold
        decision_i = decision[train_ids[fold_i]]
        train_i_id = train_ids[fold_i][collect(StratifiedKfold(decision_i, fold))[1]]
        valid_i_id = [i for i in train_ids[fold_i] if !(i in train_i_id)]
        push!(train_ids2, train_i_id)
        push!(valid_ids, valid_i_id)
    end

    # sanity check 
    for fold_i in 1 : fold
        id1, id2, id3 = train_ids2[fold_i], valid_ids[fold_i], test_ids[fold_i]
        @assert isempty(intersect(id1, id2)) && isempty(intersect(id2, id3))  && isempty(intersect(id1, id3)) "$fold_i"
        @assert length(id1) + length(id2) + length(id3) == N 
    end

    # write to file
    for (ids, filename) in zip([train_ids2, valid_ids, test_ids], ["train_ids.csv", "valid_ids.csv", "test_ids.csv"])
        maxlength = maximum(length.(ids))
        for i in 1 : fold
            ids[i] = Vector{Any}(ids[i])
            len = maxlength - length(ids[i])
            append!(ids[i], missings(len))
        end
        CSV.write(joinpath(outpath, filename), DataFrame(hcat(ids...)))
    end
    # N = size(dataframe)[1]
    # p_test = args["perct"][3]
    # test_ids = collect(StratifiedRandomSub(decision, Int(floor(N * p_test)), fold))
    # df = convert(DataFrame, hcat(test_ids...))
    # CSV.write(outpath * "testsplit.csv", df)
    # # train_ids = collect(StratifiedKfold(decision, fold))
    # train_ids = map(test_ids) do test_id
    #      [i for i in 1 : N if !(i in test_id)]
    # end

    # valid_ids = []
    # p_train = args["perct"][1]
    # for i in 1 : fold
    #     train_id = train_ids[i]
    #     train_tmp = train_id[collect(StratifiedRandomSub(decision[train_id], Int(floor(N * p_train)), 1))[1]]
    #     valid_id = [i for i in train_id if !(i in train_tmp)]
    #     push!(valid_ids, valid_id)
    #     train_ids[i] = train_tmp
    # end

    # df = convert(DataFrame, hcat(train_ids...))
    # CSV.write(outpath * "trainsplit.csv", df)

    # df = convert(DataFrame, hcat(valid_ids...))
    # CSV.write(outpath * "validsplit.csv", df)

    # save
    # for i in 1 : fold
    #     subdir = joinpath(args["outdir"], "fold-$i")
    #     if !isdir(subdir)
    #         mkdir(subdir)
    #     end
    #     test_id = test_ids[i]
    #     valid_id = valid_ids[i]
    #     train_id = train_ids[i]
    #     @assert all(sort([train_id;valid_id;test_id]) .== collect(1:N)) "$name"
    #     dataframes = [dataframe[id, :] for id in [train_id, valid_id, test_id]]
    #     for (type, data) in zip(["train", "valid", "test"], dataframes)
    #         CSV.write(joinpath(subdir, "$(args["dataset"]).$type.csv"), data;
    #         transform=(col, val) -> val ? "1" : "0")
    #     end
    # end
    # split
    # perct = args["perct"]
    # @assert isapprox(sum(perct), 1.0; atol=1e-6)
    # @assert length(perct) == 3
    # num = size(dataframe)[1]
    # rand_num = rand(num)
    # train_id, valid_id, test_id = [], [], []
    # for (i, p) in enumerate(rand_num)
    #     if p < perct[1]
    #         push!(train_id, i)
    #     elseif p < perct[1] + perct[2]
    #         push!(valid_id, i)
    #     else
    #         push!(test_id, i)
    #     end
    # end

    #num_examples =
    # ids = 1 : num_examples
    # ids_per_fold = [[] for i in 1 : fold]
    # for id in 1: num_examples
    #     push!(ids_per_fold[id % fold + 1], id)
    # end
    #
    # for i in 1 : fold
    #     subdir = joinpath(args["outdir"], "fold-$i")
    #     if !isdir(subdir)
    #         mkdir(subdir)
    #     end
    #     test_id = ids_per_fold[i]
    #     train_id = [id for id in 1 : num_examples if !(id in test_id)]
    #     valid_id = []
    #     @assert length(train_id) + length(valid_id) + length(test_id) == num_examples
    #     dataframes = [dataframe[id, :] for id in [train_id, valid_id, test_id]]
    #     for (type, data) in zip(["train", "valid", "test"], dataframes)
    #         CSV.write(joinpath(subdir, "$(args["dataset"]).$type.csv"), data;
    #         transform=(col, val) -> val ? "1" : "0")
    #     end
    # end

    # save splitted file

end

function shuffle_data(name; inpath="data/processed_data/adult_binerized.csv", seed=1337, outdir="data/shuffled_data/")
    Random.seed!(seed)
    dataframe = CSV.read(inpath;header=true, type=Bool, truestrings=["1"], falsestrings=["0"], strict=true)
    ids = shuffle(1 : size(dataframe)[1])
    df = dataframe[ids, :]
    CSV.write(joinpath(outdir, "$(name)_binerized.csv"), Int8.(df))
end
main()


