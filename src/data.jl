using DataFrames
using CSV
using Random
using Missings
export DATASET_NAMES, DATASET2DECISION, FairDataset, convert2latent, convert2nonlatent,
data_without_latent, load_data, all_dataset_property, remove_variables, data_given_constraint, 
add_missing_Df_label, remove_synthetic_Df_label, remove_missing_Df_label, reset_end_missing, 
flip_coin, reset_end_two_missing, gen_synthetic_data

"""
List of data set names in current experiments
"""
const DATASET_NAMES = ["adult", "compas", "german", "synthetic"]

"""
Mapping from dataset names to corresponding decision variable
"""
const DATASET2DECISION = Dict("adult"=>"target", "compas"=>"ScoreText_", 
    "german"=>"loan_status", "synthetic"=>"D")

"""
Dataset
"""
mutable struct FairDataset
    name :: String
    data :: DataFrame
    header :: Vector{String}
    SV :: String

    S :: Var
    D :: Var 
    Df :: Union{Var, Nothing}
    D_label :: BitVector
    true_label :: Union{BitVector, Nothing}
    
    is_latent :: Bool
    FairDataset(name, data, header, SV, S, D, Df, D_label, true_label, is_latent) = 
        new(name, data, header, SV, S, D, Df, D_label, true_label, is_latent)
end

function FairDataset(T::Type{<:StructType}, name, header, data, SV)

    true_label = nothing
    if name == "synthetic"
        true_label = data[:, end]
        data = remove_synthetic_Df_label(data)
    end

    S = findall(v -> v== SV, header)[1]
    D = num_features(data)
    D_label = data[:, end]
    
    if T <: LatentStructType
        is_latent = true
        Df = D + 1
        data = add_missing_Df_label(data)
    elseif T <: NonLatentStructType
        is_latent = false
        Df = nothing
        data = DataFrame(BitArray(Tables.matrix(data)), :auto)
    end
    FairDataset(name, data, header, SV, S, D, Df, D_label, true_label, is_latent)
end

function convert2latent(fairdata::FairDataset)
    if fairdata.is_latent
        @assert num_features(fairdata.data) == fairdata.D + 1
        data = fairdata.data
    else
        @assert num_features(fairdata.data) == fairdata.D
        data = add_missing_Df_label(fairdata.data)
    end
    FairDataset(fairdata.name, data, fairdata.header, fairdata.SV, fairdata.S, 
        fairdata.D, fairdata.D + 1, fairdata.D_label, fairdata.true_label, true)
end

function convert2nonlatent(fairdata::FairDataset)
    if fairdata.is_latent
        @assert num_features(fairdata.data) == fairdata.D + 1
        data = remove_missing_Df_label(fairdata.data)        
    else
        @assert num_features(fairdata.data) == fairdata.D
        data = fairdata.data
    end
    FairDataset(fairdata.name, data, fairdata.header, fairdata.SV, fairdata.S, 
        fairdata.D, nothing, fairdata.D_label, fairdata.true_label, false)
end

convert2latent(x...) = convert2latent.(collect(x))

convert2nonlatent(x...) = convert2nonlatent.(collect(x))

function data_without_latent(fairdata::FairDataset)
    if fairdata.is_latent
        return remove_missing_Df_label(fairdata.data)
    else
        return fairdata.data
    end
end

"""
Load data from directory `data_dir`
Return header and data
"""
function load_data(name, T, SV; data_dir="./data/processed_data/", data_type=Bool, fold=1, num_X=nothing)
    @assert name in DATASET_NAMES "Dataset $name not found in directory $data_dir"
    id_dir = "./data/splited_data/10-cv/"

    if name == "synthetic"
        data_dir = "./data/synthetic_data"
        data_path = joinpath(data_dir, "$num_X.csv")
        filename = "$name/$num_X"
    else
        data_path = joinpath(data_dir, "$(name)_binerized.csv")
        filename = "$name"
    end

    dataframe = CSV.read(data_path, DataFrame; header=true, truestrings=["1"], falsestrings=["0"], types=data_type, strict=true)
    N = size(dataframe)[1]
    
    train_ids = collect(skipmissing(CSV.read(joinpath(id_dir, filename, "train_ids.csv"), DataFrame; header=true, strict=true)[:, fold]))
    valid_ids = collect(skipmissing(CSV.read(joinpath(id_dir, filename, "valid_ids.csv"), DataFrame; header=true, strict=true)[:, fold]))
    test_ids = collect(skipmissing(CSV.read(joinpath(id_dir, filename, "test_ids.csv"), DataFrame; header=true, strict=true)[:, fold]))

    @assert isempty(intersect(train_ids, valid_ids)) && isempty(intersect(train_ids, test_ids))  && isempty(intersect(test_ids, valid_ids)) 
    
    dataframes = dataframe[train_ids, :], dataframe[valid_ids, :], dataframe[test_ids, :]
    datas = [BitArray(Tables.matrix(dataframe)) for dataframe in dataframes]
    header = map(x -> String(x), names(dataframe))
    
    if name != "synthetic"
        D = findfirst(x -> x == DATASET2DECISION[name], header)
        header[D] = header[end]
        header[end] = DATASET2DECISION[name]
        datas2 = map(datas) do data
            x = copy(data)
            x[:,[D, end]] .= data[:, [end, D]]
            x
        end
    else
        datas2 = datas
    end
    train_x, valid_x, test_x = DataFrame(datas2[1], :auto), DataFrame(datas2[2], :auto), DataFrame(datas2[3], :auto)

    
    train_x = FairDataset(T, name, header, train_x, SV)
    valid_x = FairDataset(T, name, header, valid_x, SV)
    test_x = FairDataset(T, name, header, test_x, SV)
    train_x, valid_x, test_x
end

function load_data_test(name, T, SV; data_dir="./data/test_adult/fold-1", data_type=Bool, fold=1, num_X=nothing)
    println("Load data from test")
    @assert name in DATASET_NAMES "Dataset $name not found in directory $data_dir"


    # dataframe = CSV.read(data_path, DataFrame; header=true, truestrings=["1"], falsestrings=["0"], type=data_type, strict=true)
    # N = size(dataframe)[1]
    
    train_x = CSV.read(joinpath(data_dir, "adult.train.csv"), DataFrame; header=true, strict=true)
    valid_x = CSV.read(joinpath(data_dir, "adult.valid.csv"), DataFrame; header=true, strict=true)
    test_x = CSV.read(joinpath(data_dir, "adult.test.csv"), DataFrame; header=true, strict=true)

    # @assert isempty(intersect(train_ids, valid_ids)) && isempty(intersect(train_ids, test_ids))  && isempty(intersect(test_ids, valid_ids)) 
    
    dataframes = [train_x, valid_x, test_x] # dataframe[train_ids, :], dataframe[valid_ids, :], dataframe[test_ids, :]
    datas = [Base.convert(Matrix{data_type}, dataframe) for dataframe in dataframes]
    header = map(x -> String(x), names(train_x))
    
    if name != "synthetic"
        D = findfirst(x -> x == DATASET2DECISION[name], header)
        header[D] = header[end]
        header[end] = DATASET2DECISION[name]
        datas2 = map(datas) do data
            x = copy(data)
            x[:,[D, end]] .= data[:, [end, D]]
            x
        end
    else
        datas2 = datas
    end
    train_x, valid_x, test_x = DataFrame(datas2[1], :auto), DataFrame(datas2[2], :auto), DataFrame(datas2[3], :auto)

    train_x = FairDataset(T, name, header, train_x, SV)
    valid_x = FairDataset(T, name, header, valid_x, SV)
    test_x = FairDataset(T, name, header, test_x, SV)
    train_x, valid_x, test_x
end

"""
Generate synthetic data set
"""
function gen_synthetic_data(;outdir="./data/synthetic-test", num_X=2, num_samples=10_000,
    prob=Dict("Df"=>0.5, "S"=>0.3, 
    "D|Df,S"=>0.1, "D|Df,notS"=>0.1, "D|notDf,S"=>0.9, "D|notDf,notS"=>0.9))
    
    # gen circuit, only structure
    if !isdir(outdir)
        mkpath(outdir)
    end
    data = DataFrame(BitMatrix(rand(Bool, num_samples, num_X + 2)), :auto)
    SV = "S"
    header = [[SV]; ["X$i" for i in 1 : num_X]; ["D"]]
    T = FairPC
    fairdataset = FairDataset(T, "gen-synthetic", header, data, SV)
    fairpc = initial_structure(T, fairdataset; pseudocount=1.0)
    reset_prob(fairpc, prob;prior=0.5, init_X=true)
    pc = fairpc.pc

    # sample
    gen_data = DataFrame(hcat([sample(pc) for _ in 1 : num_samples]...)', :auto)
    CSV.write(joinpath(outdir, "$num_X.csv"), Int8.(gen_data); header=[header;["Df"]])
end

function gen_synthetic_data_all(num_Xs=10:30; seed=1337)
    Random.seed!(seed)
    for num_X in num_Xs
        gen_synthetic_data(num_X=num_X)
    end
end

function all_dataset_property()
    for (name, sv) in zip(["adult", "compas", "german"], ["sex",["Ethnic_Code_Text_", "Sex_"],"sex"])
        header, train_x0, valid_x0, test_x0 = load_data(name)
        if sv isa VecOrMat
            for s in sv
                log_dataset_property(name, header, train_x0, valid_x0, test_x0, s)
            end
        else
            log_dataset_property(name, header, train_x0, valid_x0, test_x0, sv)
        end
    end

end

"""
Remove sensitive variables in `sensitive_variables`
"""
function remove_variables(header::Vector{String}, data::DataFrame;
        SV, remove_D=false)
    m = copy(data)
    N = num_features(data)

    X_variables = filter(v -> header[v] != SV, 1 : N)
    if remove_D
        X_variables = X_variables[1:end-1]
    end
    X_m = m[:, X_variables]
    return Var.(X_variables), DataFrame(X_m)
end

"""
Return the subset of dataset `data` with the constraint that
the variable `variable` has value `value`
"""
function data_given_constraint(data::DataFrame; given=Dict(), type, keep_var=true, keep_Df=true)
    n = num_features(data)
    m = num_examples(data)
    mat = copy(Matrix(data))
    vars = trues(n)
    keep = trues(m)
    for (variable, value) in given
        for i in 1 : m
            if mat[i, variable] != value
                keep[i] = false
            end
        end
        if !keep_var
            vars[variable] = false
        end
    end
    if !keep_Df
        vars[end] = false
    end
    DataFrame(type.(mat[keep, vars]), :auto)
end

"""
Add Df = -1 to all examples
"""
function add_missing_Df_label(data::DataFrame)
    df = DataFrame(missings((num_examples(data), 1)), :auto)
    hcat(data, df;makeunique=true)
end

"""
Remove Df from synthetic generated data
"""
function remove_synthetic_Df_label(data::DataFrame)
    copy(data[:, 1:end-1])
end

function remove_missing_Df_label(data::DataFrame)
    copy(data[:, 1:end-1])
end

"""
Set the last column to -1, used in prediction
"""
function reset_end_missing(data::DataFrame)
    m = copy(Matrix{Union{Missing,Bool}}(data))
    m[:, end] .= missing
    DataFrame(m, :auto)
end


function reset_end_two_missing(data::DataFrame)
    m = copy(Matrix{Union{Missing,Bool}}(data))
    m[:, end-1] .= missing
    DataFrame(m, :auto)
end

"""
Flip the coin for every bit in the data set
"""
function flip_coin(data::DataFrame; keep_prob)::DataFrame
    m = copy(Matrix{Union{Missing,Bool}}(data))
    flag = rand(num_examples(data), num_features(data)) .<= 1 - keep_prob
    m[flag] .= missing
    DataFrame(m, :auto)
end

function flip_coin(T, data::FairDataset; keep_prob)::FairDataset
    data_missing = flip_coin(data.data::DataFrame; keep_prob=keep_prob)
    FairDataset(data.name, data_missing, data.header, data.SV, data.S, data.D, data.Df, data.D_label, data.true_label, data.is_latent)
end
