using LinearAlgebra
using Statistics
using Random
using Dates

export model_fair_psdd, train_fair_psdd, learn, learn_test, fair_pc_para_learn_from_file

function model_fair_psdd(::Type{FairPC}, train_x::FairDataset, valid_x::FairDataset, test_x::FairDataset;
                            struct_iters,
                            split_heuristic,
                            split_depth,
                            init_para_alg,
                            para_iters,
                            pseudocount,
                            log_opts)
    dir = log_opts["outdir"]
    # struct learn
    train_x1, valid_x1, test_x1 = convert2nonlatent(train_x, valid_x, test_x)
    log_opts1=Dict("train_x"=>train_x1, "valid_x"=>valid_x1, "test_x"=>test_x1,
        "outdir"=>joinpath(dir, "struct"), "save"=>log_opts["save"], "patience"=>log_opts["patience"],
        "learn_mode"=>"struct")
    
    nlat_fairpc = initial_structure(NlatPC, train_x1; pseudocount=pseudocount)
    nlat_result_circuits, nlat_results = train_fair_psdd(nlat_fairpc, train_x1, "struct";
        max_iters=struct_iters,
        pseudocount=pseudocount, 
        init_para_alg="estimate",
        split_heuristic=split_heuristic,
        split_depth=split_depth,
        log_opts=log_opts1)
    
    predict_all_circuits(NlatPC, nlat_result_circuits, log_opts1, train_x1, valid_x1, test_x1)
       

    # parameter learning
    train_x2, valid_x2, test_x2 = convert2latent(train_x, valid_x, test_x)
    pc, vtree = reload_learned_pc(nlat_results, "max-ll"; opts=log_opts1, name=train_x.name)
    init_fairpc = NlatPC(pc, vtree, nlat_fairpc.S, nlat_fairpc.D)
    fairpc = initial_structure(FairPC, init_fairpc; init_alg="prior-subop")

    log_opts2=Dict("train_x"=>train_x2, "valid_x"=>valid_x2, "test_x"=>test_x2,
        "outdir"=>joinpath(dir, "para"), "save"=>log_opts["save"], "patience"=>Inf, "learn_mode"=>"para")

    result_circuits, results = train_fair_psdd(fairpc, train_x2, "para";
                            max_iters=para_iters,
                            pseudocount=pseudocount,
                            init_para_alg="void",
                            log_opts=log_opts2)
    predict_all_circuits(FairPC, result_circuits, log_opts2, train_x, valid_x, test_x)

       
    return result_circuits, results
end

function model_fair_psdd(T::Type{<:Union{LatNB, TwoNB}}, train_x::FairDataset, valid_x::FairDataset, test_x::FairDataset;
            struct_iters,
            split_heuristic,
            split_depth,
            init_para_alg,
            para_iters,
            pseudocount,
            log_opts)

    pc = initial_structure(T, train_x; pseudocount=pseudocount)
    init_para_alg = T == TwoNB ? "estimate" : "prior-latent"
    max_iters = T == TwoNB ? 1 : para_iters
    log_opts["learn_mode"] = "para"
    pcs, results = train_fair_psdd(pc, train_x, "para";
                                            max_iters=max_iters,
                                            pseudocount=pseudocount, 
                                            init_para_alg=init_para_alg,
                                            log_opts=log_opts)
    
    predict_all_circuits(T, pcs, log_opts, train_x, valid_x, test_x)
end

function model_fair_psdd(T::Type{<:StructType}, rain_x::FairDataset, valid_x::FairDataset, test_x::FairDataset; kwargs...)
    error("Functin to train model with struct type $T is undefined")
end

function train_fair_psdd(pc::StructType, train_x::FairDataset, learn_mode;
                            max_iters,
                            pseudocount=1.0,
                            init_para_alg,
                            split_heuristic=(pick_egde="eFlow", pick_var="vMI"),
                            split_depth=1,
                            log_opts)
    # init paras
    tic = time_ns()
    initial_parameters(pc, train_x; para_alg=init_para_alg, pseudocount=pseudocount)
    toc = time_ns()

    # init logs
    if issomething(log_opts)
        log_results = log_init(opts=log_opts)
        log_per_iter(pc, train_x, log_results; opts=log_opts, iter=0, time=(toc-tic)/1.0e9)
    end

    # parameter learn
    if learn_mode == "para"
        if pc isa NonLatentStructType
            @assert max_iters == 1
        end
        for i in 1 : max_iters
            tic = time_ns()
            parameter_update(pc, train_x; pseudocount=pseudocount)
            toc = time_ns()

            if issomething(log_opts)
                continue_flag = log_per_iter(pc, train_x, log_results; opts=log_opts, iter=i, time=(toc-tic)/1.0e9)
            end
            if !continue_flag
                break
            end
        end


    elseif learn_mode == "struct"
        @assert pc isa NonLatentStructType
        for i in 1 : max_iters
            tic = time_ns()
            pc, stop = structure_update(pc, train_x; 
                pick_edge=split_heuristic.pick_egde, 
                pick_var=split_heuristic.pick_var, 
                split_depth=split_depth)
            parameter_update(pc, train_x; pseudocount=pseudocount)
            if stop
                break
            end
            toc = time_ns()
            continue_flag = true
            if issomething(log_opts)
                continue_flag = log_per_iter(pc, train_x, log_results; opts=log_opts, iter=i, time=(toc-tic)/1.0e9)
            end
            if !continue_flag
                break
            end
        end
    end

    result_circuits = Dict()
    if issomething(log_opts)
        for key in ["max-ll"]
            result_circuits[key] = reload_learned_pc(log_results, key; opts=log_opts, name=train_x.name)
        end
        return result_circuits, log_results
    else
        return Dict(""=>(pc, vtree)), log_results
    end
end

function learn(name, SV;
                outdir=joinpath(pwd(), "exp-results", Dates.format(now(), "yyyymmdd-HHMMSSs")),
                save_step=1,
                seed=1337,
                patience=100,

                # data set
                # missing_perct, deprecated
                # batch_size,
                fold,
                
                # struct learn
                struct_type="FairPC",
                struct_iters=1000,
                split_heuristic=(pick_egde="eFlow", pick_var="vMI"),
                split_depth=1,

                # para learn
                init_para_alg="prior-subop",
                para_iters=500,
                pseudocount=1.0,

                # for synthetic data
                num_X=nothing)
    # output dir
    if !isdir(outdir)
        mkpath(outdir)
    end

    # seed
    Random.seed!(seed)
    T = STRUCT_STR2TYPE[struct_type]

    # data
    train_x, valid_x, test_x = load_data(name, T, SV; fold=fold, num_X=num_X)

    # learn and predicton
    model_fair_psdd(T, train_x, valid_x, test_x;
                        # struct
                        struct_iters=struct_iters,
                        split_heuristic=split_heuristic,
                        split_depth=split_depth,
                        # para
                        init_para_alg=init_para_alg,
                        para_iters=para_iters,
                        pseudocount=pseudocount,
                        log_opts=Dict("outdir"=>outdir,
                                        "save"=>save_step,
                                        "patience"=>patience,
                                        "train_x"=>train_x,
                                        "valid_x"=>valid_x,
                                        "test_x"=>test_x))
end


function learn_test()
    learn("adult", "sex",
                fold=1,
                struct_iters=5,
                para_iters=10,
                struct_type="FairPC")
end

function all_learned_circuits(;indir="/Users/mhdang/Desktop/Fairnes/FAIRPSDDExp/aaai2021/realworld_fair_10cv_0905")
    outdir = "./circuits"
    !isdir(outdir) && mkpath(outdir)
    for name in DATASET_NAMES[1:3]
        for fold in 1 : 10
            configs = read_all_exp_configs(indir; given=Dict("dataset"=>name, "fold"=>fold, "struct_type"=>"FairPC"))
            @assert length(configs) == 1
            config = configs[1]
            pcfile = joinpath(config["root"], "struct", "max-ll")
            cp(joinpath(pcfile, "$name.psdd"), joinpath(outdir, "$name-$fold.psdd"))
            cp(joinpath(pcfile, "$name.vtree"), joinpath(outdir, "$name-$fold.vtree"))
        end
    end
    name = "synthetic"
    indir = indir="/Users/mhdang/Desktop/Fairnes/FAIRPSDDExp/aaai2021/synthetic_fair_10cv_0905/"
    for num_X in 10:30
        for fold in 1 : 10
            configs = read_all_exp_configs(indir; given=Dict("dataset"=>name, "fold"=>fold, "struct_type"=>"FairPC", "num_X"=>num_X))
            @assert length(configs) == 1
            config = configs[1]
            pcfile = joinpath(config["root"], "struct", "max-ll")
            cp(joinpath(pcfile, "$name.psdd"), joinpath(outdir, "$name-$num_X-$fold.psdd"))
            cp(joinpath(pcfile, "$name.vtree"), joinpath(outdir, "$name-$num_X-$fold.vtree"))
        end
    end
end

function fair_pc_para_learn_from_file(name, SV;
                                        indir="./circuits",
                                        outdir=joinpath(pwd(), "exp-results", Dates.format(now(), "yyyymmdd-HHMMSSs")),
                                        save_step=1,
                                        seed=1337,
                                        patience=100,

                                        # data set
                                        fold,

                                        # para learn
                                        init_para_alg="void",
                                        para_iters=typemax(Int64),
                                        pseudocount=1.0,
                                        num_X=nothing,
                                        missing_perct=0.0)
    struct_type = "FairPC"
    if !isdir(outdir)
        mkpath(outdir)
    end

    # seed
    Random.seed!(seed)
    T = STRUCT_STR2TYPE[struct_type]

    # data
    train_x, valid_x, test_x = load_data(name, T, SV; fold=fold, num_X=num_X)
    train_x2, valid_x2, test_x2 = convert2latent(train_x, valid_x, test_x)
    train_x2 = flip_coin(T, train_x2; keep_prob=1-missing_perct)

    if name == "synthetic"
        pc = read((joinpath(indir, "$name-$num_X-$fold.psdd"), joinpath(indir, "$name-$num_X-$fold.vtree")), StructProbCircuit)
        vtree = pc.vtree
    else
        pc = read((joinpath(indir, "$name-$fold.psdd"), joinpath(indir, "$name-$fold.vtree")), StructProbCircuit)
        vtree = pc.vtree
    end

    init_fairpc = NlatPC(pc, vtree, train_x.S, train_x.D)
    fairpc = initial_structure(FairPC, init_fairpc; init_alg="prior-subop")

    log_opts2=Dict("train_x"=>train_x2, "valid_x"=>valid_x2, "test_x"=>test_x2,
        "outdir"=>joinpath(outdir, "para"), "save"=>save_step, "patience"=>Inf, "learn_mode"=>"para", "missing"=>missing_perct!=0.0)

    result_circuits, results = train_fair_psdd(fairpc, train_x2, "para";
                            max_iters=para_iters,
                            pseudocount=pseudocount,
                            init_para_alg=init_para_alg,
                            log_opts=log_opts2)
    predict_all_circuits(FairPC, result_circuits, log_opts2, train_x, valid_x, test_x)
end