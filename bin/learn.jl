include("parse_cmd.jl")
using GR
using Suppressor
GR.inline("pdf")

function main()
    args = parse_cmd_line()
    name = args["dataset"]
    SV = args["sensitive_variable"]
    outdir = args["dir"]
    save_step = args["save_step"]
    seed = args["seed"]
    patience = args["patience"]

    fold = args["fold"]
    struct_type = args["struct_type"]
    struct_iters = args["struct_iters"]
    pick_edge = args["pick_edge"]
    pick_var = args["pick_var"]
    split_depth = args["split_depth"]

    init_para_alg = args["init_para_alg"]
    para_iters = args["para_iters"]
    pseudocount = args["pseudocount"]

    num_X = args["num_X"]


    # log config
    if !isdir(outdir)
        mkpath(outdir)
    end
    open(joinpath(outdir, "config.json"),"w") do f
        write(f, JSON.json(args))
    end

    # io = open(joinpath(outdir, "config.log"), "w")
    # logger = SimpleLogger(io)
    # global_logger(logger)
    # arg_str = arg2str(args)
    # @info arg_str
    # flush(io)

    # try
        @suppress_out learn(name, SV;
            outdir=outdir,
            save_step=save_step,
            seed=seed,
            patience=patience,

            # data set
            # missing_perct, deprecated
            # batch_size,
            fold=fold,
            
            # struct learn
            struct_type=struct_type,
            struct_iters=struct_iters,
            split_heuristic=(pick_egde=pick_edge, pick_var=pick_var),
            split_depth=split_depth,

            # para learn
            init_para_alg=init_para_alg,
            para_iters=para_iters,
            pseudocount=pseudocount,

            # for synthetic data
            num_X=num_X)

    # catch e
    #     @error "Error! Program terminated unexpectedly."
    #     with_logger(SimpleLogger(stdout)) do
    #         @error "Error! Program terminated unexpectedly in args.\n" arg_str
    #     end
    # end

    # @info "Learn Fair PSDD ends!"
end

main()
