include("parse_cmd.jl")
using GR
using Suppressor
GR.inline("pdf")

function main()
    args = parse_cmd_line()
    name = args["dataset"]
    SV = args["sensitive_variable"]
    outdir = args["dir"]
    fold = args["fold"]
    para_iters = args["para_iters"]
    missing_perct = args["missing_perct"]
    num_X = args["num_X"]
    init_para_alg = args["init_para_alg"]

    # log config
    if !isdir(outdir)
        mkpath(outdir)
    end
    open(joinpath(outdir, "config.json"),"w") do f
        write(f, JSON.json(args))
    end

    @suppress_out fair_pc_para_learn_from_file(name, SV;
            outdir=outdir,
            fold=fold,
            para_iters=para_iters,
            num_X=num_X,
            missing_perct=missing_perct,
	    init_para_alg=init_para_alg)
end

main()
