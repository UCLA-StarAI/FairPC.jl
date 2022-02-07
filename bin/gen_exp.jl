using ArgParse
using Dates
using JSON
using Logging
using IterTools

RAND_SEED = 1337
DEFALT_CMD = "julia"

function cart_prod_dict(d)
    return [Dict(zip(keys(d), x)) for x in IterTools.Iterators.product(values(d)...)]
end

function constrained_dict(constraint, config)
    dict = Dict()
    for (key, sub_dict) in constraint
        k = [k for (k, v) in sub_dict if k in values(config)]
        @assert length(k) in [0, 1] "$k $configs"
        if length(k) == 1
            k = k[1]
            v = sub_dict[k]
            dict[key] = v
        end
    end
    dict
end

function gen_exp_configs(config)
    default = filter(x -> x[2] isa Union{String, Integer, AbstractFloat}, config)
    grid = filter(x -> x[2] isa Vector, config)
    constraint = filter(x -> x[2] isa Dict, config)

    # grid
    configs = cart_prod_dict(grid)

    # merge default
    foreach(c -> merge!(c, deepcopy(default)), configs)

    # merge constraint
    results = []
    for c in configs
        dict = constrained_dict(constraint, c)
        append!(results, map(d -> merge(deepcopy(c), deepcopy(d)), cart_prod_dict(dict)))
    end
    results
end


function vals_to_str(v)
    if isa(v, Array)
        return join(v, " ")
    else
        return repr(v)
    end
end

function parse_cmd_line()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "config"
            help = "Path to config (JSON) file"
            arg_type = String
            required = true
        "--output", "-o"
            help = "Output directory to print commands"
            arg_type = String
            default = "scripts/logs"
        "--defaults", "-d"
            help = "Path to default json"
            arg_type = String
            default = nothing
        "--constraint", "-c"
            help = "Path to constraint json"
            arg_type = String
            default = nothing
        "--bin", "-b"
            help = "Binary name"
            arg_type = String
            default = "bin/learn.jl"
        "--env", "-e"
            help = "Bash environment strings"
            arg_type = String
            default = ""
        "--exp-id"
            help = "Experiment id"
            arg_type = String
            default = "gen-script"
        "--seeds"
            help = "Seeds for random configurations"
            arg_type = AbstractArray{Int}
            default = [1337]
        "--cmd"
            help = "Command line execuation file"
            arg_type = String
            default = "julia --project"
        "--set_id"
            arg_type = Bool
            default = true
    end

    return parse_args(s)
end

function main()

    args = parse_cmd_line()

    #
    # creating output dirs, if they do not exist
    # keep track of exps via ids or dates
    date_string = Dates.format(now(), "yyyymmdd-HHMMSSs")
    out_path = joinpath(args["output"], "$(args["exp-id"])_$(date_string)")
    mkpath(out_path)

    #
    # setting local logger
    log_path_dir = joinpath(out_path, "logs")
    mkpath(log_path_dir)
    log_path = joinpath(log_path_dir, "log.txt")
    log_io = open(log_path, "w+")
    logger = SimpleLogger(log_io)


    #
    # loading default params
    defaults = Dict()
    if args["defaults"] != nothing
        open(args["defaults"]) do f
            defaults = JSON.parse(f)
        end
        @info("Loaded defaults: $defaults")
    end

    #
    # loading exp configs
    config = nothing
    open(args["config"]) do f
       config = JSON.parse(f)
    end
    @info("Loaded configs: $config")

    #
    # cartesian product of dictionaries
    configs = gen_exp_configs(config)


    for (exp_id, c) in enumerate(configs)
        #for seed in args["seeds"]
            d = merge(c, deepcopy(defaults))
            @assert haskey(d, "expname")
            #
            # create command string
            cmd = "$(args["env"]) $(args["cmd"]) $(args["bin"]) $(d["expname"])"
            pop!(d, "expname")

            if args["set_id"]
                cmd *= " --exp-id $exp_id "
            end

            if haskey(c, "dir")
                exp_path = d["dir"]
                exp_path = joinpath(exp_path, c["expname"], "$exp_id")
                cmd *= " --dir \"$exp_path\""
                pop!(d, "dir")
            end

            for (k, v) in d
                cmd *= " --$(k) $(vals_to_str(v)) "
            end

            println(cmd)
        end
    #end

end

main()
