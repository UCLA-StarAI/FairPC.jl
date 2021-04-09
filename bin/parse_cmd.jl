using ArgParse
using Dates
using JSON
using Logging
using Test
using Random
using Suppressor
using LogicCircuits
using ProbabilisticCircuits
using LearnFairPSDD


function add_basic_arg(s::ArgParseSettings)
    @add_arg_table! s begin
        "dataset"
            help = "Dataset name"
            arg_type = String
            required = true
        "--sensitive_variable"
            help = "Sensitive variables"
            arg_type = String
        "--dir", "-d"
            help = "Output directory"
            arg_type = String
            default = "exp-results/"
        "--save_step"
            help = "Save logging info per save_step iterations"
            arg_type = Int64
            default = 1
        "--seed"
            help = "Seed for random generation"
            arg_type = Int64
            default = 1337
        "--patience"
            arg_type = Int64
            default = 100
        "--fold"
            help = "fold id for k-fold cross validation or random splits"
            arg_type = Int64
        "--struct_type"
            help = "Indicate structure constrains of circuits"
            arg_type = String
            default = "FairPC"
        "--struct_iters"
            help = "Number of maximum iterations of structure learning"
            arg_type = Int64
            default = 1000
        "--pick_edge"
            arg_type = String
            help = "Heuristic method to do split"
            default = "eFlow"        
        "--pick_var"
            arg_type = String
            help = "Heuristic method to do split"
            default = "vMI"
        "--split_depth"
            arg_type = Int64
            help = "Split depth"
            default = 1
        "--init_para_alg"
            help = "Algorithm to initialize parameters"
            arg_type = String
            default = "prior-subop"
        "--para_iters"
            help = "Number of maximum iterations of parameter learning"
            arg_type = Int64
            default = 500
        "--pseudocount"
            help = "Laplace smoothing pseudocount"
            arg_type = Float64
            default = 1.0
        "--num_X"
            arg_type = Int64
            default = nothing
        "--exp-id"
            help = "Experiment id"
            arg_type = String
            default = Dates.format(now(), "yyyymmdd-HHMMSSs")
        "--missing_perct"
            arg_type = Float64
            default = 0.0
        end
end

function parse_cmd_line()
    s = ArgParseSettings()
    add_basic_arg(s)
    return parse_args(s)
end
