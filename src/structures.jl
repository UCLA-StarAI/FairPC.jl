export StructType, LatentStructType, NonLatentStructType, FairPC, LatNB, NlatPC,
TwoNB, STRUCT_STR2TYPE, node_D, node_Df, node_not_D, node_not_Df, nodes_D_sum,
nodes_DfS_sum, nodes_DX_sum, nodes_DS_sum

"""
The initial structure of pc in current scenario
`init_struct`
model with latent variables
1. fairPC
    Df is independent of S
    D is independent of X given Df and S
2. latNB
    Df is independent of S
    D is independent of X given Df and S
    X are independent of each other

model without latent variables
3. nlatPC 
    D is independent of S
    X are caused by D ans S 
4. twoNB
    D is caused by S
    X are caused by S and D 
    X are independent of each other
"""

abstract type StructType end
abstract type LatentStructType <: StructType end
abstract type NonLatentStructType <: StructType end
mutable struct FairPC <: LatentStructType
    pc::ProbCircuit
    vtree::PlainVtree
    S::Var
    D::Var
    Df::Var
    FairPC(pc, vtree, S, D) = new(pc, vtree, S, D, D+1)
end

mutable struct LatNB <: LatentStructType
    pc::ProbCircuit
    vtree::PlainVtree
    S::Var
    D::Var
    Df::Var
    LatNB(pc, vtree, S, D) = new(pc, vtree, S, D, D+1)
end

mutable struct NlatPC <: NonLatentStructType
    pc::ProbCircuit
    vtree::PlainVtree
    S::Var
    D::Var
    NlatPC(pc, vtree, S, D) = new(pc, vtree, S, D)
end

mutable struct TwoNB <: NonLatentStructType
    pc::ProbCircuit
    vtree::PlainVtree
    S::Var
    D::Var
    TwoNB(pc, vtree, S, D) = new(pc, vtree, S, D)
end

const STRUCT_STR2TYPE = Dict(
    "FairPC" => FairPC,
    "LatNB" => LatNB,
    "NlatPC" => NlatPC,
    "TwoNB" => TwoNB
)

###########
root(x::ProbCircuit) = x
left_sums(x::ProbCircuit) = map(y -> children(y)[1], children(root(x)))
right_sums(x::ProbCircuit) = map(y -> children(y)[2], children(root(x)))
left_left_true(x::ProbCircuit) = left_sums(x)[1] |> y -> children(y)[1] |> y -> children(y)[1]
left_left_false(x::ProbCircuit) = left_sums(x)[3] |> y -> children(y)[1] |> y -> children(y)[1]
left_right_true(x::ProbCircuit) = left_sums(x)[1] |> y -> children(y)[1] |> y -> children(y)[2]
left_right_false(x::ProbCircuit) = left_sums(x)[2] |> y -> children(y)[1] |> y -> children(y)[2]
right_left_sums(x::ProbCircuit) = map(y -> children(y)[1] |> y -> children(y)[1], right_sums(x))
right_right_sums(x::ProbCircuit) = map(y -> children(y)[1] |> y -> children(y)[2], right_sums(x))
right_left_true(x::ProbCircuit) = right_left_sums(x)[1]|> y -> children(y)[1]
right_left_false(x::ProbCircuit) = right_left_sums(x)[1]|> y -> children(y)[2]

function node_not_Df(f::LatentStructType)
    n = left_left_false(f.pc)
    @assert - literal(n) == f.Df
    n
end

function node_Df(f::LatentStructType)
    n = left_left_true(f.pc)
    @assert literal(n) == f.Df
    n
end

function node_D(f::LatentStructType)
    n = right_left_true(f.pc)
    @assert literal(n) == f.D
    n
end

function node_S(f::LatentStructType)
    n = left_right_true(f.pc)
    @assert literal(n) == f.S
end

function node_not_D(f::LatentStructType)
    n = right_left_false(f.pc)
    @assert - literal(n) == f.D 
    n
end


function node_D(fairpc::TwoNB)
    pc = fairpc.pc
    n = children(pc)[1] |> x -> children(x)[2] |> x -> children(x)[1] |> x -> children(x)[1]
    @assert literal(n) == fairpc.D
    n
end

function node_not_D(fairpc::TwoNB)
    pc = fairpc.pc
    n = children(pc)[1] |> x -> children(x)[2] |> x -> children(x)[2] |> x -> children(x)[1]
    @assert - literal(n) == fairpc.D
    n
end

function node_D(pc::NlatPC)
    n = left_left_true(pc.pc)
    @assert literal(n) == pc.D
    n
end

function node_not_D(pc::NlatPC)
    n = left_left_false(pc.pc)
    @assert - literal(n) == pc.D
    n
end

function non_sensitive_sub_circuits(fairpc::LatentStructType)
    right_right_sums(fairpc.pc)
end

function non_sensitive_sub_circuits(fairpc::NlatPC)
    right_sums(fairpc.pc)
end

function nodes_D_sum(fairpc::LatentStructType)
    right_left_sums(fairpc.pc)
end

function nodes_DfS_sum(fairpc::LatentStructType)
    left_sums(fairpc.pc)
end

function nodes_DS_sum(fairpc::NlatPC)
    left_sums(fairpc.pc)
end

function nodes_DX_sum(fairpc::LatentStructType)
    right_sums(fairpc.pc)
end

