using Test
using SparsePropV11

@testset "SparsePropV11" begin
    include("test_gradient.jl")
    include("test_event_sparsity.jl")
end
