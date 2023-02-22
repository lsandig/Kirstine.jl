using Test
using LinearAlgebra: norm
using Statistics: mean
using Random: seed!, rand!

using Kirstine

@testset "Kirstine" begin
    include("pso.jl")
    include("designmeasure.jl")
    include("design.jl")
end
