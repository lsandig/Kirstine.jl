using Test
using LinearAlgebra: Symmetric, UpperTriangular, norm, tr, det, diagm
using Statistics: mean
using Random: seed!, rand!
using Plots: plot

using Kirstine

@testset "Kirstine" begin
    include("pso.jl")
    include("designmeasure.jl")
    include("types.jl")
    include("design-common.jl")
    include("design-doptimal.jl")
    include("util.jl")
    include("plot.jl")
end
