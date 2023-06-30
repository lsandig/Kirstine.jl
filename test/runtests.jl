using Test

@testset "Kirstine" begin
    include("pso.jl")
    include("designmeasure.jl")
    include("types.jl")
    include("design-common.jl")
    include("design-doptimal.jl")
    include("util.jl")
    include("plot.jl")
end
