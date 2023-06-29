@testset "plot.jl" begin
@testset "plot" begin
    @test_throws "1- or 2-dimensional" plot(one_point_design([1, 2, 3]))
end
end
