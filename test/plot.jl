@testset "plot" begin
    @test_throws "1- or 2-dimensional" plot(singleton_design([1, 2, 3]))
end
