using BenchmarkTools: @ballocated

macro test_noalloc(expr)
    @test(@ballocated($expr; evals = 1, samples = 1) == 0)
end
