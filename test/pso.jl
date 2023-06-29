@testset "Pso" begin
    struct Pnt <: Kirstine.AbstractPoint
        x::Vector{Float64}
    end
    function Kirstine.randomize!(p::Pnt, constraints)
        lb, ub = constraints
        rand!(p.x)
        p.x .*= ub.x .- lb.x
        p.x .+= lb.x
        return p
    end
    function Kirstine.difference!(v::AbstractVector{<:Real}, p::Pnt, q::Pnt)
        v .= p.x .- q.x
        return v
    end
    Kirstine.flat_length(p::Pnt) = length(p.x)
    function Kirstine.copy!(to::Pnt, from::Pnt)
        to.x .= from.x
        return to
    end
    function Kirstine.move!(p::Pnt, v::AbstractVector{<:Real}, constraints)
        lb, ub = constraints
        p.x .+= v
        p.x .= max.(lb.x, p.x)
        p.x .= min.(ub.x, p.x)
        return p
    end

    @test_throws "must be positive" Pso(iterations = -1, swarmsize = 5, c1 = 3, c2 = 4)
    @test_throws "at least 2" Pso(iterations = 10, swarmsize = 1, c1 = 3, c2 = 4)
    @test_throws "c1 + c2 > 4" Pso(iterations = 10, swarmsize = 5, c1 = 1, c2 = 1)
    @test_throws "non-negative" Pso(iterations = 10, swarmsize = 5, c1 = -1, c2 = 6)
    @test_throws "non-negative" Pso(iterations = 10, swarmsize = 5, c1 = 6, c2 = -1)

    let n = 4,
        xstar = ones(n),
        f(p) = -sum((p.x .- xstar) .^ 2),
        pso = Pso(; iterations = 100, swarmsize = 20),
        prototype = Pnt(zeros(n)),
        constr1 = (Pnt(fill(-1, n)), Pnt(fill(2, n))),
        constr2 = (Pnt(fill(-1, n)), Pnt(collect(1:n) ./ 2)),
        _ = seed!(4711),
        # This solution will be at xstar, inside the constraints...
        r1 = Kirstine.optimize(pso, f, [prototype], constr1),
        # ... and this one will be on the boundary at [0.5, 1, 1, ..., 1]
        r2 = Kirstine.optimize(pso, f, [prototype], constr2; trace_state = true),
        mean_speed = map(st -> mean(norm, st.v), r2.trace_state)

        @test r1.maximizer.x ≈ xstar rtol = 1e-4
        @test r1.maximum ≈ 0.0 atol = 1e-4
        @test issorted(r1.trace_fx)
        @test r2.maximizer.x ≈ [0.5; xstar[2:end]] rtol = 1e-4
        @test r2.maximum ≈ -0.25 rtol = 1e-4
        @test issorted(r2.trace_fx)
        # geometric mean of relative speed reductions (after telescope product)
        @test (mean_speed[end] / mean_speed[2])^(1 / (pso.iterations - 1)) < 1
    end
end
