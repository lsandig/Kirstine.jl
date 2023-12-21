# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module PsoTests
using Test
using Kirstine
using Random: seed!, rand!
using LinearAlgebra: norm
using Statistics: mean

struct Pnt <: Kirstine.AbstractPoint
    x::Vector{Float64}
end

struct PntDiff <: Kirstine.AbstractPointDifference
    v::Vector{Float64}
end

struct BoxConstraints <: Kirstine.AbstractConstraints
    lb::Pnt
    ub::Pnt
end

function Kirstine.ap_rand!(p::Pnt, c::BoxConstraints)
    rand!(p.x)
    p.x .*= c.ub.x .- c.lb.x
    p.x .+= c.lb.x
    return p
end

function Kirstine.ap_diff!(d::PntDiff, p::Pnt, q::Pnt)
    d.v .= p.x .- q.x
    return d.v
end

function Kirstine.ap_copy!(to::Pnt, from::Pnt)
    to.x .= from.x
    return to
end

function Kirstine.ap_add!(p::Pnt, d::PntDiff, c::BoxConstraints)
    p.x .+= d.v
    p.x .= max.(c.lb.x, p.x)
    p.x .= min.(c.ub.x, p.x)
    return p
end

function Kirstine.ap_as_difference(p::Pnt)
    return PntDiff(deepcopy(p.x))
end

function Kirstine.ap_rand!(d::PntDiff, lb::Real, ub::Real)
    rand!(d.v)
    d.v .*= (ub - lb)
    d.v .+= lb
    return d
end

function Kirstine.ap_mul!(d1::PntDiff, d2::PntDiff)
    d1.v .*= d2.v
    return d1
end

function Kirstine.ap_mul!(d::PntDiff, a::Real)
    d.v .*= a
    return d
end

function Kirstine.ap_add!(d1::PntDiff, d2::PntDiff)
    d1.v .+= d2.v
    return d1
end

@testset "pso.jl" begin
    @testset "Pso" begin
        @test_throws "must be positive" Pso(iterations = -1, swarmsize = 5, c1 = 3, c2 = 4)
        @test_throws "at least 2" Pso(iterations = 10, swarmsize = 1, c1 = 3, c2 = 4)
        @test_throws "c1 + c2 > 4" Pso(iterations = 10, swarmsize = 5, c1 = 1, c2 = 1)
        @test_throws "non-negative" Pso(iterations = 10, swarmsize = 5, c1 = -1, c2 = 6)
        @test_throws "non-negative" Pso(iterations = 10, swarmsize = 5, c1 = 6, c2 = -1)
    end

    @testset "optimize" begin
        let n = 4,
            xstar = ones(n),
            f(p) = -sum((p.x .- xstar) .^ 2),
            pso = Pso(; iterations = 100, swarmsize = 20),
            prototype = Pnt(zeros(n)),
            constr1 = BoxConstraints(Pnt(fill(-1, n)), Pnt(fill(2, n))),
            constr2 = BoxConstraints(Pnt(fill(-1, n)), Pnt(collect(1:n) ./ 2)),
            _ = seed!(4711),
            # This solution will be at xstar, inside the constraints...
            r1 = Kirstine.optimize(f, pso, [prototype], constr1),
            # ... and this one will be on the boundary at [0.5, 1, 1, ..., 1]
            r2 = Kirstine.optimize(f, pso, [prototype], constr2; trace_state = true),
            mean_speed = map(st -> mean(vi -> norm(vi.v), st.v), r2.trace_state)

            @test r1.maximizer.x ≈ xstar rtol = 1e-4
            @test r1.maximum ≈ 0.0 atol = 1e-4
            @test issorted(r1.trace_fx)
            @test r2.maximizer.x ≈ [0.5; xstar[2:end]] rtol = 1e-4
            @test r2.maximum ≈ -0.25 rtol = 1e-4
            @test issorted(r2.trace_fx)
            # geometric mean of relative speed reductions (after telescope product)
            @test (mean_speed[end] / mean_speed[2])^(1 / (pso.iterations - 1)) < 1
            # number of iterations requested is not off by 1 (initial state is iteration 1)
            @test length(r1.trace_fx) == 100
            @test length(r1.trace_x) == 100
            @test length(r2.trace_state) == 100

            @test_throws "infinite objective" Kirstine.optimize(
                p -> p.x[1] < 0 ? Inf : sum(p.x),
                pso,
                [Pnt([-1, 0, 1, 2])],
                constr1,
            )
            @test_throws "NaN on all" Kirstine.optimize(p -> NaN, pso, [prototype], constr1)
        end
    end
end
end
