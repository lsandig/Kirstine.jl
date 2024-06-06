# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module PsoTests
using Test
using Kirstine
using Random: seed!, rand!
using LinearAlgebra: norm
using Statistics: mean

include("example-pnt.jl")

@testset "optimizer-pso.jl" begin
    @testset "Pso" begin
        @test_throws "must be positive" Pso(iterations = -1, swarmsize = 5, c1 = 3, c2 = 4)
        @test_throws "at least 2" Pso(iterations = 10, swarmsize = 1, c1 = 3, c2 = 4)
        @test_throws "c1 + c2 > 4" Pso(iterations = 10, swarmsize = 5, c1 = 1, c2 = 1)
        @test_throws "non-negative" Pso(iterations = 10, swarmsize = 5, c1 = -1, c2 = 6)
        @test_throws "non-negative" Pso(iterations = 10, swarmsize = 5, c1 = 6, c2 = -1)
    end

    @testset "optimizer_state" begin
        let fconst(p) = 42,
            finf(p) = p.x[1] > 3 ? Inf : p.x[1],
            fnan(p) = NaN,
            o3 = Pso(; swarmsize = 3, iterations = 10),
            o5 = Pso(; swarmsize = 5, iterations = 10),
            ps = [Pnt([i]) for i in 1:5],
            c = BoxConstraints(Pnt([0]), Pnt([10]))

            @test_throws ">= number" Kirstine.optimizer_state(fconst, o3, ps, c)
            @test_throws "infinite objective" Kirstine.optimizer_state(finf, o5, ps, c)
            @test_throws "NaN on all" Kirstine.optimizer_state(fnan, o5, ps, c)
        end

        let f(p) = -p.x[1],
            o = Pso(; swarmsize = 5, iterations = 10),
            ps = [Pnt([i]) for i in 0:2],
            c = BoxConstraints(Pnt([0]), Pnt([10]))

            state = Kirstine.optimizer_state(f, o, ps, c)

            @test state isa Kirstine.PsoState
            # prototypes are used first
            @test state.x[1:3] == ps
            # remaining particles are random
            @test all(Ref(state.x[4]) .!= state.x[1:3])
            @test all(Ref(state.x[5]) .!= state.x[1:4])
            # objective was evaluated
            @test all(state.fx .== [-pt.x[1] for pt in state.x])
            # personal and global bests are initialized
            @test state.fp == state.fx
            @test state.fg == -state.x[1].x[1]
            @test state.fg == maximum(state.fx)
            # each point is evaluated once
            @test state.n_eval == o.swarmsize
        end
    end

    @testset "optimize" begin
        let n = 4,
            xstar = ones(n),
            f(p) = -sum((p.x .- xstar) .^ 2),
            pso = Pso(; iterations = 100, swarmsize = 20),
            prototype = Pnt(zeros(n)),
            outside = Pnt([-1, -1, 5, -1]),
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

            seed!(4711)
            @test_warn "maximum is not finite" Kirstine.optimize(
                p -> sum((p.x .- xstar) .^ 2) < 0.1 ? Inf : f(p),
                pso,
                [Pnt([-1, 0, 1, 2])],
                constr1,
            )

            @test_throws "do not satisfy" Kirstine.optimize(f, pso, [outside], constr1)
        end
    end
end
end
