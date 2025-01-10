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

    @testset "pso_update_velocity!" begin
        let x1 = [2, 1],
            p1 = [1, 1],
            v1 = [1, 0.5],
            g = [4, 5],
            r1 = [0.8, 0.5],
            r2 = [1.0, 0.4],
            c1 = 2.0,
            c2 = 2.05,
            chi = 0.8,
            o = Pso(; swarmsize = 2, iterations = 1),
            ps = [Pnt(x1), Pnt([7, 7])], # we need >= 2 points for a sensible swarm
            c = BoxConstraints(Pnt([0, 0]), Pnt([10, 10])),
            state = Kirstine.optimizer_state(p -> 42, o, ps, c) # dummy function

            # We have already initialized the current position of the first particle.
            # Now we its personal best location as well as its current velocity,
            # and also some global best position.
            # We set the random components to deterministic values and run the velocity update.
            # The second particle is conceptually the same and not tested.
            # Note that the _values_ at x/p/g are irrelevant for the velocity update.
            state.p[1].x .= p1
            state.v[1].v .= v1
            state.g.x .= g
            state.r1.v .= r1
            state.r2.v .= r2

            v1new = @. chi * (v1 + c1 * r1 * (p1 - x1) + c2 * r2 * (g - x1))
            @test v1new == [2.8, 3.024]

            Kirstine.pso_update_velocity!(state, 1, c1, c2, chi)
            @test state.v[1].v == v1new
        end
    end

    @testset "pso_randomize_velocity_multipliers!()" begin
        let r1 = [0.42, 0.21],
            r2 = [0.5, 1.0],
            o = Pso(; swarmsize = 2, iterations = 1),
            ps = [Pnt([2, 1]), Pnt([7, 7])],
            c = BoxConstraints(Pnt([0, 0]), Pnt([10, 10])),
            state = Kirstine.optimizer_state(p -> 42, o, ps, c)

            state.r1.v .= r1
            state.r2.v .= r2
            Kirstine.pso_randomize_velocity_multipliers!(state)

            @test state.r1.v != state.r2.v
            @test state.r1.v != r1
            @test state.r2.v != r2
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
            @test length(r1.trace_gx) == 0
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
