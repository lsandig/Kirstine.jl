# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module FrankWolfeTests
using Kirstine, Test
using Plots
using LinearAlgebra

include("example-pnt.jl")

function Kirstine.fw_linear_maximization_oracle!(
    v::Pnt,
    gx::AbstractVector{<:Real},
    c::BoxConstraints,
)
    box_corners = Iterators.product(Iterators.zip(c.lb.x, c.ub.x)...)
    v.x .= argmax(corner -> mapreduce(*, +, gx, corner), box_corners)
    return v
end

function Kirstine.fw_evaluate_linearization(
    gx::AbstractVector{<:Real},
    p::PntDiff,
    c::BoxConstraints,
)
    return dot(gx, p.v)
end

function Kirstine.fw_gradient_length(p::Pnt, c::BoxConstraints)
    return length(p.x)
end

function Kirstine.fw_norm(d::PntDiff)
    return norm(d.v)
end

@testset "optimizer-frank-wolfe.jl" begin
    @testset "OpenLoop" begin
        @test_throws ">= 1" Kirstine.OpenLoop(0)
    end

    @testset "Adaptive" begin
        Adaptive = Kirstine.Adaptive
        @test_throws "eta must be <= 1" Adaptive(; eta = 2)
        @test_throws "tau must be > 1" Adaptive(; tau = 1)
        @test_throws "threshold must be > 0" Adaptive(; smoothness_threshold = 0)
        @test_throws "(0, threshold]" Adaptive(; initial_smoothness_estimate = 2e10)
        @test_throws "(0, threshold]" Adaptive(; initial_smoothness_estimate = -1)
    end

    @testset "FrankWolfe" begin
        # check both positional and keyword-based constructor
        msg = "must be positive"
        @test_throws msg FrankWolfe(-1, Kirstine.OpenLoop(2))
        @test_throws msg FrankWolfe(; iterations = -1, step = Kirstine.OpenLoop(2))
    end

    @testset "stepsize_openloop" begin
        @test Kirstine.stepsize_openloop(2, 0) == 1
        @test Kirstine.stepsize_openloop(5, 0) == 1
        @test Kirstine.stepsize_openloop(2, 3) == 0.4
        @test Kirstine.stepsize_openloop(5, 5) == 0.5
    end

    @testset "stepsize_constnumiter" begin
        @test Kirstine.stepsize_constnumiter(9, 0) == 1 / 3
        @test Kirstine.stepsize_constnumiter(16, 0) == 1 / 4
        @test Kirstine.stepsize_constnumiter(9, 2) == 1 / 3
        @test Kirstine.stepsize_constnumiter(16, 3) == 1 / 4

        @test_throws "past maximum" Kirstine.stepsize_constnumiter(9, 10)
    end

    @testset "fw_estimate_initial_smoothness!" begin
        let n = 4,
            xstar = ones(n),
            g!(grad, p) = grad .= -2 .* (p.x .- xstar),
            x = Pnt(3:6),
            x_tmp = Pnt(fill(NaN, 4)),
            d = PntDiff([1, -1, 1, -2]),
            d_tmp = PntDiff(fill(NaN, 4)),
            gx = g!(zeros(4), x),
            gx_tmp = fill(NaN, 4),
            c = BoxConstraints(Pnt(fill(-1, n)), Pnt(fill(10, n))),
            w = Kirstine.AdaptiveWorkspace(NaN, x_tmp, d_tmp, gx_tmp)

            res = Kirstine.fw_estimate_initial_smoothness!(g!, w, x, gx, d, c)

            @test w.smoothness_estimate ≈ 2
            @test res == w
        end
    end

    @testset "fw_workspace" begin
        let ol = Kirstine.OpenLoop(2),
            cni = Kirstine.ConstNumIter(),
            ad = Kirstine.Adaptive()

            w_ol = Kirstine.fw_workspace(ol, Pnt([0]), 1)
            w_cni = Kirstine.fw_workspace(cni, Pnt([0]), 1)
            w_ad = Kirstine.fw_workspace(ad, Pnt([0, 1, 2]), 3)

            @test w_ol isa Kirstine.FWDummyWorkspace
            @test w_cni isa Kirstine.FWDummyWorkspace
            @test w_ad isa Kirstine.AdaptiveWorkspace
            @test w_ad.x_tmp isa Pnt
            @test w_ad.d_tmp isa PntDiff
            @test length(w_ad.gx_tmp) == 3
            @test all(isnan, w_ad.gx_tmp)
        end
    end

    @testset "optimizer_state" begin
        let f(p) = -(p.x[1] - 1)^2,
            g!(grad, p) = grad[1] = -2 * (p.x[1] - 1),
            gnan!(grad, p) = grad[1] = NaN,
            ginf!(grad, p) = grad[1] = Inf,
            gninf!(grad, p) = grad[1] = -Inf,
            opt = FrankWolfe(5, Kirstine.OpenLoop(2)),
            # |------|------|
            # 0   x0=5     10
            x0 = Pnt([5]),
            c = BoxConstraints(Pnt([0]), Pnt([10])),
            state = Kirstine.optimizer_state(f, g!, opt, [x0], c)

            @test_throws "exactly 1" Kirstine.optimizer_state(f, g!, opt, [x0, x0], c)
            @test_throws "NaN" Kirstine.optimizer_state(f, gnan!, opt, [x0], c)
            @test_throws "Inf" Kirstine.optimizer_state(f, ginf!, opt, [x0], c)
            @test_throws "Inf" Kirstine.optimizer_state(f, gninf!, opt, [x0], c)
            @test state isa Kirstine.FrankWolfeState{Pnt,PntDiff}
            # we should not share memory with given arguments
            @test state.x == c.lb
            @test state.x !== c.lb
            @test state.x !== x0
            @test state.v == c.lb
            @test state.diff.v == [-5]
            @test state.stepsize == 1
            @test state.fx == -1
            @test state.gx == [-8]
            @test state.n_eval_f == 1
            @test state.n_eval_g == 1
        end
    end

    @testset "tick!" begin
        let opt = FrankWolfe(2, Kirstine.OpenLoop(2)),
            n = 2,
            xstar = ones(n),
            f(p) = -sum((p.x .- xstar) .^ 2),
            g!(grad, p) = grad .= -2 .* (p.x .- xstar),
            c = BoxConstraints(Pnt([0, 0]), Pnt([10, 10])),
            state = Kirstine.optimizer_state(f, g!, opt, [Pnt([5, 0.5])], c),
            iter = 2

            Kirstine.tick!(f, g!, state, opt, c, iter)
            @test state.gx == [2, -18]
            @test state.v == Pnt([10, 0])
            @test state.stepsize ≈ 2 / 3
            @test state.x.x ≈ [20 / 3, 10 / 3]
            @test state.n_eval_f == 2
            @test state.n_eval_g == 2
        end
    end

    @testset "optimize" begin
        let n = 4,
            xstar = ones(n),
            f(p) = -sum((p.x .- xstar) .^ 2),
            g!(grad, p) = grad .= -2 .* (p.x .- xstar),
            fw = FrankWolfe(; iterations = 2000, step = Kirstine.OpenLoop(2)),
            prototype = Pnt(zeros(n)),
            outside = Pnt([-1, -1, 5, -1]),
            constr1 = BoxConstraints(Pnt(fill(-1, n)), Pnt(fill(2, n))),
            constr2 = BoxConstraints(Pnt(fill(-1, n)), Pnt(collect(1:n) ./ 2))

            # This solution will be at xstar, inside the constraints...
            r1 = Kirstine.optimize(f, g!, fw, [prototype], constr1)
            # ... and this one will be on the boundary at [0.5, 1, 1, ..., 1]
            r2 = Kirstine.optimize(f, g!, fw, [prototype], constr2; trace_state = true)

            @test r1.maximizer.x ≈ xstar rtol = 1e-4
            @test r1.maximum ≈ 0.0 atol = 1e-4
            @test length(r1.trace_gx) == 2000
            @test r1.trace_gx[end] ≈ zeros(n) atol = 1e-2 # local maximum
            @test r2.maximizer.x ≈ [0.5; xstar[2:end]] rtol = 1e-4
            @test r2.maximum ≈ -0.25 rtol = 1e-4
            @test length(r2.trace_gx) == 2000
            @test r2.trace_gx[end] ≈ [1, 0, 0, 0] rtol = 1e-2 # not a local maximum
        end
    end
end
end
