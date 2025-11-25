# SPDX-FileCopyrightText: 2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module FrankWolfeTests
using Kirstine, Test
using LinearAlgebra

include("example-pnt.jl")

function Kirstine.fw_linear_maximization_oracle!(
    v::Pnt,
    gx::AbstractVector{<:Real},
    c::BoxConstraints,
)
    box_corners = Iterators.product(Iterators.zip(c.lb.x, c.ub.x)...)
    # Workaround for https://github.com/domluna/JuliaFormatter.jl/issues/926
    #! format: off
    v.x .= argmax(corner -> mapreduce(*, +, gx, corner), box_corners)
    #! format: on
    return v
end

function Kirstine.fw_evaluate_linearization(
    gx::AbstractVector{<:Real},
    p::PntDiff,
    c::BoxConstraints,
)
    return dot(gx, p.v)
end

function Kirstine.fw_evaluate_linearization(
    gx::AbstractVector{<:Real},
    p::Pnt,
    c::BoxConstraints,
)
    return dot(gx, p.x)
end

function Kirstine.fw_gradient_length(p::Pnt, c::BoxConstraints)
    return length(p.x)
end

function Kirstine.fw_norm(d::PntDiff)
    return norm(d.v)
end

function Kirstine.ap_convex_combination!(
    p::Pnt,
    as::AbstractVector{<:Real},
    qs::AbstractVector{Pnt},
    c::BoxConstraints,
)
    p.x .= 0
    for i in 1:length(as)
        p.x .+= as[i] .* qs[i].x
    end
    p.x .= max.(c.lb.x, p.x)
    p.x .= min.(c.ub.x, p.x)
    return p
end

@testset "optimizer-frank-wolfe.jl" begin
    @testset "OpenLoop" begin
        @test_throws ">= 1" OpenLoop(0)
    end

    @testset "Adaptive" begin
        @test_throws "eta must be <= 1" Adaptive(; eta = 2)
        @test_throws "tau must be > 1" Adaptive(; tau = 1)
        @test_throws "threshold must be > 0" Adaptive(; smoothness_threshold = 0)
        @test_throws "(0, threshold]" Adaptive(; initial_smoothness_estimate = 2e10)
        @test_throws "(0, threshold]" Adaptive(; initial_smoothness_estimate = -1)
    end

    @testset "FrankWolfe" begin
        # check both positional and keyword-based constructor
        msg = "number of iterations must be positive"
        @test_throws msg FrankWolfe(-1, 1e-3, OpenLoop(2), false)
        @test_throws msg FrankWolfe(;
            iterations = -1,
            fw_gap_tolerance = 1e-3,
            stepsize = OpenLoop(2),
            pairwise = false,
        )
        @test_throws "fw_gap_tolerance must be non-negative" FrankWolfe(;
            iterations = 10,
            fw_gap_tolerance = -1,
            stepsize = OpenLoop(2),
            pairwise = false,
        )
    end

    @testset "stepsize_openloop" begin
        @test Kirstine.stepsize_openloop(2, 0, 1) == 1
        @test Kirstine.stepsize_openloop(5, 0, 1) == 1
        @test Kirstine.stepsize_openloop(2, 3, 1) == 0.4
        @test Kirstine.stepsize_openloop(5, 5, 1) == 0.5
        @test Kirstine.stepsize_openloop(5, 5, 0.2) == 0.2
    end

    @testset "stepsize_constnumiter" begin
        @test Kirstine.stepsize_constnumiter(9, 0, 1) == 1 / 3
        @test Kirstine.stepsize_constnumiter(16, 0, 1) == 1 / 4
        @test Kirstine.stepsize_constnumiter(9, 2, 1) == 1 / 3
        @test Kirstine.stepsize_constnumiter(16, 3, 1) == 1 / 4
        @test Kirstine.stepsize_constnumiter(16, 3, 0.2) == 0.2

        @test_throws "past maximum" Kirstine.stepsize_constnumiter(9, 10, 1)
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
            w = Kirstine.AdaptiveWorkspace(NaN, NaN, x_tmp, d_tmp, gx_tmp, 0, false)

            res = Kirstine.fw_estimate_initial_smoothness!(g!, w, x, gx, d, c, 1)

            @test w.smoothness_estimate ≈ 2
            @test res == w
        end
    end

    @testset "stepsize!(::Adaptive)" begin
        let n = 4,
            xstar = ones(n),
            f(p) = -sum((p.x .- xstar) .^ 2),
            g!(grad, p) = grad .= -2 .* (p.x .- xstar),
            c = BoxConstraints(Pnt(fill(-1, n)), Pnt(fill(10, n))),
            x = Pnt(3:6),
            gx = g!(zeros(4), x),
            v = Kirstine.fw_linear_maximization_oracle!(Pnt(fill(NaN, 4)), gx, c),
            original_diff = v.x - x.x,
            d = PntDiff(fill(NaN, 4))

            # We can only test stepsize() indirectly, since building the optimizer state already
            # performs the first full tick!().
            a1 = Adaptive(; initial_smoothness_estimate = 1, eta = 0.8, tau = 1.5)
            fw1 = FrankWolfe(1, 0, a1, false)
            s1 = Kirstine.optimizer_state((f, g!), fw1, [x], c)
            # To get from 0.8 * 1 to over 2, we need to multiply by 1.5 3 times and get 2.7
            @test s1.work.smoothness_estimate ≈ 2.7
            # 1 from first tick!(), 4 from while loop in stepsize!()
            @test s1.n_eval_g == 5
            @test s1.stepsize == dot(gx, original_diff) / (2.7 * norm(original_diff)^2)

            # Since step-size computation depends only on the smoothness estimate and constants,
            # the other cases do not differ in this respect.
            # For them we only check the bookkeeping of the number of gradient evaluations.
            a2 = Adaptive(; initial_smoothness_estimate = 3, eta = 0.8, tau = 1.5)
            fw2 = FrankWolfe(1, 0, a2, false)
            s2 = Kirstine.optimizer_state((f, g!), fw2, [x], c)
            # 0.8 * 3 = 2.4 is already larger than 2, so this is the estimate
            @test s2.work.smoothness_estimate ≈ 2.4
            # 1 from first tick!(), 1 from while loop in stepsize!()
            @test s2.n_eval_g == 2

            a3 = Adaptive(; initial_smoothness_estimate = Inf, eta = 0.9, tau = 2)
            fw3 = FrankWolfe(1, 0, a3, false)
            s3 = Kirstine.optimizer_state((f, g!), fw3, [x], c)
            # the initial estimate is ≈ 2, which becomes too small as 2 * 0.9 = 1.8,
            # but doubling once again is sufficient
            @test s3.work.smoothness_estimate ≈ 3.6
            # 1 from first tick!(), 1 from initial estimation, 2 from while loop in stepsize!()
            @test s3.n_eval_g == 4
        end
    end

    @testset "fw_workspace" begin
        let w_ol = Kirstine.fw_workspace(OpenLoop(2), Pnt([0]), 1),
            w_cni = Kirstine.fw_workspace(ConstNumIter(), Pnt([0]), 1),
            w_ad = Kirstine.fw_workspace(Adaptive(), Pnt([0, 1, 2]), 3)

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
            h(state) = Kirstine.check_gradient(state; only_warn = false),
            opt = FrankWolfe(5, 0, OpenLoop(2), false),
            # |------|------|
            # 0   x0=5     10
            x0 = Pnt([5]),
            c = BoxConstraints(Pnt([0]), Pnt([10])),
            state = Kirstine.optimizer_state((f, g!), opt, [x0], c)

            @test_throws "exactly 1" Kirstine.optimizer_state((f, g!), opt, [x0, x0], c)
            @test_throws "NaN" h(Kirstine.optimizer_state((f, gnan!), opt, [x0], c))
            @test_throws "Inf" h(Kirstine.optimizer_state((f, ginf!), opt, [x0], c))
            @test_throws "Inf" h(Kirstine.optimizer_state((f, gninf!), opt, [x0], c))
            @test state isa Kirstine.FrankWolfeState{Pnt,PntDiff}
            # we should not share memory with given arguments
            @test state.x == c.lb
            @test state.x !== c.lb
            @test state.x !== x0
            @test state.v == c.lb
            @test state.direction.v == [-5]
            @test state.stepsize == 1
            @test state.fx == -1
            @test state.gx == [-8]
            @test state.n_eval_f == 2
            @test state.n_eval_g == 1
            @test state.active_points_added == 1
            @test state.active_points_removed == 0
            @test state.active_set == [x0, c.lb]
        end
    end

    @testset "tick!" begin
        let opt = FrankWolfe(2, 0, OpenLoop(2), false),
            n = 2,
            xstar = ones(n),
            f(p) = -sum((p.x .- xstar) .^ 2),
            g!(grad, p) = grad .= -2 .* (p.x .- xstar),
            c = BoxConstraints(Pnt([0, 0]), Pnt([10, 10])),
            state = Kirstine.optimizer_state((f, g!), opt, [Pnt([5, 0.5])], c)

            Kirstine.tick!((f, g!), state, opt, c)
            @test state.iteration_from_zero == 1
            @test state.gx == [2, -18]
            @test state.v == Pnt([10, 0])
            @test state.stepsize ≈ 2 / 3
            @test state.x.x ≈ [20 / 3, 10 / 3]
            @test state.n_eval_f == 3
            @test state.n_eval_g == 2
        end
    end

    @testset "optimize" begin
        let n = 4,
            xstar = ones(n),
            f(p) = -sum((p.x .- xstar) .^ 2),
            g!(grad, p) = grad .= -2 .* (p.x .- xstar),
            fw = FrankWolfe(2000, 0, OpenLoop(2), false),
            pfw = FrankWolfe(2000, 0, Adaptive(; initial_smoothness_estimate = 2), true),
            et = FrankWolfe(2000, 1e-7, Adaptive(; initial_smoothness_estimate = 2), true),
            prototype = Pnt(zeros(n)),
            outside = Pnt([-1, -1, 5, -1]),
            constr1 = BoxConstraints(Pnt(fill(-1, n)), Pnt(fill(2, n))),
            constr2 = BoxConstraints(Pnt(fill(-1, n)), Pnt(collect(1:n) ./ 2))

            # This solution will be at xstar, inside the constraints...
            r1 = Kirstine.optimize((f, g!), fw, [prototype], constr1)
            # ... and this one will be on the boundary at [0.5, 1, 1, ..., 1]
            r2 = Kirstine.optimize((f, g!), fw, [prototype], constr2; trace_state = true)
            # paired steps will be better at going to the boundary
            r3 = Kirstine.optimize((f, g!), pfw, [prototype], constr2; trace_state = true)
            # early termination
            r4 = Kirstine.optimize((f, g!), et, [prototype], constr2; trace_state = true)

            @test !r1.has_converged
            @test r1.maximizer.x ≈ xstar rtol = 1e-4
            @test r1.maximum ≈ 0.0 atol = 1e-4
            @test length(r1.trace_gx) == 2000
            @test r1.n_eval_g == 2000
            # initial and subsequent gradient traces should not share memory
            @test !(r1.trace_gx[1] === r1.trace_gx[2])
            @test !(r1.trace_gx[2] === r1.trace_gx[end])
            @test r1.trace_gx[end] ≈ zeros(n) atol = 1e-2 # local maximum
            @test !r2.has_converged # we compare with < against 0
            @test r2.maximizer.x ≈ [0.5; xstar[2:end]] rtol = 1e-4
            @test r2.maximum ≈ -0.25 rtol = 1e-4
            @test length(r2.trace_gx) == 2000
            @test r2.trace_gx[end] ≈ [1, 0, 0, 0] rtol = 1e-2 # not a local maximum
            # adaptive has more gradient evaluations
            @test r3.n_eval_g > length(r3.trace_gx)
            # pairwise steps converge faster:
            @test r3.maximum > r2.maximum
            @test r3.trace_state[end].frank_wolfe_gap < r2.trace_state[end].frank_wolfe_gap
            @test norm(r3.maximizer.x .- xstar) < norm(r2.maximizer.x .- xstar)
            @test r3.trace_state[end].active_points_added == 6
            @test r3.trace_state[end].active_points_removed == 3
            @test r3.trace_state[end].frank_wolfe_gap == 0
            # and even faster still with early termination
            @test r4.has_converged
            @test length(r4.trace_fx) == 44
            @test r4.trace_state[end].frank_wolfe_gap <= 1e-7
        end
    end
end
end
