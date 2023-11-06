# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module SolveDirectmaximizationTests
using Test
using Kirstine

include("example-emax.jl")

@testset "solve-directmaximization.jl" begin
    @testset "solve_with" begin
        let dp = DesignProblem(;
                criterion = DOptimality(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            pso = Pso(; iterations = 20, swarmsize = 10),
            # random init, no additional constraints
            str1 = DirectMaximization(;
                optimizer = pso,
                prototype = random_design(region(dp), 3),
            ),
            # uniform init, weights and some points fixed
            str2 = DirectMaximization(;
                optimizer = pso,
                prototype = equidistant_design(region(dp), 3),
                fixedweights = 1:3,
                fixedpoints = [1, 3],
            ),
            # prototypes incompatible with design region
            str3 = DirectMaximization(;
                optimizer = pso,
                prototype = uniform_design([[0], [20]]),
            ),
            str4 = DirectMaximization(;
                optimizer = pso,
                prototype = uniform_design([[0, 1], [1, 0]]),
            ),
            r1 = Kirstine.solve_with(dp, str1, false),
            # additionally trace optimizer states
            r2 = Kirstine.solve_with(dp, str2, true),
            design_trace = optimization_result(r2).trace_x,
            lower_dp_vals = map(d -> points(d)[1][1], design_trace),
            upper_dp_vals = map(d -> points(d)[3][1], design_trace),
            middle_dp_dists = map(d -> abs(points(d)[2][1] - 2.5), design_trace)

            # errors from check_compatible() in DesignConstraints constructor
            @test_throws "outside design region" (@test_warn "dp =" solve(dp, str3))
            @test_throws "must match" solve(dp, str4)
            # result type
            @test isa(r1, DirectMaximizationResult)
            @test isa(r2, DirectMaximizationResult)
            # traced state
            @test length(optimization_result(r1).trace_state) == 1
            @test length(optimization_result(r2).trace_state) == 20
            # increasing objective
            @test issorted(optimization_result(r1).trace_fx)
            @test issorted(optimization_result(r2).trace_fx)
            # fixed things
            @test all(map(d -> all(weights(d) .≈ 1 / 3), design_trace))
            @test all(lower_dp_vals .== lowerbound(region(dp))[1])
            @test all(upper_dp_vals .== upperbound(region(dp))[1])
            # non-fixed design point converges to 2.5
            @test issorted(middle_dp_dists, rev = true)
        end

        # DirectMaximization, but with fixed weights / design points that we know are not
        # optimal, in various combinations
        let dp = DesignProblem(;
                criterion = DOptimality(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            pt = DesignMeasure([[0], [5], [7], [8], [10]], [0.1, 0.5, 0.0, 0.0, 0.4]),
            sws(; fw, fp) = Kirstine.solve_with(
                dp,
                DirectMaximization(;
                    optimizer = Pso(; iterations = 10, swarmsize = 5),
                    prototype = pt,
                    fixedweights = fw,
                    fixedpoints = fp,
                ),
                true,
            ),
            r1 = sws(; fw = [2], fp = [2]),
            r2 = sws(; fw = [2], fp = Int64[]),
            r3 = sws(; fw = Int64[], fp = [2]),
            r4 = sws(; fw = [5], fp = [5]),
            r5 = sws(; fw = [1, 5], fp = [5]),
            is_const_fn(acc, r, k) = all([
                all(map(d -> acc(d)[k] == acc(pt)[k], s.x)) for
                s in optimization_result(r).trace_state
            ])

            @test is_const_fn(weights, r1, 2)
            @test is_const_fn(points, r1, 2)
            @test is_const_fn(weights, r2, 2)
            @test is_const_fn(points, r3, 2)
            @test is_const_fn(weights, r4, 5)
            @test is_const_fn(points, r4, 5)
            @test is_const_fn(weights, r5, 1)
            @test is_const_fn(weights, r5, 5)
            @test is_const_fn(points, r5, 5)
            @test_logs(
                (:warn, "fixed weights already sum to one"),
                match_mode = :any,
                sws(fw = [1, 2, 5], fp = Int64[])
            )
        end
    end
end
end
