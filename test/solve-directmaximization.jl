# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module SolveDirectmaximizationTests
using Test
using Kirstine

include("example-emax.jl")

@testset "solve-directmaximization.jl" begin
    @testset "solve_with" begin
        let dp = DesignProblem(;
                design_criterion = DOptimality(),
                design_region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            pso = Pso(; iterations = 20, swarmsize = 10),
            # random init, no additional constraints
            str1 =
                DirectMaximization(; optimizer = pso, prototype = random_design(dp.dr, 3)),
            # uniform init, weights and some points fixed
            str2 = DirectMaximization(;
                optimizer = pso,
                prototype = equidistant_design(dp.dr, 3),
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
            r2 = Kirstine.solve_with(dp, str2, true)

            # errors from check_compatible() in DesignConstraints constructor
            @test_throws "outside design region" solve(dp, str3)
            @test_throws "must match" solve(dp, str4)
            # result type
            @test isa(r1, DirectMaximizationResult)
            @test isa(r2, DirectMaximizationResult)
            # traced state
            @test length(r1.or.trace_state) == 1
            @test length(r2.or.trace_state) == 21
            # increasing objective
            @test issorted(r1.or.trace_fx)
            @test issorted(r2.or.trace_fx)
            # fixed things
            @test all(map(d -> all(weights(d) .≈ 1 / 3), r2.or.trace_x))
            @test all(map(d -> points(d)[1][1], r2.or.trace_x) .== dp.dr.lowerbound[1])
            @test all(map(d -> points(d)[3][1], r2.or.trace_x) .== dp.dr.upperbound[1])
            # non-fixed design point converges to 2.5
            @test issorted(map(d -> abs(points(d)[2][1] - 2.5), r2.or.trace_x), rev = true)
        end

        # DirectMaximization, but with fixed weights / design points that we know are not
        # optimal, in various combinations
        let dp = DesignProblem(;
                design_criterion = DOptimality(),
                design_region = DesignInterval(:dose => (0, 10)),
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
            isconstw(r, k) = all([
                all(map(d -> weights(d)[k] == weights(pt)[k], s.x)) for
                s in r.or.trace_state
            ]),
            isconstp(r, k) = all([
                all(map(d -> points(d)[k] == points(pt)[k], s.x)) for s in r.or.trace_state
            ])

            @test isconstw(r1, 2)
            @test isconstp(r1, 2)
            @test isconstw(r2, 2)
            @test isconstp(r3, 2)
            @test isconstw(r4, 5)
            @test isconstp(r4, 5)
            @test isconstw(r5, 1)
            @test isconstw(r5, 5)
            @test isconstp(r5, 5)
            @test_logs(
                (:warn, "fixed weights already sum to one"),
                match_mode = :any,
                sws(fw = [1, 2, 5], fp = Int64[])
            )
        end
    end
end
end
