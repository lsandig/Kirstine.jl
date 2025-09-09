# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module SolveExchangeTests
using Test
using Random: seed!
using Kirstine

include("example-emax.jl")

@testset "solve-exchange.jl" begin
    @testset "Exchange" begin
        let st = Pso(; swarmsize = 10, iterations = 10)
            @test_throws "steps must be >= 1" Exchange(
                optimizer_direction = st,
                optimizer_weight = st,
                steps = 0,
                gateaux_tolerance = 1e-3,
                initial_weight = NaN,
                candidate = one_point_design([42]),
            )
            @test_throws "gateaux_tolerance must be >= 0" Exchange(
                optimizer_direction = st,
                optimizer_weight = st,
                steps = 1,
                gateaux_tolerance = -1,
                initial_weight = NaN,
                candidate = one_point_design([42]),
            )
            @test_throws "initial_weight must be in [0, 1) or NaN" Exchange(
                optimizer_direction = st,
                optimizer_weight = st,
                steps = 1,
                gateaux_tolerance = 1e-3,
                initial_weight = 1,
                candidate = one_point_design([42]),
            )
        end
    end

    @testset "solve_with(::Pso, ::Pso)" begin
        let dp = DesignProblem(;
                criterion = DCriterion(),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
                region = DesignInterval(:dose => (0, 10)),
            ),
            od = Pso(; iterations = 20, swarmsize = 10),
            ow = Pso(; iterations = 20, swarmsize = 10),
            str1 = Exchange(;
                optimizer_weight = ow,
                optimizer_direction = od,
                candidate = uniform_design([[5], [0], [10]]),
                steps = 3,
                gateaux_tolerance = 0,
                initial_weight = NaN,
                simplify_args = Dict(:maxweight => 1e-3),
            ),
            _ = seed!(54321),
            r1 = Kirstine.solve_with(dp, str1, true),
            # When the direction of steepest ascent is in the support of the candidate, the
            # number of design points of the intermediate design measure should not
            # increase. When merged, the new point should move to the front of the vector of
            # design points. We check this here by starting with a candidate that has the
            # correct design points, but unequal weights.
            str2 = Exchange(;
                optimizer_weight = ow,
                optimizer_direction = od,
                candidate = DesignMeasure([[0], [2.5], [10]], [0.6, 0.3, 0.1]),
                gateaux_tolerance = 0,
                initial_weight = NaN,
                steps = 1,
            ),
            _ = seed!(13579),
            r2 = Kirstine.solve_with(dp, str2, false),
            # prototypes incompatible with design region
            str3 = Exchange(;
                optimizer_weight = ow,
                optimizer_direction = od,
                candidate = uniform_design([[0], [20]]),
                steps = 1,
                gateaux_tolerance = 0,
                initial_weight = NaN,
            ),
            str4 = Exchange(;
                optimizer_weight = ow,
                optimizer_direction = od,
                candidate = uniform_design([[0, 1], [1, 0]]),
                steps = 1,
                gateaux_tolerance = 0,
                initial_weight = NaN,
            ),
            # initial design is singular
            str5 = Exchange(;
                optimizer_weight = ow,
                optimizer_direction = od,
                candidate = one_point_design([5]),
                steps = 1,
                gateaux_tolerance = 0,
                initial_weight = NaN,
            )

            # errors from check_compatible
            @test_throws "outside design region" (@test_warn "dp =" solve(dp, str3))
            @test_throws "must match" solve(dp, str4)
            # singular candidate design
            @test_throws "step 1: candidate design is singular" solve(dp, str5)
            # result type
            @test isa(r1, Kirstine.ExchangeResult)
            @test isa(r2, Kirstine.ExchangeResult)
            # traced state
            @test all(
                or -> length(or.trace_state) == 20,
                optimization_results_direction(r1),
            )
            @test all(or -> length(or.trace_state) == 20, optimization_results_weight(r1))
            @test all(or -> length(or.trace_state) == 0, optimization_results_direction(r2))
            @test all(or -> length(or.trace_state) == 0, optimization_results_weight(r2))
            # increasing objective (from step to step)
            @test issorted(optimization_results_weight(r1), by = or -> or.maximum)
            @test issorted(optimization_results_weight(r2), by = or -> or.maximum)
            # no new point, but last one of the candidate
            @test numpoints(optimization_results_weight(r2)[1].maximizer) == 3
            @test points(optimization_results_weight(r2)[1].maximizer)[1] == [10]
            # test that runtimes can be accessed
            rt_direction = total_runtime_direction(r1)
            rt_weight = total_runtime_weight(r1)
            @test rt_direction > 0
            @test rt_weight > 0
            @test total_runtime(r1) == rt_direction + rt_weight
        end
    end

    @testset "solve_with(::GridSearch, ::Pso)" begin
        let guess = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            dp = DesignProblem(;
                criterion = DCriterion(),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([guess]),
                region = DesignInterval(:dose => (0, 10)),
            ),
            exact_solution = emax_solution(guess, region(dp)),
            # grid that contains the points of the solution [[0], [2.5], [10]]
            pg101 = Kirstine.ProductGrid(101),
            str1 = Exchange(;
                optimizer_weight = Pso(; iterations = 40, swarmsize = 20),
                optimizer_direction = Kirstine.GridSearch(; gridspec = pg101),
                candidate = uniform_design([[5], [0], [10]]),
                steps = 5,
                gateaux_tolerance = 0,
                initial_weight = NaN,
            ),
            # grid that does not contain [2.5]
            pg11 = Kirstine.ProductGrid(11),
            str2 = Exchange(;
                optimizer_weight = Pso(; iterations = 40, swarmsize = 20),
                optimizer_direction = Kirstine.GridSearch(; gridspec = pg11),
                candidate = uniform_design([[5], [0], [10]]),
                steps = 5,
                gateaux_tolerance = 0,
                initial_weight = NaN,
            ),
            # grid does not contain [2.5], but a prototype point is off-grid, single step
            str3 = Exchange(;
                optimizer_weight = Pso(; iterations = 40, swarmsize = 20),
                optimizer_direction = Kirstine.GridSearch(; gridspec = pg11),
                candidate = uniform_design([[2.9], [0], [10]]),
                steps = 1,
                gateaux_tolerance = 0,
                initial_weight = NaN,
            ),
            # check that early stopping works
            str4 = Exchange(;
                optimizer_weight = Pso(; iterations = 40, swarmsize = 20),
                optimizer_direction = Kirstine.GridSearch(; gridspec = pg101),
                candidate = uniform_design([[5], [0], [10]]),
                steps = 10,
                gateaux_tolerance = -3 * log(0.999),
                initial_weight = NaN,
            ),
            # but one step is performed even if the candidate is optimal
            str5 = Exchange(;
                optimizer_weight = Pso(; iterations = 40, swarmsize = 20),
                optimizer_direction = Kirstine.GridSearch(; gridspec = pg101),
                candidate = exact_solution,
                steps = 10,
                gateaux_tolerance = -3 * log(0.999),
                initial_weight = NaN,
            ),
            # initial weights that do _not_ depend on the support size of the candidate
            str6 = Exchange(;
                optimizer_weight = Pso(; iterations = 40, swarmsize = 20),
                optimizer_direction = Kirstine.GridSearch(; gridspec = pg101),
                candidate = uniform_design([[5], [0], [10]]),
                steps = 1,
                gateaux_tolerance = 0,
                initial_weight = 0,
            )

            seed!(1234)
            r1 = Kirstine.solve_with(dp, str1, true)
            sol1 = solution(r1)
            @test [2.5] in points(sol1)
            @test efficiency(sol1, exact_solution, dp) > 1 - 1e-10
            ord_step_1 = first(optimization_results_direction(r1))
            @test ord_step_1.trace_state isa Vector{<:Kirstine.GridSearchState}
            @test ord_step_1.maximizer == one_point_design([2.2])
            @test length(ord_step_1.trace_state[1].fx) == 101
            orw_step_1 = first(optimization_results_weight(r1))
            # prototype has K=3 points, initial_weight = NaN implies new point in front hast weight 1/(k+1) = 0.25
            # Note that we can't test this with Frank-Wolfe since there we don't trace the initial prototype.
            @test weights(orw_step_1.trace_state[1].x[1]) ≈ fill(0.25, 4)

            seed!(1234)
            r2 = Kirstine.solve_with(dp, str2, true)
            sol2 = solution(r2)
            @test points(sort_points(sol2)) == [[0], [2], [3], [10]]
            @test maximum(last(optimization_results_direction(r2)).trace_state[1].fx) < 1e-3

            seed!(1234)
            r3 = Kirstine.solve_with(dp, str3, true)
            sol3 = solution(r3)
            @test length(first(optimization_results_direction(r3)).trace_state[1].fx) == 12
            @test [2.9] in points(sol3)

            seed!(1234)
            r4 = Kirstine.solve_with(dp, str4, false)
            @test efficiency(solution(r4), exact_solution, dp) > 1 - 1e-8
            @test length(optimization_results_weight(r4)) == 4
            @test length(optimization_results_direction(r4)) == 4

            seed!(1234)
            r5 = Kirstine.solve_with(dp, str5, false)
            @test length(optimization_results_weight(r5)) == 1
            @test length(optimization_results_direction(r5)) == 1

            seed!(1234)
            r6 = Kirstine.solve_with(dp, str6, true)
            orw_step_1 = first(optimization_results_weight(r6))
            # prototype has K=3 points, initial_weight = 0 is passed through
            @test weights(orw_step_1.trace_state[1].x[1]) ≈ [0, 1/3, 1/3, 1/3]
        end
    end

    @testset "solve_with(::GridSearch, ::FrankWolfe)" begin
        let guess = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            dp = DesignProblem(;
                criterion = DCriterion(),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([guess]),
                region = DesignInterval(:dose => (0, 10)),
            ),
            pg101 = Kirstine.ProductGrid(101),
            str = Exchange(;
                optimizer_weight = Kirstine.FrankWolfe(
                    iterations = 100,
                    stepsize = Kirstine.Adaptive(; initial_smoothness_estimate = 1),
                    pairwise = true,
                    fw_gap_tolerance = 1e-4,
                ),
                optimizer_direction = Kirstine.GridSearch(; gridspec = pg101),
                candidate = uniform_design([[5], [0], [10]]),
                steps = 5,
                gateaux_tolerance = 0,
                initial_weight = NaN,
            )

            @test_logs(
                (:warn, "current design is singular"),
                (:error, r"singular exception in adaptive step-size computation"),
                @test_throws Kirstine.AdaptiveStepsizeException solve(dp, str)
            )
        end
    end
end
end
