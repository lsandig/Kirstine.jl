# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module SolveExchangeTests
using Test
using Random: seed!
using Kirstine

include("example-emax.jl")

@testset "solve-exchange.jl" begin
    @testset "solve_with" begin
        let dp = DesignProblem(;
                design_criterion = DOptimality(),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
                design_region = DesignInterval(:dose => (0, 10)),
            ),
            od = Pso(; iterations = 20, swarmsize = 10),
            ow = Pso(; iterations = 20, swarmsize = 10),
            str1 = Exchange(;
                ow = ow,
                od = od,
                candidate = uniform_design([[5], [0], [10]]),
                steps = 3,
            ),
            _ = seed!(54321),
            r1 = Kirstine.solve_with(dp, str1, true),
            # When the direction of steepest ascent is in the support of the candidate, the
            # number of design points of the intermediate design measure should not
            # increase. When merged, the new point should move to the front of the vector of
            # design points. We check this here by starting with a candidate that has the
            # correct design points, but unequal weights.
            str2 = Exchange(;
                ow = ow,
                od = od,
                candidate = DesignMeasure([[0], [2.5], [10]], [0.6, 0.3, 0.1]),
                steps = 1,
            ),
            _ = seed!(13579),
            r2 = Kirstine.solve_with(dp, str2, false),
            # prototypes incompatible with design region
            str3 = Exchange(;
                ow = ow,
                od = od,
                candidate = uniform_design([[0], [20]]),
                steps = 1,
            ),
            str4 = Exchange(;
                ow = ow,
                od = od,
                candidate = uniform_design([[0, 1], [1, 0]]),
                steps = 1,
            )

            # errors from check_compatible
            @test_throws "outside design region" solve(dp, str3)
            @test_throws "must match" solve(dp, str4)
            # result type
            @test isa(r1, ExchangeResult)
            @test isa(r2, ExchangeResult)
            # traced state
            @test all(or -> length(or.trace_state) == 21, r1.ord)
            @test all(or -> length(or.trace_state) == 21, r1.orw)
            @test all(or -> length(or.trace_state) == 1, r2.ord)
            @test all(or -> length(or.trace_state) == 1, r2.orw)
            # increasing objective (from step to step)
            @test issorted(r1.orw, by = or -> or.maximum)
            @test issorted(r2.orw, by = or -> or.maximum)
            # no new point, but last one of the candidate
            @test numpoints(r2.orw[1].maximizer) == 3
            @test points(r2.orw[1].maximizer)[1] == [10]
        end
    end
end
end
