# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignproblemTests
using Test
using Kirstine
using Random: seed!

include("example-emax.jl")

@testset "designproblem.jl" begin
    @testset "solve" begin
        # check that solution is sorted and simplified
        let dp = DesignProblem(;
                design_criterion = DOptimality(),
                design_region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            str = DirectMaximization(;
                optimizer = Pso(; swarmsize = 10, iterations = 20),
                prototype = uniform_design([[0], [2.45], [2.55], [10]]),
            ),
            _ = seed!(1234),
            (d, r) = solve(dp, str; trace_state = true, mindist = 1e-1)

            # design points should be sorted...
            @test issorted(reduce(vcat, designpoints(d)))
            # ... and simplified
            @test length(designpoints(d)) == 3
            # states should be traced (initial + 20 iterations)
            @test length(r.or.trace_state) == 21

            # `minposdist` doesn't exist, the correct argument name is `mindist`. Because we
            # have not implemented `simplify_unique()` for EmaxModel, the generic method
            # should complain about gobbling up `minposdist` in its varargs. (We can't test
            # this in designmeasure.jl because we need a Model and a
            # CovariateParameterization in order to call `simplify()`.)
            @test_logs(
                (
                    :warn,
                    "unused keyword arguments given to generic `simplify_unique` method",
                ),
                solve(dp, str, minposdist = 1e-2)
            )
        end
    end
end
end
