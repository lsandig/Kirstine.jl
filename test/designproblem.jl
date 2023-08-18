# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignproblemTests
using Test
using Kirstine
using Random: seed!

include("example-emax.jl")
include("example-compartment.jl")

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
    @testset "efficiency" begin
        # Atkinson et al. example
        let _ = seed!(4711),
            # prior guess for locally optimal design
            g0 = PriorSample([TPCPar(; a = 4.298, e = 0.05884, s = 21.80)]),
            # a draw from the strongly informative prior
            g1 = draw_from_prior(1000, 2),
            t_id = Identity(),
            t_auc = DeltaMethod(Dauc),
            dp_for(pk, trafo) = DesignProblem(;
                design_criterion = DOptimality(),
                # not used in efficiency calculation!
                design_region = DesignInterval(:time => [0, 48]),
                model = TPCMod(1),
                covariate_parameterization = CopyTime(),
                prior_knowledge = pk,
                transformation = trafo,
            ),
            # singular locally optimal designs from Table 4
            a3 = DesignMeasure([0.1793] => 0.6062, [3.5671] => 0.3938),
            a4 = DesignMeasure([1.0122] => 1.0),
            dp1 = dp_for(g0, t_id),
            dp2 = dp_for(g0, t_auc),
            # Bayesian optimal designs from Table 5
            a6 = DesignMeasure([0.2288] => 1 / 3, [1.4170] => 1 / 3, [18.4513] => 1 / 3),
            a7 = DesignMeasure([0.2449] => 0.0129, [1.4950] => 0.0387, [18.4903] => 0.9484),
            dp6 = dp_for(g1, t_id),
            dp7 = dp_for(g1, t_auc)

            # Compare with published efficiencies in Table 5. Due to Monte-Carlo uncertainty
            # and rounding of the published values, this is not very exact.
            @test efficiency(a7, a6, dp6) ≈ 0.234 atol = 1e-2
            @test efficiency(a6, a7, dp7) ≈ 0.370 atol = 1e-2
            # Check that singular designs are handled correctly both with Identity and with
            # DeltaMethod transformation.
            @test efficiency(a4, a6, dp1) == 0
            @test efficiency(a4, a6, dp2) == 0
            @test efficiency(a6, a4, dp1) == Inf
            @test efficiency(a6, a4, dp2) == Inf
            @test isnan(efficiency(a3, a4, dp1))
            @test isnan(efficiency(a3, a4, dp2))
            # Note: although a mathematical argument could be made that the efficiency should be
            # equal to 1 in the following case, we still want it to be NaN:
            @test isnan(efficiency(a4, a4, dp1))
            @test isnan(efficiency(a4, a4, dp2))
            # differently sized transformed parameters
            @test_throws DimensionMismatch efficiency(a7, a6, dp1, dp2)
        end
    end
end
end
