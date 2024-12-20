# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignproblemTests
using Test
using Kirstine
using Random: seed!
using BenchmarkTools: @ballocated

include("example-emax.jl")
include("example-compartment.jl")

@testset "designproblem.jl" begin
    @testset "DesignProblem" begin
        let dp = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            )
            dp_str = repr(dp)
            dp_parsed = eval(Meta.parse(dp_str))
            @test dp == dp_parsed
        end

        let dp = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            dp_a = DesignProblem(dp; criterion = ACriterion()),
            dp_copy = DesignProblem(dp)

            # replacement works
            @test criterion(dp_a) == ACriterion()
            # other fields are equal (check by substituting back) ...
            @test dp == DesignProblem(dp_a; criterion = DCriterion())
            # ... but not identical when they don't refer to singleton objects
            @test prior_knowledge(dp_a) !== prior_knowledge(dp)

            @test dp_copy == dp
            @test dp_copy !== dp
        end
    end

    @testset "==" begin
        let dp1 = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            dp2 = deepcopy(dp1)

            @test dp1 == dp2
        end
    end

    @testset "hash" begin
        let dr = DesignInterval(:dose => (0, 10)),
            m = EmaxModel(1),
            cp = CopyDose(),
            pk = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            dp1 = DesignProblem(;
                criterion = DCriterion(),
                region = dr,
                model = m,
                covariate_parameterization = cp,
                prior_knowledge = pk,
            ),
            dp2 = DesignProblem(;
                criterion = ACriterion(),
                region = dr,
                model = m,
                covariate_parameterization = cp,
                prior_knowledge = pk,
            ),
            dp3 = deepcopy(dp1)

            # implicitly constructs a Set, which calls hash()
            @test unique([dp1, dp2, dp3]) == [dp1, dp2]
        end
    end

    @testset "objective_constants" begin
        let dp = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            )
            oc = Kirstine.objective_constants(dp)

            @test oc isa Kirstine.DPObjectiveConstants
        end
    end

    @testset "derivative_constants" begin
        let dp = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            at = equidistant_design(region(dp), 4)

            dc = Kirstine.derivative_constants(at, dp)

            @test dc isa Kirstine.DPDerivativeConstants
        end
    end

    @testset "derivative_constants!" begin
        let dp = DesignProblem(;
                criterion = ACriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            at = equidistant_design(region(dp), 4),
            somewhere_else = DesignMeasure([[1], [2], [3], [4]], [0.1, 0.2, 0.3, 0.4]),
            w = Kirstine.allocate_workspace(at, dp),
            oc = Kirstine.objective_constants(dp),
            dc = Kirstine.derivative_constants(at, dp),
            dc_copy = deepcopy(dc)

            Kirstine.derivative_constants!(dc, w, somewhere_else, dp, oc)

            @test dc.gc.A[1] != dc_copy.gc.A[1]
            @test dc.gc.tr_B[1] != dc_copy.gc.tr_B[1]
        end
    end

    @testset "objective!" begin
        # The correct implementation of the criteria is checked in criterion-{a,d}.jl.
        # Here we test whether there are spurious memory allocations
        # that could point to type instability.
        let dp = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 1)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            d = equidistant_design(region(dp), 4),
            ws = Kirstine.allocate_workspace(d, dp),
            oc = Kirstine.objective_constants(dp)

            ba = @ballocated Kirstine.objective!($ws, $d, $dp, $oc) evals = 1 samples = 1

            @test ba == 0
        end
    end

    @testset "gateauxderivative!" begin
        # The correct implementation of the criteria is checked in criterion-{a,d}.jl.
        # Here we test whether there are spurious memory allocations
        # that could point to type instability.
        let dp = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 1)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            at = equidistant_design(region(dp), 4),
            dir = one_point_design([0.5]),
            ws = Kirstine.allocate_workspace(dir, dp),
            dcon = Kirstine.derivative_constants(at, dp)

            ba = @ballocated(
                Kirstine.gateauxderivative!($ws, $dir, $dp, $dcon);
                evals = 1,
                samples = 1
            )

            @test ba == 0
        end
    end

    @testset "simplify" begin
        let dp = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            d = DesignMeasure([0 2.495 2.505 10], [0.492, 0.008, 0.008, 0.492]),
            s = sort_points(simplify(d, dp; maxweight = 1e-2, maxdist = 1e-2))

            @test objective(s, dp) > -Inf
            @test points(s) == [[0], [2.5], [10]]
            @test weights(s) == [0.492, 0.016, 0.492]
        end
    end

    @testset "efficiency" begin
        # Atkinson et al. example
        let _ = seed!(4711),
            # prior guess for locally optimal design
            g0 = PriorSample([TPCParameter(; a = 4.298, e = 0.05884, s = 21.80)]),
            # a draw from the strongly informative prior
            g1 = draw_from_prior(1000, 2),
            t_id = Identity(),
            t_auc = DeltaMethod(Dauc),
            dp_for(pk, trafo) = DesignProblem(;
                criterion = DCriterion(),
                # not used in efficiency calculation!
                region = DesignInterval(:time => [0, 48]),
                model = TPCModel(; sigma = 1),
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
            dp7 = dp_for(g1, t_auc),
            # swap out criterion
            dp6a = DesignProblem(;
                region = region(dp6),
                model = model(dp6),
                covariate_parameterization = covariate_parameterization(dp6),
                criterion = ACriterion(),
                prior_knowledge = prior_knowledge(dp6),
            )

            # Compare with published efficiencies in Table 5. Due to Monte-Carlo uncertainty
            # and rounding of the published values, this is not very exact.
            @test efficiency(a7, a6, dp6) ≈ 0.234 atol = 1e-2
            @test efficiency(a6, a7, dp7) ≈ 0.370 atol = 1e-2
            # check that criterion is ignored
            @test efficiency(a7, a6, dp6a) ≈ 0.234 atol = 1e-2
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

    @testset "shannon_information" begin
        # Atkinson et al. locally optimal example
        let dp = DesignProblem(;
                region = DesignInterval(:time => [0, 48]),
                model = TPCModel(; sigma = 1),
                covariate_parameterization = CopyTime(),
                criterion = DCriterion(),
                normal_approximation = FisherMatrix(),
                prior_knowledge = PriorSample([
                    TPCParameter(; a = 4.298, e = 0.05884, s = 21.80),
                ]),
                transformation = Identity(),
            ),
            dpa = DesignProblem(;
                region = region(dp),
                model = model(dp),
                covariate_parameterization = covariate_parameterization(dp),
                criterion = ACriterion(),
                prior_knowledge = prior_knowledge(dp),
            ),
            d1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3),
            c = 3 / 2 * (log(2 * pi) - 1),
            half_ob = 0.5 * 7.3887,
            si = shannon_information

            # published objective value is rounded to four places
            @test si(d1, dp, 1) ≈ c + half_ob atol = 5e-5
            @test si(d1, dp, 42) ≈ 3 / 2 * log(42) + c + half_ob atol = 5e-5
            @test si(d1, dp, 250) ≈ 3 / 2 * log(250) + c + half_ob atol = 5e-5
            # the criterion of the problem should be ignored
            @test si(d1, dpa, 1) ≈ c + half_ob atol = 5e-5
        end
    end
end
end
