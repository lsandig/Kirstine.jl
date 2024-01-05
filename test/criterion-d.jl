# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignDOptimalTests
using Test
using Kirstine
using Random: seed!
using LinearAlgebra: Symmetric, diagm, tr

include("example-compartment.jl")

@testset "criterion-d.jl" begin
    @testset "criterion_functional!" begin
        # The functional should be log(det(m)) or -(log(det(inv_m))), depending on whether m
        # is passed as already inverted. For singular matrices it should always return -Inf.
        let mreg = [1.0 0.5; 0.5 2.0],
            msng = [1.0 0.5; 0.5 0.25],
            dc = DCriterion(),
            ci! = Kirstine.criterion_functional!

            # interpret m as not inverted
            @test ci!(deepcopy(mreg), false, dc) ≈ log(1.75)
            @test ci!(deepcopy(msng), false, dc) == -Inf
            # interpret m as inverted
            @test ci!(deepcopy(mreg), true, dc) ≈ -log(1.75)
            @test ci!(deepcopy(msng), true, dc) == -Inf
        end
    end

    @testset "objective" begin
        # Note: this testset effectively tests functionality in `objective!`, since
        # `objective` simply allocates a few objects to work on. Not having to do this
        # manually to test the mutating version is a bit more convenient.

        # Identity Transformation: Atkinson et al. locally optimal example
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
            # singular, no inversion of the information matrix
            d0 = one_point_design([1]),
            # solution
            d1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3)

            @test objective(d0, dp) == -Inf
            # published objective value is rounded to four places
            @test objective(d1, dp) ≈ 7.3887 atol = 1e-5
        end

        # DeltaMethod Transformation: Atkinson et al. strong prior AUC estimation example
        let _ = seed!(4711),
            dp = DesignProblem(;
                region = DesignInterval(:time => [0, 48]),
                model = TPCModel(; sigma = 1),
                covariate_parameterization = CopyTime(),
                criterion = DCriterion(),
                normal_approximation = FisherMatrix(),
                prior_knowledge = draw_from_prior(1000, 2),
                transformation = DeltaMethod(Dauc),
            ),
            # singular, with inversion of the information matrix
            d0 = one_point_design([1]),
            # solution
            d1 = DesignMeasure([0.2449] => 0.0129, [1.4950] => 0.0387, [18.4903] => 0.9484)

            @test objective(d0, dp) == -Inf
            # The article uses numerical quadrature, we are using MC integration. Hence the
            # objective value cannot be expected to be terribly accurate.
            @test exp(-objective(d1, dp)) ≈ 2463.3 rtol = 1e-2
        end
    end

    @testset "gateaux_constants" begin
        # Identity transformation
        #
        # The GateauxConstants essentially wrap the inverse of the normalized information
        # matrix at the candidate solution. Only the upper triangle is stored. There is one
        # matrix for each prior parameter value.
        #
        # We use an example model from Atkinson et al. with a two-point prior.
        let dc = DCriterion(),
            a1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3),
            a4 = DesignMeasure([1.0122] => 1.0), # singular
            m = TPCModel(; sigma = 1),
            cp = CopyTime(),
            g1 = TPCParameter(; a = 4.298, e = 0.05884, s = 21.80),
            g2 = TPCParameter(; a = 4.298 + 0.5, e = 0.05884 + 0.005, s = 21.80), # g1 + 1 * se
            pk = PriorSample([g1, g2]),
            trafo = Identity(), # the codomain dimension is not used in this test
            na = FisherMatrix(),
            pgc = Kirstine.gateaux_constants,
            gc = pgc(dc, a1, m, cp, pk, trafo, na)

            # Singular designs should raise an exception. It will be caught by the caller.
            @test_throws "SingularException" pgc(dc, a4, m, cp, pk, trafo, na)
            @test isa(gc, Kirstine.GCPriorSample)
            @test length(gc.A) == 2
            @test length(gc.tr_B) == 2
            # Note: the informationmatrix return value is already wrapped in Symmetric()
            @test Symmetric(gc.A[1]) ≈ inv(informationmatrix(a1, m, cp, g1, na))
            @test Symmetric(gc.A[2]) ≈ inv(informationmatrix(a1, m, cp, g2, na))
            @test all(gc.tr_B .== 3)
        end

        # DeltaMethod transformation
        #
        # The GateauxConstants essentially wrap the three-factor matrix product
        #   inv(M) * B * inv(M),
        # where M is the informationmatrix at the candidate solution, and
        #   B = J' * inv(J * inv(M) * J') * J
        # with J the Jacobian matrix of the transformation.
        # There is one product matrix for each prior parameter value.
        #
        # We use an example model from Atkinson et al. with a two-point prior
        # and transformation to a 1-dimensional quantity.
        let dc = DCriterion(),
            a1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3),
            a4 = DesignMeasure([1.0122] => 1.0), # singular
            m = TPCModel(; sigma = 1),
            cp = CopyTime(),
            g1 = TPCParameter(; a = 4.298, e = 0.05884, s = 21.80),
            g2 = TPCParameter(; a = 4.298 + 0.5, e = 0.05884 + 0.005, s = 21.80), # g1 + 1 * se
            pk = PriorSample([g1, g2]),
            J = [Dauc(g1), Dauc(g2)],
            trafo = DeltaMethod(Dauc),
            na = FisherMatrix(),
            pgc = Kirstine.gateaux_constants,
            gc = pgc(dc, a1, m, cp, pk, trafo, na),
            # Note: the informationmatrix return value is already wrapped in Symmetric()
            M1 = informationmatrix(a1, m, cp, g1, na),
            M2 = informationmatrix(a1, m, cp, g2, na),
            B1 = J[1]' * inv(J[1] * inv(M1) * J[1]') * J[1],
            B2 = J[2]' * inv(J[2] * inv(M2) * J[2]') * J[2]

            # Singular designs should raise an exception. It will be caught by the caller.
            @test_throws "SingularException" pgc(dc, a4, m, cp, pk, trafo, na)
            @test isa(gc, Kirstine.GCPriorSample)
            @test length(gc.A) == 2
            @test length(gc.tr_B) == 2
            @test Symmetric(gc.A[1]) ≈ inv(M1) * B1 * inv(M1)
            @test Symmetric(gc.A[2]) ≈ inv(M2) * B2 * inv(M2)
            @test all(gc.tr_B .== 1)
        end
    end

    @testset "gateauxderivative" begin
        # Identity Transformation: Atkinson et al. locally optimal example
        #
        # DeltaMethod Transformation: A direct test with Bayesian solution is not possible
        # because of MC-integration variability. Indirectly check that an Identity
        # transformation and an equivalent DeltaMethod transformation give the same results.
        let dpi = DesignProblem(;
                transformation = Identity(),
                region = DesignInterval(:time => [0, 48]),
                model = TPCModel(; sigma = 1),
                covariate_parameterization = CopyTime(),
                criterion = DCriterion(),
                normal_approximation = FisherMatrix(),
                prior_knowledge = PriorSample([
                    TPCParameter(; a = 4.298, e = 0.05884, s = 21.80),
                ]),
            ),
            dpd = DesignProblem(;
                transformation = DeltaMethod(p -> diagm([1, 1, 1])),
                region = DesignInterval(:time => [0, 48]),
                model = TPCModel(; sigma = 1),
                covariate_parameterization = CopyTime(),
                criterion = DCriterion(),
                normal_approximation = FisherMatrix(),
                prior_knowledge = PriorSample([
                    TPCParameter(; a = 4.298, e = 0.05884, s = 21.80),
                ]),
            ),
            dir = [one_point_design([t]) for t in range(0, 48; length = 21)],
            # singular, with inversion of the information matrix
            d0 = one_point_design([1]),
            # solution
            d1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3),
            d1dir = one_point_design.(points(d1))

            @test_throws "one-point directions" gateauxderivative(d1, [d1], dpi)
            @test all(isnan.(gateauxderivative(d0, dir, dpi)))
            @test all(isnan.(gateauxderivative(d0, dir, dpd)))
            @test maximum(gateauxderivative(d1, dir, dpi)) <= 0
            @test all(abs.(gateauxderivative(d1, d1dir, dpi)) .< 1e-4)
            @test gateauxderivative(d1, dir, dpi) ≈ gateauxderivative(d1, dir, dpd)
        end
    end
end
end
