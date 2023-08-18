# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignDOptimalTests
using Test
using Kirstine
using Random: seed!
using LinearAlgebra: Symmetric, diagm, tr

include("example-compartment.jl")

@testset "criterion-d.jl" begin
    @testset "criterion_integrand!" begin
        # The functional should be log(det(m)) or -(log(det(inv_m))), depending on whether m
        # is passed as already inverted. For singular matrices it should always return -Inf.
        let mreg = [1.0 0.5; 0.5 2.0],
            msng = [1.0 0.5; 0.5 0.25],
            dc = DOptimality(),
            ci! = Kirstine.criterion_integrand!

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
                design_region = DesignInterval(:time => [0, 48]),
                model = TPCMod(1),
                covariate_parameterization = CopyTime(),
                design_criterion = DOptimality(),
                normal_approximation = FisherMatrix(),
                prior_knowledge = PriorSample([
                    TPCPar(; a = 4.298, e = 0.05884, s = 21.80),
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
                design_region = DesignInterval(:time => [0, 48]),
                model = TPCMod(1),
                covariate_parameterization = CopyTime(),
                design_criterion = DOptimality(),
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

    @testset "precalculate_gateaux_constants" begin
        # Identity transformation
        #
        # The GateauxConstants essentially wrap the inverse of the normalized information
        # matrix at the candidate solution. Only the upper triangle is stored. There is one
        # matrix for each prior parameter value.
        #
        # We use an example model from Atkinson et al. with a two-point prior.
        let dc = DOptimality(),
            a1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3),
            a4 = DesignMeasure([1.0122] => 1.0), # singular
            m = TPCMod(1),
            cp = CopyTime(),
            g1 = TPCPar(; a = 4.298, e = 0.05884, s = 21.80),
            g2 = TPCPar(; a = 4.298 + 0.5, e = 0.05884 + 0.005, s = 21.80), # g1 + 1 * se
            pk = PriorSample([g1, g2]),
            tc = Kirstine.TCIdentity(3), # the codomain dimension is not used in this test
            na = FisherMatrix(),
            pgc = Kirstine.precalculate_gateaux_constants,
            gc = pgc(dc, a1, m, cp, pk, tc, na)

            # Singular designs should raise an exception. It will be caught by the caller.
            @test_throws "SingularException" pgc(dc, a4, m, cp, pk, tc, na)
            @test isa(gc, Kirstine.GCDIdentity)
            @test length(gc.invM) == 2
            # Note: the informationmatrix return value is already wrapped in Symmetric()
            @test Symmetric(gc.invM[1]) ≈ inv(informationmatrix(a1, m, cp, g1, na))
            @test Symmetric(gc.invM[2]) ≈ inv(informationmatrix(a1, m, cp, g2, na))
            @test gc.parameter_length == 3
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
        let dc = DOptimality(),
            a1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3),
            a4 = DesignMeasure([1.0122] => 1.0), # singular
            m = TPCMod(1),
            cp = CopyTime(),
            g1 = TPCPar(; a = 4.298, e = 0.05884, s = 21.80),
            g2 = TPCPar(; a = 4.298 + 0.5, e = 0.05884 + 0.005, s = 21.80), # g1 + 1 * se
            pk = PriorSample([g1, g2]),
            J = [Dauc(g1), Dauc(g2)],
            tc = Kirstine.TCDeltaMethod(1, J),
            na = FisherMatrix(),
            pgc = Kirstine.precalculate_gateaux_constants,
            gc = pgc(dc, a1, m, cp, pk, tc, na),
            # Note: the informationmatrix return value is already wrapped in Symmetric()
            M1 = informationmatrix(a1, m, cp, g1, na),
            M2 = informationmatrix(a1, m, cp, g2, na),
            B1 = J[1]' * inv(J[1] * inv(M1) * J[1]') * J[1],
            B2 = J[2]' * inv(J[2] * inv(M2) * J[2]') * J[2]

            # Singular designs should raise an exception. It will be caught by the caller.
            @test_throws "SingularException" pgc(dc, a4, m, cp, pk, tc, na)
            @test isa(gc, Kirstine.GCDDeltaMethod)
            @test length(gc.invM_B_invM) == 2
            @test Symmetric(gc.invM_B_invM[1]) ≈ inv(M1) * B1 * inv(M1)
            @test Symmetric(gc.invM_B_invM[2]) ≈ inv(M2) * B2 * inv(M2)
            @test gc.transformed_parameter_length == 1
        end
    end

    @testset "gateaux_integrand" begin
        # Identity transformation
        #
        # This should compute tr(A * B) - r, using only upper triangles from A and B. A is
        # taken from the GateauxConstants, B is the normalized information matrix
        # corresponding to the direction. Both matrices have size (r, r). Here we test an
        # example with r = 2.
        let A = [2.0 3.0; 0.0 7.0],
            c = Kirstine.GCDIdentity([A], 2),
            B = [0.4 1.0; 0.0 0.5],
            gi = Kirstine.gateaux_integrand

            @test gi(c, B, 1) == tr(Symmetric(A) * Symmetric(B)) - 2
            # rule out unintentionally symmetric input
            @test gi(c, B, 1) != tr(A * B) - 2
        end

        # DeltaMethod transformation
        #
        # This should compute tr(A * B) - t, using only upper triangles from A and B. A is
        # taken from the GateauxConstants, B is the normalized information matrix
        # corresponding to the direction. Both matrices have size (r, r). t is the length of
        # the transformed parameter. Here we test an example with r = 2 and t = 1.
        let A = [2.0 3.0; 0.0 7.0],
            c = Kirstine.GCDDeltaMethod([A], 1),
            B = [0.4 1.0; 0.0 0.5],
            gi = Kirstine.gateaux_integrand

            @test gi(c, B, 1) == tr(Symmetric(A) * Symmetric(B)) - 1
            # rule out unintentionally symmetric input
            @test gi(c, B, 1) != tr(A * B) - 2
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
                design_region = DesignInterval(:time => [0, 48]),
                model = TPCMod(1),
                covariate_parameterization = CopyTime(),
                design_criterion = DOptimality(),
                normal_approximation = FisherMatrix(),
                prior_knowledge = PriorSample([
                    TPCPar(; a = 4.298, e = 0.05884, s = 21.80),
                ]),
            ),
            dpd = DesignProblem(;
                transformation = DeltaMethod(p -> diagm([1, 1, 1])),
                design_region = DesignInterval(:time => [0, 48]),
                model = TPCMod(1),
                covariate_parameterization = CopyTime(),
                design_criterion = DOptimality(),
                normal_approximation = FisherMatrix(),
                prior_knowledge = PriorSample([
                    TPCPar(; a = 4.298, e = 0.05884, s = 21.80),
                ]),
            ),
            dir = [one_point_design([t]) for t in range(0, 48; length = 21)],
            # singular, with inversion of the information matrix
            d0 = one_point_design([1]),
            # solution
            d1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3),
            d1dir = one_point_design.(designpoints(d1))

            @test_throws "one-point design" gateauxderivative(d1, [d1], dpi)
            @test all(isnan.(gateauxderivative(d0, dir, dpi)))
            @test all(isnan.(gateauxderivative(d0, dir, dpd)))
            @test maximum(gateauxderivative(d1, dir, dpi)) <= 0
            @test all(abs.(gateauxderivative(d1, d1dir, dpi)) .< 1e-4)
            @test gateauxderivative(d1, dir, dpi) ≈ gateauxderivative(d1, dir, dpd)
        end
    end
end
end
