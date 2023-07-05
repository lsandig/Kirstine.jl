module DesignDOptimalTests
using Test
using Kirstine
using Random: seed!
using LinearAlgebra: Symmetric, diagm, tr

include("example-compartment.jl")

@testset "design-doptimal.jl" begin
    @testset "efficiency" begin
        # Atkinson et al. example
        let ds = DesignInterval(:time => [0, 48]),
            _ = seed!(4711),
            # prior guess for locally optimal design
            g0 = DiscretePrior((a = 4.298, e = 0.05884, s = 21.80)),
            # a draw from the strongly informative prior
            g1 = draw_from_prior(1000, 2),
            m = TPCMod(1),
            cp = CopyTime(),
            dc = DOptimality(),
            na = FisherMatrix(),
            t_id = Identity(),
            t_auc = DeltaMethod(Dauc),
            # singular locally optimal designs from Table 4
            a3 = DesignMeasure([0.1793] => 0.6062, [3.5671] => 0.3938),
            a4 = DesignMeasure([1.0122] => 1.0),
            # Bayesian optimal designs from Table 5
            a6 = DesignMeasure([0.2288] => 1 / 3, [1.4170] => 1 / 3, [18.4513] => 1 / 3),
            a7 = DesignMeasure([0.2449] => 0.0129, [1.4950] => 0.0387, [18.4903] => 0.9484)

            # Compare with published efficiencies in Table 5. Due to Monte-Carlo uncertainty
            # and rounding of the published values, this is not very exact.
            @test efficiency(a7, a6, m, cp, g1, t_id, na) ≈ 0.234 atol = 1e-2
            @test efficiency(a6, a7, m, cp, g1, t_auc, na) ≈ 0.370 atol = 1e-2

            # Check that singular designs are handled correctly both with Identity and with
            # DeltaMethod transformation.
            @test efficiency(a4, a6, m, cp, g0, t_id, na) == 0
            @test efficiency(a4, a6, m, cp, g0, t_auc, na) == 0
            @test efficiency(a6, a4, m, cp, g0, t_id, na) == Inf
            @test efficiency(a6, a4, m, cp, g0, t_auc, na) == Inf
            @test isnan(efficiency(a3, a4, m, cp, g0, t_id, na))
            @test isnan(efficiency(a3, a4, m, cp, g0, t_auc, na))
            # Note: although a mathematical argument could be made that the efficiency should be
            # equal to 1 in the following case, we still want it to be NaN:
            @test isnan(efficiency(a4, a4, m, cp, g0, t_id, na))
            @test isnan(efficiency(a4, a4, m, cp, g0, t_auc, na))
        end
    end

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
            g1 = (a = 4.298, e = 0.05884, s = 21.80),
            g2 = (a = 4.298 + 0.5, e = 0.05884 + 0.005, s = 21.80), # g1 + 1 * se
            pk = DiscretePrior([g1, g2]),
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
            g1 = (a = 4.298, e = 0.05884, s = 21.80),
            g2 = (a = 4.298 + 0.5, e = 0.05884 + 0.005, s = 21.80), # g1 + 1 * se
            pk = DiscretePrior([g1, g2]),
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

    @testset "gateaux_integrand!" begin
        # Identity transformation
        #
        # This should compute tr(A * B) - r, using only upper Triangles from A and B. A is
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
        # This should compute tr(A * B) - t, using only upper Triangles from A and B. A is
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
end
end
