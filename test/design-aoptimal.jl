# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignDOptimalTests
using Test
using Kirstine
# using Random: seed!
using LinearAlgebra: SingularException, Symmetric, tr #  diagm,

include("example-compartment.jl")

@testset "design-aoptimal.jl" begin
    @testset "criterion_integrand!" begin
        # The functional should be -tr(m) or -tr(inv_m), depending on whether m
        # is passed as already inverted. For singular matrices it should return 0 or -Inf.
        let mreg = [1.0 0.5; 0.5 2.0],
            msng = [1.0 0.5; 0.5 0.25],
            dc = AOptimality(),
            ci! = Kirstine.criterion_integrand!

            # interpret m as not inverted
            @test ci!(deepcopy(mreg), false, dc) ≈ -12 / 7
            @test_throws SingularException ci!(deepcopy(msng), false, dc)
            # interpret m as inverted
            @test ci!(deepcopy(mreg), true, dc) ≈ -3
            @test ci!(deepcopy(msng), true, dc) <= 0
        end
    end

    @testset "precalculate_gateaux_constants" begin
        # Identity transformation
        #
        # The GateauxConstants wrap the square and the trace of the inverse of the
        # normalized information matrix at the candidate solution. Only the upper triangle
        # is stored. There is one matrix and trace for each prior parameter value.
        #
        # We use an example model from Atkinson et al. with a two-point prior.
        let dc = AOptimality(),
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
            # reference information matrices, already wrapped in Symmetric()
            m1 = informationmatrix(a1, m, cp, g1, na),
            m2 = informationmatrix(a1, m, cp, g2, na),
            gc = pgc(dc, a1, m, cp, pk, tc, na)

            # Singular designs should raise an exception. It will be caught by the caller.
            @test_throws "SingularException" pgc(dc, a4, m, cp, pk, tc, na)
            @test isa(gc, Kirstine.GCAIdentity)
            @test length(gc.B) == 2
            @test length(gc.tr_C) == 2
            @test Symmetric(gc.B[1]) ≈ inv(m1)^2
            @test Symmetric(gc.B[2]) ≈ inv(m2)^2
            @test gc.tr_C[1] ≈ tr(inv(m1))
            @test gc.tr_C[2] ≈ tr(inv(m2))
        end

        # DeltaMethod transformation
        #
        # The GateauxConstants wrap M^{-1} J' J M^{-1} and the trace of M_T^{-1} at the
        # candidate solution. There is one matrix and trace for each prior parameter value.
        #
        # We use an example model from Atkinson et al. with a two-point prior
        # and transformation to a 1-dimensional quantity
        let dc = AOptimality(),
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
            # reference information matrices, already wrapped in Symmetric()
            m1 = informationmatrix(a1, m, cp, g1, na),
            m2 = informationmatrix(a1, m, cp, g2, na),
            gc = pgc(dc, a1, m, cp, pk, tc, na)

            # Singular designs should raise an exception. It will be caught by the caller.
            @test_throws "SingularException" pgc(dc, a4, m, cp, pk, tc, na)
            @test isa(gc, Kirstine.GCADeltaMethod)
            @test length(gc.B) == 2
            @test length(gc.tr_C) == 2
            @test Symmetric(gc.B[1]) ≈ inv(m1) * J[1]' * J[1] * inv(m1)
            @test Symmetric(gc.B[2]) ≈ inv(m2) * J[2]' * J[2] * inv(m2)
            @test gc.tr_C[1] ≈ tr(J[1] * inv(m1) * J[1]')
            @test gc.tr_C[2] ≈ tr(J[2] * inv(m2) * J[2]')
        end
    end

    @testset "gateaux_integrand" begin
        # Identity transformation
        #
        # This should compute tr(A * B) - d, using only upper triangles from
        # A and B. A is taken from the GateauxConstants, B is the normalized information
        # matrix corresponding to the direction. Both matrices have size (r, r). Here we
        # test an example with r = 2.
        let A = [2.0 3.0; 0.0 7.0],
            c = Kirstine.GCAIdentity([A], [1.8]),
            B = [0.4 1.0; 0.0 0.5],
            gi = Kirstine.gateaux_integrand

            @test gi(c, B, 1) == tr(Symmetric(A) * Symmetric(B)) - 1.8
            # rule out unintentionally symmetric input
            @test gi(c, B, 1) != tr(A * B) - 1.8
        end

        # DeltaMethod transformation
        #
        # This should compute tr(A * B) - d, using only upper triangles from
        # A and B. A is taken from the GateauxConstants, B is the normalized information
        # matrix corresponding to the direction. Both matrices have size (r, r). Here we
        # test an example with r = 2.
        let A = [2.0 3.0; 0.0 7.0],
            c = Kirstine.GCADeltaMethod([A], [1.8]),
            B = [0.4 1.0; 0.0 0.5],
            gi = Kirstine.gateaux_integrand

            @test gi(c, B, 1) == tr(Symmetric(A) * Symmetric(B)) - 1.8
            # rule out unintentionally symmetric input
            @test gi(c, B, 1) != tr(A * B) - 1.8
        end
    end
end
end
