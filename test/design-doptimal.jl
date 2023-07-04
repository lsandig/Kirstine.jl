module DesignDOptimalTests
using Test
using Kirstine
using Random: seed!
using LinearAlgebra: Symmetric, diagm, tr

include("example-emax.jl")
include("example-compartment.jl")

@testset "design-doptimal.jl" begin
    @testset "efficiency" begin
        let dc = DOptimality(),
            na_ml = FisherMatrix(),
            trafo = Identity(),
            m = EmaxModel(1),
            cp = CopyDose(),
            p1 = (e0 = 1, emax = 10, ec50 = 5),
            p2 = (e0 = 5, emax = -3, ec50 = 2),
            pk1 = DiscretePrior(p1),
            pk2 = DiscretePrior([0.75, 0.25], [p1, p2]),
            pk3 = DiscretePrior([p1, p2]),
            ds = DesignSpace(:dose => (0, 10)),
            # sol is optimal for pk1
            sol = emax_solution(p1, ds),
            a = ds.lowerbound[1],
            b = ds.upperbound[1],
            x_star = sol.designpoint[2][1],
            # not_sol is not optimal for any of pk1, pk2, pk3
            not_sol = DesignMeasure(
                [0.2, 0.3, 0.5],
                [[a + 0.1 * (b - a)], [x_star * 1.1], [a + (0.9 * (b - a))]],
            ),
            # a design with fewer than three points is singular
            sng1 = uniform_design([[a]]),
            sng2 = uniform_design([[a], [b]])

            # sol is better than not_sol for pk1 (by construction)
            @test efficiency(sol, not_sol, m, cp, pk1, trafo, na_ml) > 1
            # it also happens to be better for pk2 and pk3
            @test efficiency(sol, not_sol, m, cp, pk2, trafo, na_ml) > 1
            @test efficiency(sol, not_sol, m, cp, pk3, trafo, na_ml) > 1
            # check that `efficiency` correctly deals with singular designs
            @test efficiency(sng2, sol, m, cp, pk1, trafo, na_ml) == 0
            @test efficiency(sol, sng2, m, cp, pk1, trafo, na_ml) == Inf
            @test isnan(efficiency(sng1, sng2, m, cp, pk1, trafo, na_ml))
            # Note: although a mathematical argument could be made that the efficiency should be
            # equal to 1 in the following case, we still want it to be NaN:
            @test isnan(efficiency(sng1, sng1, m, cp, pk1, trafo, na_ml))
        end

        # DeltaMethod for Atkinson et al. examples
        let ds = DesignSpace(:time => [0, 48]),
            _ = seed!(4711),
            g1 = draw_from_prior(1000, 2),
            m = TPCMod(1),
            cp = CopyTime(),
            dc = DOptimality(),
            na_ml = FisherMatrix(),
            # Some designs from Tables 1 and 2, and corresponding transformations
            #! format: off
            a6  = DesignMeasure([0.2288] => 1/3,    [1.4170] => 1/3,    [18.4513] => 1/3),
            a7  = DesignMeasure([0.2449] => 0.0129, [1.4950] => 0.0387, [18.4903] => 0.9484),
            a8  = DesignMeasure([0.1829] => 0.6023, [2.4639] => 0.2979, [8.8542]  => 0.0998),
            a9  = DesignMeasure([0.3608] => 0.0730, [1.1446] => 0.9094, [20.9218] => 0.0176),
            a10 = DesignMeasure([0.2235] => 0.2366, [1.4875] => 0.3838, [18.8293] => 0.3796),
            a0_time = [1/6, 1/3, 1/2, 2/3, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 24, 30, 48],
            a0 = uniform_design([[t] for t in a0_time]),
            #! format: on
            t1 = Identity(),
            t2 = DeltaMethod(Dauc),
            t3 = DeltaMethod(Dttm),
            t4 = DeltaMethod(Dcmax),
            #! format: off
            ebay = [100    37    67.2  39.3;
                    23.4 100     3.2   4.5;
                    57.4   5.1 100    19.6;
                    28.2   1.9  12.4 100  ;
                    97.5  42.1  53.9  45.2;
                    68.4  26    30.2  41  ],
            #! format: on
            ef1(a, zs, ts) =
                map((z, t) -> 100 * efficiency(a, z, m, cp, g1, t, na_ml), zs, ts)

            # Table 5 from the article, precise to 1 percentage point
            @test ebay[1, :] ≈ ef1(a6, [a6, a7, a8, a9], [t1, t2, t3, t4]) atol = 1
            @test ebay[2, :] ≈ ef1(a7, [a6, a7, a8, a9], [t1, t2, t3, t4]) atol = 1
            @test ebay[3, :] ≈ ef1(a8, [a6, a7, a8, a9], [t1, t2, t3, t4]) atol = 1
            @test ebay[4, :] ≈ ef1(a9, [a6, a7, a8, a9], [t1, t2, t3, t4]) atol = 1
            @test ebay[5, :] ≈ ef1(a10, [a6, a7, a8, a9], [t1, t2, t3, t4]) atol = 1
            @test ebay[6, :] ≈ ef1(a0, [a6, a7, a8, a9], [t1, t2, t3, t4]) atol = 1
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
