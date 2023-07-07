module DesignCommonTests
using Test
using Kirstine
using Random: seed!
using LinearAlgebra: Symmetric, UpperTriangular, tr, det, diagm

include("example-testpar.jl")
include("example-emax.jl")
include("example-compartment.jl")

@testset "design-common.jl" begin
    # helpers
    @testset "DesignConstraints" begin
        let dc = DOptimality(),
            ds = DesignInterval(:dose => (0, 10)),
            d = equidistant_design(ds, 3),
            dcon = Kirstine.DesignConstraints,
            c = dcon(d, ds, [2], [3])

            # Trigger erros for out-of-range weight / designpoint indices.
            @test_throws "between 1 and 3" dcon(d, ds, [0], Int64[])
            @test_throws "between 1 and 3" dcon(d, ds, [4], Int64[])
            @test_throws "between 1 and 3" dcon(d, ds, Int64[], [-1])
            @test_throws "between 1 and 3" dcon(d, ds, Int64[], [5])
            # If only one weight index is not given as fixed, it is still implicitly fixed
            # because of the simplex constraint. In this case we want to have it fixed
            # explicitly.
            @test_logs(
                (:info, "explicitly fixing implicitly fixed weight"),
                dcon(d, ds, [1, 3], Int64[])
            )
            @test c.fixw == [false, true, false]
            @test c.fixp == [false, false, true]
        end
    end

    @testset "tr_prod" begin
        let A = reshape(collect(1:9), 3, 3),
            B = reshape(collect(11:19), 3, 3),
            C = reshape(collect(1:12), 3, 4)

            @test tr(Symmetric(A) * Symmetric(B)) == Kirstine.tr_prod(A, B, :U)
            @test tr(Symmetric(A, :L) * Symmetric(B, :L)) == Kirstine.tr_prod(A, B, :L)
            @test_throws "identical size" Kirstine.tr_prod(A, C, :U)
            @test_throws "either :U or :L" Kirstine.tr_prod(A, B, :F)
        end
    end

    @testset "precalculate_trafo_constants" begin
        let pk = DiscretePrior([TestPar2(1, 2), TestPar2(-1, -2)]),
            dt1 = p -> [p.a; p.b], # too few columns
            D1 = DeltaMethod(dt1),
            dt2 = p -> p.a > 0 ? [p.a p.b] : [p.a p.b; p.a p.b], # different number of rows
            D2 = DeltaMethod(dt2)

            @test_throws "2 columns" Kirstine.precalculate_trafo_constants(D1, pk)
            @test_throws "identical" Kirstine.precalculate_trafo_constants(D2, pk)
        end
    end

    @testset "log_det!" begin
        let A = rand(Float64, 3, 3), B = A * A'

            # note: log_det! overwrites B, so it can't be called first
            @test log(det(B)) ≈ Kirstine.log_det!(B)
        end
    end

    @testset "apply_transformation" begin
        # Applying a DeltaMethod transformation should give the same results whether the
        # incoming normalized information matrix is already inverted or not. For a DeltaMethod
        # that is actually an identity transformation in disguise, the result should simply be
        # the inverse of the argument.
        #
        # Note: we have to recreate the circumstances in which apply_transformation! is called:
        # nim is allowed to be only upper triangular, and is allowed to be overwritten. Hence we
        # must use deepcopys, and Symmetric wrappers where necessary.
        let pk = DiscretePrior([TestPar3(1, 2, 3)]),
            tid = DeltaMethod(p -> diagm(ones(3))),
            ctid = Kirstine.precalculate_trafo_constants(tid, pk),
            tsc = DeltaMethod(p -> diagm([0.5, 2.0, 4.0])),
            ctsc = Kirstine.precalculate_trafo_constants(tsc, pk),
            _ = seed!(4321),
            A = reshape(rand(9), 3, 3),
            nim = collect(UpperTriangular(A' * A)),
            inv_nim = collect(UpperTriangular(inv(A' * A))),
            m = 1,
            r = 3,
            t = 3,
            wm1 = Kirstine.WorkMatrices(m, r, t),
            wm2 = Kirstine.WorkMatrices(m, r, t),
            wm3 = Kirstine.WorkMatrices(m, r, t),
            wm4 = Kirstine.WorkMatrices(m, r, t),
            # workaround for `.=` not being valid let statement syntax
            _ = broadcast!(identity, wm1.r_x_r, nim),
            _ = broadcast!(identity, wm2.r_x_r, inv_nim),
            _ = broadcast!(identity, wm3.r_x_r, nim),
            _ = broadcast!(identity, wm4.r_x_r, inv_nim),
            (tnim1, _) = Kirstine.apply_transformation!(wm1, false, ctid, 1),
            (tnim2, _) = Kirstine.apply_transformation!(wm2, true, ctid, 1),
            # scaling parameters should be able to be pulled out
            (tnim3, _) = Kirstine.apply_transformation!(wm3, false, ctsc, 1),
            (tnim4, _) = Kirstine.apply_transformation!(wm4, true, ctsc, 1)

            @test Symmetric(tnim1) ≈ Symmetric(inv_nim)
            @test Symmetric(tnim2) ≈ Symmetric(inv_nim)
            @test Symmetric(tnim3) ≈ Symmetric(tnim4)
            @test det(Symmetric(tnim3)) ≈ 4^2 * det(Symmetric(inv_nim))
        end
    end

    @testset "objective" begin
        # correct handling of singular designs
        let dc = DOptimality(),
            na = FisherMatrix(),
            t1 = Identity(),
            t2 = DeltaMethod(p -> diagm([1, 1, 1])),
            m = EmaxModel(1),
            cp = CopyDose(),
            pk = DiscretePrior([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ds = DesignInterval(:dose => (0, 10)),
            d = one_point_design([5])

            # no explicit inversion
            @test objective(dc, d, m, cp, pk, t1, na) == -Inf
            # with explicit inversion
            @test objective(dc, d, m, cp, pk, t2, na) == -Inf
        end

        let dc = DOptimality(),
            na_ml = FisherMatrix(),
            trafo = Identity(),
            m = EmaxModel(1),
            cp = CopyDose(),
            p1 = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            pk1 = DiscretePrior([p1]),
            ds = DesignInterval(:dose => (0, 10)),
            # sol is optimal for pk1
            sol = emax_solution(p1, ds),
            a = ds.lowerbound[1],
            b = ds.upperbound[1],
            x_star = sol.designpoint[2][1],
            # not_sol is not optimal for pk1
            not_sol = DesignMeasure(
                [0.2, 0.3, 0.5],
                [[a + 0.1 * (b - a)], [x_star * 1.1], [a + (0.9 * (b - a))]],
            ),
            sng2 = uniform_design([[a], [b]]),
            ob(d, pk, na) = objective(dc, d, m, cp, pk, trafo, na)

            @test ob(not_sol, pk1, na_ml) < ob(sol, pk1, na_ml)
            # a design with less than 3 support points is singular
            @test isinf(ob(sng2, pk1, na_ml))
        end

        # DeltaMethod for Atkinson et al. examples
        let ds = DesignInterval(:time => [0, 48]),
            g0 = DiscretePrior([TPCPar(; a = 4.298, e = 0.05884, s = 21.80)]),
            _ = seed!(4711),
            g1 = draw_from_prior(1000, 2),
            m = TPCMod(1),
            cp = CopyTime(),
            dc = DOptimality(),
            na_ml = FisherMatrix(),
            # Some designs from Tables 1 and 2, and corresponding transformations
            #! format: off
            a1  = DesignMeasure([0.2288] => 1/3,    [1.3886] => 1/3,    [18.417] => 1/3),
            a6  = DesignMeasure([0.2288] => 1/3,    [1.4170] => 1/3,    [18.4513] => 1/3),
            a7  = DesignMeasure([0.2449] => 0.0129, [1.4950] => 0.0387, [18.4903] => 0.9484),
            a8  = DesignMeasure([0.1829] => 0.6023, [2.4639] => 0.2979, [8.8542]  => 0.0998),
            a9  = DesignMeasure([0.3608] => 0.0730, [1.1446] => 0.9094, [20.9218] => 0.0176),
            #! format: on
            t1 = Identity(),
            t1_delta = DeltaMethod(p -> diagm(ones(3))),
            t2 = DeltaMethod(Dauc),
            t3 = DeltaMethod(Dttm),
            t4 = DeltaMethod(Dcmax),
            ob(a, t, na) = objective(dc, a, m, cp, g0, t, na),
            ob1(a, t, na) = objective(dc, a, m, cp, g1, t, na)

            # Locally optimal solution for estimating the whole of θ
            @test ob(a1, t1, na_ml) ≈ 7.3887 rtol = 1e-4
            # the Gateaux derivative should be about zero at the design points of the solution
            # compare DeltaMethod identity with actual Identity
            @test ob(a1, t1, na_ml) ≈ ob(a1, t1_delta, na_ml)
            # Now the Bayesian design problems with the strong prior
            # Due to MC error the published solutions are not very precise, checking gateaux
            # derivatives makes not mutch sense here.
            @test ob1(a6, t1, na_ml) ≈ 7.3760 rtol = 1e-1
            @test exp(-ob1(a7, t2, na_ml)) ≈ 2463.3 rtol = 1e-1
            @test exp(-ob1(a8, t3, na_ml)) ≈ 0.030303 rtol = 1e-1
            @test exp(-ob1(a9, t4, na_ml)) ≈ 1.1133 rtol = 1e-1
        end
    end

    @testset "gateauxderivative" begin
        # correct handling of singular designs
        let dc = DOptimality(),
            na = FisherMatrix(),
            t1 = Identity(),
            t2 = DeltaMethod(p -> diagm([1, 1, 1])),
            m = EmaxModel(1),
            cp = CopyDose(),
            pk = DiscretePrior([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ds = DesignInterval(:dose => (0, 10)),
            d = one_point_design([5])

            # explicit inversions in both cases
            @test isnan(gateauxderivative(dc, d, [d], m, cp, pk, t1, na)[1])
            @test isnan(gateauxderivative(dc, d, [d], m, cp, pk, t2, na)[1])
        end

        let dc = DOptimality(),
            na_ml = FisherMatrix(),
            trafo = Identity(),
            m = EmaxModel(1),
            cp = CopyDose(),
            p1 = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            p2 = EmaxPar(; e0 = 5, emax = -3, ec50 = 2),
            pk1 = DiscretePrior([p1]),
            pk2 = DiscretePrior([p1, p2], [0.75, 0.25]),
            pk3 = DiscretePrior([p1, p2]),
            ds = DesignInterval(:dose => (0, 10)),
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
            to_dirac(d) = map(one_point_design, designpoints(simplify_drop(d, 0))),
            gd(s, d, pk, na) = gateauxderivative(dc, s, to_dirac(d), m, cp, pk, trafo, na)

            #! format: off
            @test_throws "one-point design" gateauxderivative(
                dc, sol, [equidistant_design(ds, 2)], m, cp, pk1, trafo, na_ml,
            )
            #! format: on
            @test all(abs.(gd(sol, sol, pk1, na_ml)) .<= sqrt(eps()))
            @test all(abs.(gd(sol, not_sol, pk1, na_ml)) .> 0.01)
            @test all(abs.(gd(not_sol, not_sol, pk2, na_ml)) .> 0.1)
            @test all(abs.(gd(not_sol, not_sol, pk3, na_ml)) .> 0.1)
        end

        # DeltaMethod for Atkinson et al. examples
        let ds = DesignInterval(:time => [0, 48]),
            g0 = DiscretePrior([TPCPar(; a = 4.298, e = 0.05884, s = 21.80)]),
            _ = seed!(4711),
            m = TPCMod(1),
            cp = CopyTime(),
            dc = DOptimality(),
            na_ml = FisherMatrix(),
            # Some designs from Tables 1 and 2, and corresponding transformations
            a1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3),
            t1 = Identity(),
            t1_delta = DeltaMethod(p -> diagm(ones(3))),
            dir = [one_point_design([t]) for t in range(0, 48; length = 21)],
            gd(a, t, na) = gateauxderivative(dc, a, dir, m, cp, g0, t, na),
            dp2dir(d) = [one_point_design(dp) for dp in designpoints(d)],
            abs_gd_at_sol_dp(a, t) =
                abs.(gateauxderivative(dc, a, dp2dir(a), m, cp, g0, t, na_ml))

            # Locally optimal solution for estimating the whole of θ
            @test maximum(gd(a1, t1, na_ml)) <= 0
            # the Gateaux derivative should be about zero at the design points of the solution
            @test all(abs_gd_at_sol_dp(a1, t1) .< 1e-4)
            # compare DeltaMethod identity with actual Identity
            @test gd(a1, t1, na_ml) ≈ gd(a1, t1_delta, na_ml)
            # Now the Bayesian design problems with the strong prior
            # Due to MC error the published solutions are not very precise, checking gateaux
            # derivatives makes not mutch sense here.
        end
    end

    @testset "optimize_design" begin
        # Can we find the locally D-optimal design?
        let ds = DesignInterval(:dose => (0, 10)),
            p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            sol = emax_solution(p, ds),
            dp = DesignProblem(;
                design_criterion = DOptimality(),
                design_space = ds,
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = DiscretePrior([p]),
                transformation = Identity(),
                normal_approximation = FisherMatrix(),
            ),
            pso = Pso(; iterations = 50, swarmsize = 20),
            optim(; kwargs...) = optimize_design(pso, dp; kwargs...),
            # search from a random starting design
            _ = seed!(4711),
            (d1, o1) = optim(),
            # search with lower and upper bound already known and fixed, uniform weights
            _ = seed!(4711),
            cand = equidistant_design(ds, 3),
            (d2, o2) = optim(; prototype = cand, fixedweights = 1:3, fixedpoints = [1, 3])

            @test d1.weight ≈ sol.weight rtol = 1e-3
            for k in 1:3
                @test d1.designpoint[k] ≈ sol.designpoint[k] atol = 1e-2
            end

            @test d2.weight ≈ sol.weight rtol = 1e-3
            for k in 1:3
                @test d2.designpoint[k] ≈ sol.designpoint[k] atol = 1e-3
            end
            # fixed weights should not change, using ≈ because numerically 1-2/3 != 1/3
            @test all(map(d -> all(d.weight .≈ 1 / 3), o2.trace_x))
            # fixed points should not move
            @test all(map(d -> d.designpoint[1][1], o2.trace_x) .== ds.lowerbound[1])
            @test all(map(d -> d.designpoint[3][1], o2.trace_x) .== ds.upperbound[1])
            # the middle point should only get closer to the optimal one
            # (note that we don't need to sort the atoms of d)
            @test issorted(
                map(d -> abs(d.designpoint[2][1] - sol.designpoint[2][1]), o2.trace_x),
                rev = true,
            )

            @test_throws "outside design space" optim(
                prototype = uniform_design([[0], [20]]),
            )
            @test_throws "must match" optim(prototype = uniform_design([[0, 1], [1, 0]]))
            # `minposdist` doesn't exist, the correct argument name is `mindist`. Because we
            # have not implemented `simplify_unique()` for EmaxModel, the generic method should
            # complain about gobbling up `minposdist` in its varargs. (We can't test this in
            # designmeasure.jl because we need a Model and a CovariateParameterization in order
            # to call `simplify()`.)
            @test_logs(
                (
                    :warn,
                    "unused keyword arguments given to generic `simplify_unique` method",
                ),
                optim(minposdist = 1e-2)
            )
        end

        # fixed weights and / or points should never change
        let p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            dp = DesignProblem(;
                design_criterion = DOptimality(),
                design_space = ds = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = DiscretePrior([p]),
                transformation = Identity(),
                normal_approximation = FisherMatrix(),
            ),
            pso = Pso(; iterations = 2, swarmsize = 5),
            # this is not the optimal solution
            prototype =
                DesignMeasure([0.1, 0.5, 0.0, 0.0, 0.4], [[0], [5], [7], [8], [10]]),
            #! format: off
            opt(; fw = Int64[], fp = Int64[]) = optimize_design(
                pso, dp;
                prototype = prototype, fixedweights = fw, fixedpoints = fp, trace_state = true,
            ),
            #! format: on
            is_const_w(o, ref, k) = all([
                all(map(d -> d.weight[k] == ref.weight[k], s.x)) for s in o.trace_state
            ]),
            is_const_d(o, ref, k) = all([
                all(map(d -> d.designpoint[k] == ref.designpoint[k], s.x)) for
                s in o.trace_state
            ]),
            _ = seed!(4711),
            (_, o1) = opt(; fw = [2], fp = [2]),
            (_, o2) = opt(; fw = [2]),
            (_, o3) = opt(; fp = [2]),
            (_, o4) = opt(; fw = [5], fp = [5]),
            (_, o5) = opt(; fw = [1, 5], fp = [5])

            @test is_const_w(o1, prototype, 2)
            @test is_const_d(o1, prototype, 2)
            @test is_const_w(o2, prototype, 2)
            @test is_const_d(o3, prototype, 2)
            @test is_const_w(o4, prototype, 5)
            @test is_const_d(o4, prototype, 5)
            @test is_const_w(o5, prototype, 1)
            @test is_const_w(o5, prototype, 5)
            @test is_const_d(o5, prototype, 5)

            @test_logs(
                (:warn, "fixed weights already sum to one"),
                match_mode = :any,
                opt(fw = [1, 2, 5])
            )
        end
    end

    @testset "refine_design" begin
        # Does refinement work?
        let p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            ds = DesignInterval(:dose => (0, 10)),
            sol = emax_solution(p, ds),
            od = Pso(; iterations = 50, swarmsize = 100),
            ow = Pso(; iterations = 50, swarmsize = 50),
            _ = seed!(1234),
            cand = uniform_design([[[5]]; designpoints(sol)[[1, 3]]]),
            dp = DesignProblem(design_criterion = DOptimality(),
                               normal_approximation = FisherMatrix(),
                               transformation = Identity(),
                               model = EmaxModel(1),
                               covariate_parameterization = CopyDose(),
                               prior_knowledge = DiscretePrior([p]),
                               design_space = ds),
            (r, rd, rw) = refine_design(od, ow, 3, cand, dp)

            @test abs(designpoints(r)[2][1] - designpoints(sol)[2][1]) <
                  abs(designpoints(cand)[1][1] - designpoints(sol)[2][1])
            @test issorted([r.maximum for r in rw])
            @test all([r.maximum > 0 for r in rd])

            # When the direction of steepest ascent is in the support of the candidate, the
            # support of the intermediate design measure should not grow. When merged, the
            # new point should move to the front of the vector of design points. We check
            # this here by starting with a candidate that has the correct design points, but
            # unequal weights. Then we do one step of refinement and examine the (unsorted!)
            # results.
            near_sol = DesignMeasure([0.6, 0.3, 0.1], designpoints(sol))
            (s, sd, sw) = refine_design(od, ow, 1, near_sol, dp)
            @test length(weights(sw[1].maximizer)) == 3
            @test designpoints(sw[1].maximizer)[1] == designpoints(near_sol)[3]
        end
    end
end
end
