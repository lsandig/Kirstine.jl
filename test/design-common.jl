# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

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
            dr = DesignInterval(:dose => (0, 10)),
            d = equidistant_design(dr, 3),
            dcon = Kirstine.DesignConstraints,
            c = dcon(d, dr, [2], [3])

            # Trigger erros for out-of-range weight / designpoint indices.
            @test_throws "between 1 and 3" dcon(d, dr, [0], Int64[])
            @test_throws "between 1 and 3" dcon(d, dr, [4], Int64[])
            @test_throws "between 1 and 3" dcon(d, dr, Int64[], [-1])
            @test_throws "between 1 and 3" dcon(d, dr, Int64[], [5])
            # If only one weight index is not given as fixed, it is still implicitly fixed
            # because of the simplex constraint. In this case we want to have it fixed
            # explicitly.
            @test_logs(
                (:info, "explicitly fixing implicitly fixed weight"),
                dcon(d, dr, [1, 3], Int64[])
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
        # DeltaMethod
        let pk = PriorSample([TestPar2(1, 2), TestPar2(-1, -2)]),
            dt1 = p -> [p.a; p.b], # too few columns
            D1 = DeltaMethod(dt1),
            dt2 = p -> p.a > 0 ? [p.a p.b] : [p.a p.b; p.a p.b], # different number of rows
            D2 = DeltaMethod(dt2),
            dt3 = p -> [3 * p.a -p.b^2],
            D3 = DeltaMethod(dt3),
            tc = Kirstine.precalculate_trafo_constants(D3, pk)

            @test_throws "2 columns" Kirstine.precalculate_trafo_constants(D1, pk)
            @test_throws "identical" Kirstine.precalculate_trafo_constants(D2, pk)
            @test isa(tc, Kirstine.TCDeltaMethod)
            @test Kirstine.codomain_dimension(tc) == 1
            @test tc.jm == [[3 -4], [-3 -4]]
        end

        # Identity
        let pk = PriorSample([TestPar2(1, 2), TestPar2(-1, -2)]),
            id = Identity(),
            tc = Kirstine.precalculate_trafo_constants(id, pk)

            @test isa(tc, Kirstine.TCIdentity)
            @test Kirstine.codomain_dimension(tc) == 2
        end
    end

    @testset "informationmatrix!" begin
        let nim = zeros(3, 3),
            jm = zeros(1, 3),
            w = [0.25, 0.75],
            m = EmaxModel(1),
            ic = Kirstine.invcov(m),
            c = [Dose(0), Dose(5)],
            p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            na = FisherMatrix(),
            res = Kirstine.informationmatrix!(nim, jm, w, m, ic, c, p, na)

            ref = mapreduce(+, enumerate(c)) do (k, dose)
                Kirstine.jacobianmatrix!(jm, m, dose, p)
                return w[k] * ic * jm' * jm
            end

            # returns first argument?
            @test res === nim
            # same result as non-BLAS computation?
            @test Symmetric(res) == ref
            # complement of upper triangle not used?
            @test res[2, 1] == 0
            @test res[3, 1] == 0
            @test res[3, 2] == 0
        end
    end

    @testset "informationmatrix" begin
        let d = DesignMeasure([0] => 0.25, [5] => 0.75),
            m = EmaxModel(1),
            cp = CopyDose(),
            p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            na = FisherMatrix(),
            res = informationmatrix(d, m, cp, p, na)

            ref = [16 6 -6; 6 3 -3; -6 -3 3] ./ 16

            @test res ≈ ref
        end
    end

    @testset "log_det!" begin
        let A = rand(Float64, 3, 3), B = A * A'

            # note: log_det! overwrites B, so it can't be called first
            @test log(det(B)) ≈ Kirstine.log_det!(B)
            @test Kirstine.log_det!([1.0 0.0; 0.0 0.0]) == -Inf
        end
    end

    @testset "apply_transformation" begin
        # Applying a DeltaMethod transformation should give the same results whether the
        # incoming normalized information matrix is already inverted or not. For a DeltaMethod
        # that is actually an identity transformation in disguise, the result should simply be
        # the inverse of the argument.
        #
        # Applying a DeltaMethod transformation _always_ returns an inverted normalized
        # information matrix.
        #
        # Note: we have to recreate the circumstances in which apply_transformation! is called:
        # nim is allowed to be only upper triangular, and is allowed to be overwritten. Hence we
        # must use deepcopys, and Symmetric wrappers where necessary.
        let pk = PriorSample([TestPar3(1, 2, 3)]),
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
            (tnim1, i1) = Kirstine.apply_transformation!(wm1, false, ctid, 1),
            (tnim2, i2) = Kirstine.apply_transformation!(wm2, true, ctid, 1),
            # scaling parameters should be able to be pulled out
            (tnim3, _) = Kirstine.apply_transformation!(wm3, false, ctsc, 1),
            (tnim4, _) = Kirstine.apply_transformation!(wm4, true, ctsc, 1)

            @test Symmetric(tnim1) ≈ Symmetric(inv_nim)
            @test tnim1 === wm1.t_x_t
            @test i1 = true
            @test Symmetric(tnim2) ≈ Symmetric(inv_nim)
            @test tnim2 === wm2.t_x_t
            @test i1 == true
            @test Symmetric(tnim3) ≈ Symmetric(tnim4)
            @test tnim3 === wm3.t_x_t
            @test tnim4 === wm4.t_x_t
            @test det(Symmetric(tnim3)) ≈ 4^2 * det(Symmetric(inv_nim))
        end

        # The Identity transformation simply passes through the input matrix and its
        # inversion flag unchanged.
        let pk = PriorSample([TestPar3(1, 2, 3)]),
            t = Identity(),
            tc = Kirstine.precalculate_trafo_constants(t, pk),
            _ = seed!(4321),
            A = reshape(rand(9), 3, 3),
            nim = collect(UpperTriangular(A' * A)),
            inv_nim = collect(UpperTriangular(inv(A' * A))),
            m = 1,
            r = 3,
            t = 3,
            wm1 = Kirstine.WorkMatrices(m, r, t),
            wm2 = Kirstine.WorkMatrices(m, r, t),
            # workaround for `.=` not being valid let statement syntax
            _ = broadcast!(identity, wm1.r_x_r, nim),
            _ = broadcast!(identity, wm2.r_x_r, inv_nim),
            (tnim1, i1) = Kirstine.apply_transformation!(wm1, false, tc, 1),
            (tnim2, i2) = Kirstine.apply_transformation!(wm2, true, tc, 1)

            @test Symmetric(tnim1) ≈ Symmetric(nim)
            @test tnim1 === wm1.t_x_t
            @test i1 == false
            @test Symmetric(tnim2) ≈ Symmetric(inv_nim)
            @test tnim2 === wm2.t_x_t
            @test i2 == true
        end
    end

    @testset "allocate_initialize_covariates" begin
        let d = DesignMeasure([[0], [1], [2]], [0.2, 0.3, 0.5]),
            m = EmaxModel(1),
            cp = CopyDose(),
            c = Kirstine.allocate_initialize_covariates(d, m, cp)

            @test length(c) == length(designpoints(d))
            @test c[1].dose == designpoints(d)[1][1]
            @test c[2].dose == designpoints(d)[2][1]
            @test c[3].dose == designpoints(d)[3][1]
        end
    end

    @testset "inverse_information_matrices" begin
        let d = DesignMeasure([[1], [5], [9]], [0.1, 0.2, 0.7]),
            m = EmaxModel(1),
            cp = CopyDose(),
            pk = PriorSample([
                EmaxPar(; e0 = 1, emax = 10, ec50 = 2),
                EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            ]),
            na = FisherMatrix(),
            iim = Kirstine.inverse_information_matrices(d, m, cp, pk, na)

            # test that working matrices were deepcopied
            @test iim[1] !== iim[2]
            # compare potrf/potri result to higher-level inv call
            @test Symmetric(iim[1]) ≈ inv(informationmatrix(d, m, cp, pk.p[1], na))
            @test Symmetric(iim[2]) ≈ inv(informationmatrix(d, m, cp, pk.p[2], na))
        end
    end

    @testset "objective" begin


        # DeltaMethod A-optimality for Atkinson et al Omnibus criterion
        let g0 = PriorSample([TPCPar(; a = 4.298, e = 0.05884, s = 21.80)]),
            _ = seed!(4711),
            g1 = draw_from_prior(1000, 2),
            t1 = DeltaMethod(DOmnibus),
            t2 = DeltaMethod(p -> diagm(fill(1, 3))),
            t3 = Identity(),
            dp_for(pk, t) = DesignProblem(;
                design_region = DesignInterval(:time => [0, 48]),
                model = TPCMod(1),
                covariate_parameterization = CopyTime(),
                design_criterion = AOptimality(),
                normal_approximation = FisherMatrix(),
                prior_knowledge = pk,
                transformation = t,
            ),
            # Designs from Tables 1, 2
            a5 = DesignMeasure([0.2176] => 0.2337, [1.4343] => 0.3878, [18.297] => 0.3785),
            a10 = DesignMeasure([0.2235] => 0.2366, [1.4875] => 0.3838, [18.8293] => 0.3796)

            # Locally optimal
            @test -objective(a5, dp_for(g0, t1)) ≈ 6.883 atol = 1e-3
            # Strong prior
            @test -objective(a10, dp_for(g1, t1)) ≈ 7.1220 atol = 1e-1
            # Compare DeltaMethod identity with actual Identity
            @test objective(a5, dp_for(g0, t2)) ≈ objective(a5, dp_for(g0, t3))
        end
    end

    @testset "gateauxderivative" begin
        # correct handling of singular designs
        let t1 = Identity(),
            t2 = DeltaMethod(p -> diagm([1, 1, 1])),
            dp_for_trafo(trafo) = DesignProblem(;
                design_criterion = DOptimality(),
                normal_approximation = FisherMatrix(),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
                design_region = DesignInterval(:dose => (0, 10)),
                transformation = trafo,
            ),
            d = one_point_design([5])

            # explicit inversions in both cases
            @test isnan(gateauxderivative(d, [d], dp_for_trafo(t1))[1])
            @test isnan(gateauxderivative(d, [d], dp_for_trafo(t2))[1])
        end

        let p1 = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            p2 = EmaxPar(; e0 = 5, emax = -3, ec50 = 2),
            pk1 = PriorSample([p1]),
            pk2 = PriorSample([p1, p2], [0.75, 0.25]),
            pk3 = PriorSample([p1, p2]),
            dr = DesignInterval(:dose => (0, 10)),
            dp_for_pk(pk) = DesignProblem(;
                design_criterion = DOptimality(),
                normal_approximation = FisherMatrix(),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = pk,
                design_region = dr,
                transformation = Identity(),
            ),
            # sol is optimal for pk1
            sol = emax_solution(p1, dr),
            a = dr.lowerbound[1],
            b = dr.upperbound[1],
            x_star = sol.designpoint[2][1],
            # not_sol is not optimal for any of pk1, pk2, pk3
            not_sol = DesignMeasure(
                [[a + 0.1 * (b - a)], [x_star * 1.1], [a + (0.9 * (b - a))]],
                [0.2, 0.3, 0.5],
            ),
            # a design with fewer than three points is singular
            to_dirac(d) = map(one_point_design, designpoints(simplify_drop(d, 0))),
            gd(s, d, pk) = gateauxderivative(s, to_dirac(d), dp_for_pk(pk))

            @test_throws "one-point design" gateauxderivative(sol, [sol], dp_for_pk(pk1))
            @test all(abs.(gd(sol, sol, pk1) .<= sqrt(eps())))
            @test all(abs.(gd(sol, not_sol, pk1)) .> 0.01)
            @test all(abs.(gd(not_sol, not_sol, pk2)) .> 0.1)
            @test all(abs.(gd(not_sol, not_sol, pk3)) .> 0.1)
        end

        # DeltaMethod for Atkinson et al. examples
        let dp_for_trafo(trafo) = DesignProblem(;
                design_criterion = DOptimality(),
                normal_approximation = FisherMatrix(),
                model = TPCMod(1),
                covariate_parameterization = CopyTime(),
                prior_knowledge = PriorSample([
                    TPCPar(; a = 4.298, e = 0.05884, s = 21.80),
                ]),
                design_region = DesignInterval(:time => [0, 48]),
                transformation = trafo,
            ),
            # Some designs from Tables 1 and 2, and corresponding transformations
            a1 = DesignMeasure([0.2288] => 1 / 3, [1.3886] => 1 / 3, [18.417] => 1 / 3),
            t1 = Identity(),
            t1_delta = DeltaMethod(p -> diagm(ones(3))),
            dir = [one_point_design([t]) for t in range(0, 48; length = 21)],
            gd(a, t) = gateauxderivative(a, dir, dp_for_trafo(t)),
            dp2dir(d) = [one_point_design(dp) for dp in designpoints(d)],
            abs_gd_at_sol_dp(a, t) = abs.(gateauxderivative(a, dp2dir(a), dp_for_trafo(t)))

            # Locally optimal solution for estimating the whole of θ
            @test maximum(gd(a1, t1)) <= 0
            # the Gateaux derivative should be about zero at the design points of the solution
            @test all(abs_gd_at_sol_dp(a1, t1) .< 1e-4)
            # compare DeltaMethod identity with actual Identity
            @test gd(a1, t1) ≈ gd(a1, t1_delta)
            # Now the Bayesian design problems with the strong prior
            # Due to MC error the published solutions are not very precise, checking gateaux
            # derivatives makes not mutch sense here.
        end

        # DeltaMethod A-optimality for Atkinson et al Omnibus criterion
        let g0 = PriorSample([TPCPar(; a = 4.298, e = 0.05884, s = 21.80)]),
            _ = seed!(4711),
            g1 = draw_from_prior(1000, 2),
            t1 = DeltaMethod(DOmnibus),
            t2 = DeltaMethod(p -> diagm(fill(1, 3))),
            t3 = Identity(),
            dp_for(pk, t) = DesignProblem(;
                design_region = DesignInterval(:time => [0, 48]),
                model = TPCMod(1),
                covariate_parameterization = CopyTime(),
                design_criterion = AOptimality(),
                normal_approximation = FisherMatrix(),
                prior_knowledge = pk,
                transformation = t,
            ),
            # Designs from Tables 1, 2
            a5 = DesignMeasure([0.2176] => 0.2337, [1.4343] => 0.3878, [18.297] => 0.3785),
            a10 =
                DesignMeasure([0.2235] => 0.2366, [1.4875] => 0.3838, [18.8293] => 0.3796),
            dir = [one_point_design([t]) for t in range(0, 48; length = 21)],
            gd(a, pk, t) = gateauxderivative(a, dir, dp_for(pk, t)),
            dp2dir(d) = [one_point_design(dp) for dp in designpoints(d)],
            abs_gd_at_sol_dp(a, pk, t) =
                abs.(gateauxderivative(a, dp2dir(a), dp_for(pk, t)))

            # Locally optimal
            @test maximum(gd(a5, g0, t1)) <= 0
            @test maximum(abs_gd_at_sol_dp(a5, g0, t1)) < 1e-2
            # Bayesian optimal (less strict comparison due to MC error)
            @test maximum(gd(a5, g1, t1)) <= 1e-1
            @test maximum(abs_gd_at_sol_dp(a5, g1, t1)) < 0.5
            # compare DeltaMethod identity with actual Identity
            @test gd(a5, g0, t2) ≈ gd(a5, g0, t3)
        end
    end

    @testset "solve with DirectMaximization" begin
        # Can we find the locally D-optimal design?
        let dr = DesignInterval(:dose => (0, 10)),
            p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            sol = emax_solution(p, dr),
            dp = DesignProblem(;
                design_criterion = DOptimality(),
                design_region = dr,
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([p]),
                transformation = Identity(),
                normal_approximation = FisherMatrix(),
            ),
            pso = Pso(; iterations = 50, swarmsize = 20),
            # search from a random starting design
            _ = seed!(4711),
            (d1, r1) = solve(
                dp,
                DirectMaximization(; optimizer = pso, prototype = random_design(dr, 3)),
            ),
            o1 = r1.or, # unwrap the OptimizationResult
            # search with lower and upper bound already known and fixed, uniform weights
            _ = seed!(4711),
            cand = equidistant_design(dr, 3),
            (d2, r2) = solve(
                dp,
                DirectMaximization(;
                    optimizer = pso,
                    prototype = cand,
                    fixedweights = 1:3,
                    fixedpoints = [1, 3],
                ),
            ),
            o2 = r2.or

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
            @test all(map(d -> d.designpoint[1][1], o2.trace_x) .== dr.lowerbound[1])
            @test all(map(d -> d.designpoint[3][1], o2.trace_x) .== dr.upperbound[1])
            # the middle point should only get closer to the optimal one
            # (note that we don't need to sort the atoms of d)
            @test issorted(
                map(d -> abs(d.designpoint[2][1] - sol.designpoint[2][1]), o2.trace_x),
                rev = true,
            )

            @test_throws "outside design region" solve(
                dp,
                DirectMaximization(;
                    optimizer = pso,
                    prototype = uniform_design([[0], [20]]),
                ),
            )
            @test_throws "must match" solve(
                dp,
                DirectMaximization(;
                    optimizer = pso,
                    prototype = uniform_design([[0, 1], [1, 0]]),
                ),
            )
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
                solve(
                    dp,
                    DirectMaximization(;
                        optimizer = pso,
                        prototype = equidistant_design(dr, 3),
                    ),
                    minposdist = 1e-2,
                )
            )
        end

        # fixed weights and / or points should never change
        let p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            dp = DesignProblem(;
                design_criterion = DOptimality(),
                design_region = dr = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([p]),
                transformation = Identity(),
                normal_approximation = FisherMatrix(),
            ),
            # this is not the optimal solution
            prototype =
                DesignMeasure([[0], [5], [7], [8], [10]], [0.1, 0.5, 0.0, 0.0, 0.4]),
            opt(; fw = Int64[], fp = Int64[]) = solve(
                dp,
                DirectMaximization(;
                    optimizer = Pso(; iterations = 2, swarmsize = 5),
                    prototype = prototype,
                    fixedweights = fw,
                    fixedpoints = fp,
                );
                trace_state = true,
            ),
            is_const_w(o, ref, k) = all([
                all(map(d -> d.weight[k] == ref.weight[k], s.x)) for s in o.or.trace_state
            ]),
            is_const_d(o, ref, k) = all([
                all(map(d -> d.designpoint[k] == ref.designpoint[k], s.x)) for
                s in o.or.trace_state
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

    @testset "solve with Exchange" begin
        # Does refinement work?
        let p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            dr = DesignInterval(:dose => (0, 10)),
            sol = emax_solution(p, dr),
            od = Pso(; iterations = 50, swarmsize = 100),
            ow = Pso(; iterations = 50, swarmsize = 50),
            _ = seed!(1234),
            cand = uniform_design([[[5]]; designpoints(sol)[[1, 3]]]),
            dp = DesignProblem(;
                design_criterion = DOptimality(),
                normal_approximation = FisherMatrix(),
                transformation = Identity(),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([p]),
                design_region = dr,
            ),
            (d, r) = solve(dp, Exchange(; ow = ow, od = od, candidate = cand, steps = 3))

            @test abs(designpoints(d)[2][1] - designpoints(sol)[2][1]) <
                  abs(designpoints(cand)[1][1] - designpoints(sol)[2][1])
            @test issorted([r.maximum for r in r.orw])
            @test all([r.maximum > 0 for r in r.ord])

            # When the direction of steepest ascent is in the support of the candidate, the
            # support of the intermediate design measure should not grow. When merged, the
            # new point should move to the front of the vector of design points. We check
            # this here by starting with a candidate that has the correct design points, but
            # unequal weights. Then we do one step of refinement and examine the (unsorted!)
            # results.
            near_sol = DesignMeasure(designpoints(sol), [0.6, 0.3, 0.1])
            (s, r) =
                solve(dp, Exchange(; ow = ow, od = od, candidate = near_sol, steps = 1))
            @test length(weights(r.orw[1].maximizer)) == 3
            @test designpoints(r.orw[1].maximizer)[1] == designpoints(near_sol)[3]
        end
    end
end
end
