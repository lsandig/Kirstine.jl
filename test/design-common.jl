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

    @testset "solve_with" begin
        # DirectMaximization
        let dp = DesignProblem(;
                design_criterion = DOptimality(),
                design_region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            pso = Pso(; iterations = 20, swarmsize = 10),
            # random init, no additional constraints
            str1 =
                DirectMaximization(; optimizer = pso, prototype = random_design(dp.dr, 3)),
            # uniform init, weights and some points fixed
            str2 = DirectMaximization(;
                optimizer = pso,
                prototype = equidistant_design(dp.dr, 3),
                fixedweights = 1:3,
                fixedpoints = [1, 3],
            ),
            # prototypes incompatible with design region
            str3 = DirectMaximization(;
                optimizer = pso,
                prototype = uniform_design([[0], [20]]),
            ),
            str4 = DirectMaximization(;
                optimizer = pso,
                prototype = uniform_design([[0, 1], [1, 0]]),
            ),
            r1 = Kirstine.solve_with(dp, str1, false),
            # additionally trace optimizer states
            r2 = Kirstine.solve_with(dp, str2, true)

            # errors from check_compatible() in DesignConstraints constructor
            @test_throws "outside design region" solve(dp, str3)
            @test_throws "must match" solve(dp, str4)
            # result type
            @test isa(r1, DirectMaximizationResult)
            @test isa(r2, DirectMaximizationResult)
            # traced state
            @test length(r1.or.trace_state) == 1
            @test length(r2.or.trace_state) == 21
            # increasing objective
            @test issorted(r1.or.trace_fx)
            @test issorted(r2.or.trace_fx)
            # fixed things
            @test all(map(d -> all(d.weight .≈ 1 / 3), r2.or.trace_x))
            @test all(map(d -> d.designpoint[1][1], r2.or.trace_x) .== dp.dr.lowerbound[1])
            @test all(map(d -> d.designpoint[3][1], r2.or.trace_x) .== dp.dr.upperbound[1])
            # non-fixed design point converges to 2.5
            @test issorted(
                map(d -> abs(d.designpoint[2][1] - 2.5), r2.or.trace_x),
                rev = true,
            )
        end

        # DirectMaximization, but with fixed weights / design points that we know are not
        # optimal, in various combinations
        let dp = DesignProblem(;
                design_criterion = DOptimality(),
                design_region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            pt = DesignMeasure([[0], [5], [7], [8], [10]], [0.1, 0.5, 0.0, 0.0, 0.4]),
            sws(; fw, fp) = Kirstine.solve_with(
                dp,
                DirectMaximization(;
                    optimizer = Pso(; iterations = 10, swarmsize = 5),
                    prototype = pt,
                    fixedweights = fw,
                    fixedpoints = fp,
                ),
                true,
            ),
            r1 = sws(; fw = [2], fp = [2]),
            r2 = sws(; fw = [2], fp = Int64[]),
            r3 = sws(; fw = Int64[], fp = [2]),
            r4 = sws(; fw = [5], fp = [5]),
            r5 = sws(; fw = [1, 5], fp = [5]),
            isconstw(r, k) = all([
                all(map(d -> d.weight[k] == weights(pt)[k], s.x)) for s in r.or.trace_state
            ]),
            isconstp(r, k) = all([
                all(map(d -> d.designpoint[k] == designpoints(pt)[k], s.x)) for
                s in r.or.trace_state
            ])

            @test isconstw(r1, 2)
            @test isconstp(r1, 2)
            @test isconstw(r2, 2)
            @test isconstp(r3, 2)
            @test isconstw(r4, 5)
            @test isconstp(r4, 5)
            @test isconstw(r5, 1)
            @test isconstw(r5, 5)
            @test isconstp(r5, 5)
            @test_logs(
                (:warn, "fixed weights already sum to one"),
                match_mode = :any,
                sws(fw = [1, 2, 5], fp = Int64[])
            )
        end

        # Exchange
        let dp = DesignProblem(;
                design_criterion = DOptimality(),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
                design_region = DesignInterval(:dose => (0, 10)),
            ),
            od = Pso(; iterations = 20, swarmsize = 10),
            ow = Pso(; iterations = 20, swarmsize = 10),
            str1 = Exchange(;
                ow = ow,
                od = od,
                candidate = uniform_design([[5], [0], [10]]),
                steps = 3,
            ),
            _ = seed!(54321),
            r1 = Kirstine.solve_with(dp, str1, true),
            # When the direction of steepest ascent is in the support of the candidate, the
            # number of design points of the intermediate design measure should not
            # increase. When merged, the new point should move to the front of the vector of
            # design points. We check this here by starting with a candidate that has the
            # correct design points, but unequal weights.
            str2 = Exchange(;
                ow = ow,
                od = od,
                candidate = DesignMeasure([[0], [2.5], [10]], [0.6, 0.3, 0.1]),
                steps = 1,
            ),
            _ = seed!(13579),
            r2 = Kirstine.solve_with(dp, str2, false),
            # prototypes incompatible with design region
            str3 = Exchange(;
                ow = ow,
                od = od,
                candidate = uniform_design([[0], [20]]),
                steps = 1,
            ),
            str4 = Exchange(;
                ow = ow,
                od = od,
                candidate = uniform_design([[0, 1], [1, 0]]),
                steps = 1,
            )

            # errors from check_compatible
            @test_throws "outside design region" solve(dp, str3)
            @test_throws "must match" solve(dp, str4)
            # result type
            @test isa(r1, ExchangeResult)
            @test isa(r2, ExchangeResult)
            # traced state
            @test all(or -> length(or.trace_state) == 21, r1.ord)
            @test all(or -> length(or.trace_state) == 21, r1.orw)
            @test all(or -> length(or.trace_state) == 1, r2.ord)
            @test all(or -> length(or.trace_state) == 1, r2.orw)
            # increasing objective (from step to step)
            @test issorted(r1.orw, by = or -> or.maximum)
            @test issorted(r2.orw, by = or -> or.maximum)
            # no new point, but last one of the candidate
            @test length(weights(r2.orw[1].maximizer)) == 3
            @test designpoints(r2.orw[1].maximizer)[1] == [10]
        end
    end
end
end
