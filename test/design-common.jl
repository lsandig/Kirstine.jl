# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignCommonTests
using Test
using Kirstine
using Random: seed!
using LinearAlgebra: Symmetric, UpperTriangular, tr, det, diagm

include("example-testpar.jl")
include("example-emax.jl")
include("example-vector.jl")

@testset "design-common.jl" begin
    @testset "WorkMatrices" begin
        let K = 5, m = 2, r = 4, t = 3, wm = Kirstine.WorkMatrices(K, m, r, t)
            @test size(wm.r_x_r) == (r, r)
            @test size(wm.t_x_t) == (t, t)
            @test size(wm.r_x_t) == (r, t)
            @test size(wm.t_x_r) == (t, r)
            @test size(wm.m_x_r) == (m, r)
            @test length(wm.m_x_m) == K
            for k in 1:K
                @test size(wm.m_x_m[k]) == (m, m)
            end
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

    @testset "informationmatrix!" begin
        let nim = zeros(3, 3),
            jm = zeros(1, 3),
            w = [0.25, 0.75],
            m = EmaxModel(1),
            # cholesky factor of the one-element 1.0 matrix is 1.0.
            cvc = [[1.0;;] for _ in 1:length(w)],
            c = [Dose(0), Dose(5)],
            p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            na = FisherMatrix(),
            res = Kirstine.informationmatrix!(nim, jm, w, m, cvc, c, p, na)

            ref = mapreduce(+, enumerate(c)) do (k, dose)
                Kirstine.jacobianmatrix!(jm, m, dose, p)
                return w[k] * jm' * inv([1.0;;]) * jm
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

        # vector unit with non-constant covariance
        let d = DesignMeasure([5] => 0.25, [20] => 0.75),
            m = VUMod(0.1, 3),
            cp = EquiTime(),
            p = VUPar(; a = 4.298, e = 0.05884, s = 21.80),
            na = FisherMatrix(),
            res = informationmatrix(d, m, cp, p, na),
            # recreate by hand
            c = [VUCovariate([0, 5, 10]), VUCovariate([0, 20, 40])],
            jm = Kirstine.jacobianmatrix!.([zeros(3, 3), zeros(3, 3)], [m], c, [p]),
            S = [diagm([0.1, 0.6, 1.1]), diagm([0.1, 2.1, 4.1])],
            F1 = jm[1]' * inv(S[1]) * jm[1],
            F2 = jm[2]' * inv(S[2]) * jm[2],
            ref = 0.25 * F1 + 0.75 * F2

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
            K = 1, # dummy, we already have the informationmatrix
            wm1 = Kirstine.WorkMatrices(K, m, r, t),
            wm2 = Kirstine.WorkMatrices(K, m, r, t),
            wm3 = Kirstine.WorkMatrices(K, m, r, t),
            wm4 = Kirstine.WorkMatrices(K, m, r, t),
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
            K = 1, # dummy, we already have the informationmatrix
            wm1 = Kirstine.WorkMatrices(K, m, r, t),
            wm2 = Kirstine.WorkMatrices(K, m, r, t),
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
end
end
