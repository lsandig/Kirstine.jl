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
    @testset "NIMWorkspace" begin
        let r = 4, t = 3, nw = Kirstine.NIMWorkspace(r, t)
            @test_throws "must be positive" Kirstine.NIMWorkspace(0, 2)
            @test_throws "must be positive" Kirstine.NIMWorkspace(1, -1)
            @test size(nw.r_x_r) == (r, r)
            @test size(nw.t_x_t) == (t, t)
            @test size(nw.r_x_t) == (r, t)
            @test size(nw.t_x_r) == (t, r)
        end
    end

    @testset "NRWorkspace" begin
        let K = 5, m = 10, r = 3, mw = Kirstine.NRWorkspace(K, m, r)
            @test_throws "at least one design point" Kirstine.NRWorkspace(0, m, r)
            @test_throws "must be positive" Kirstine.NRWorkspace(K, m, 0)
            @test size(mw.m_x_r) == (10, 3)
            @test length(mw.m_x_m) == K
            for k in 1:K
                @test size(mw.m_x_m[k]) == (10, 10)
            end
        end
    end

    @testset "allocate_model_workspace(NonlinearRegression)" begin
        let K = 5,
            m = VUMod(1, 10),
            pk = PriorSample([VUParameter(; a = 1, e = 1, s = 1)]),
            mw = Kirstine.allocate_model_workspace(K, m, pk)

            @test isa(mw, Kirstine.NRWorkspace)
            @test size(mw.m_x_r) == (10, 3)
            @test length(mw.m_x_m) == K
            for k in 1:K
                @test size(mw.m_x_m[k]) == (10, 10)
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

    @testset "average_fishermatrix!" begin
        let nim = zeros(3, 3),
            jm = zeros(1, 3),
            w = [0.25, 0.75],
            m = EmaxModel(1),
            p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            pk = PriorSample([p]),
            # dummy -----------------------v
            nw = Kirstine.NIMWorkspace(size(jm, 2), 1),
            mw = Kirstine.allocate_model_workspace(length(w), m, pk),
            # cholesky factor of the one-element 1.0 matrix is 1.0.
            # workaround since we can't assign to fields in let blocks
            _ = setindex!(mw.m_x_m[1], 1.0, 1),
            _ = setindex!(mw.m_x_m[2], 1.0, 1),
            c = [Dose(0), Dose(5)],
            res = Kirstine.average_fishermatrix!(nw.r_x_r, mw, w, m, c, p),
            ref = mapreduce(+, enumerate(c)) do (k, dose)
                Kirstine.jacobianmatrix!(jm, m, dose, p)
                return w[k] * jm' * inv([1.0;;]) * jm
            end

            # returns workspace matrix?
            @test res === nw.r_x_r
            # same result as non-BLAS computation?
            @test Symmetric(nw.r_x_r) == ref
            # complement of upper triangle not used?
            @test nw.r_x_r[2, 1] == 0
            @test nw.r_x_r[3, 1] == 0
            @test nw.r_x_r[3, 2] == 0
        end
    end

    @testset "informationmatrix!" begin
        # The `FisherMatrix` normal approximation should just give the average Fisher
        # information matrix as NIM.
        let m = EmaxModel(1),
            w = [1.0],
            c = [Dose(1)],
            p = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            pk = PriorSample([p]),
            mw1 = Kirstine.allocate_model_workspace(1, m, pk),
            mw2 = Kirstine.allocate_model_workspace(1, m, pk),
            nim1 = zeros(3, 3),
            nim2 = zeros(3, 3),
            _ = setindex!(mw1.m_x_m[1], 1.0, 1),
            _ = setindex!(mw2.m_x_m[1], 1.0, 1),
            na = FisherMatrix(),
            res1 = Kirstine.informationmatrix!(nim1, mw1, w, m, c, p, na),
            res2 = Kirstine.average_fishermatrix!(nim2, mw2, w, m, c, p)

            @test nim1 == nim2
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
            p = VUParameter(; a = 4.298, e = 0.05884, s = 21.80),
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
            r = 3,
            t = 3,
            nw1 = Kirstine.NIMWorkspace(r, t),
            nw2 = Kirstine.NIMWorkspace(r, t),
            nw3 = Kirstine.NIMWorkspace(r, t),
            nw4 = Kirstine.NIMWorkspace(r, t),
            # workaround for `.=` not being valid let statement syntax
            _ = broadcast!(identity, nw1.r_x_r, nim),
            _ = broadcast!(identity, nw2.r_x_r, inv_nim),
            _ = broadcast!(identity, nw3.r_x_r, nim),
            _ = broadcast!(identity, nw4.r_x_r, inv_nim),
            (r1, i1) = Kirstine.apply_transformation!(nw1, false, ctid, 1),
            (r2, i2) = Kirstine.apply_transformation!(nw2, true, ctid, 1),
            # scaling parameters should be able to be pulled out
            (r3, _) = Kirstine.apply_transformation!(nw3, false, ctsc, 1),
            (r4, _) = Kirstine.apply_transformation!(nw4, true, ctsc, 1)

            @test Symmetric(nw1.t_x_t) ≈ Symmetric(inv_nim)
            @test r1 === nw1
            @test i1 = true
            @test Symmetric(nw2.t_x_t) ≈ Symmetric(inv_nim)
            @test r2 === nw2
            @test i1 == true
            @test Symmetric(nw3.t_x_t) ≈ Symmetric(nw4.t_x_t)
            @test r3 === nw3
            @test r4 === nw4
            @test det(Symmetric(nw3.t_x_t)) ≈ 4^2 * det(Symmetric(inv_nim))
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
            r = 3,
            t = 3,
            nw1 = Kirstine.NIMWorkspace(r, t),
            nw2 = Kirstine.NIMWorkspace(r, t),
            # workaround for `.=` not being valid let statement syntax
            _ = broadcast!(identity, nw1.r_x_r, nim),
            _ = broadcast!(identity, nw2.r_x_r, inv_nim),
            (r1, i1) = Kirstine.apply_transformation!(nw1, false, tc, 1),
            (r2, i2) = Kirstine.apply_transformation!(nw2, true, tc, 1)

            @test Symmetric(nw1.t_x_t) ≈ Symmetric(nim)
            @test r1 === nw1
            @test i1 == false
            @test Symmetric(nw2.t_x_t) ≈ Symmetric(inv_nim)
            @test r2 === nw2
            @test i2 == true
        end
    end

    @testset "transformed_information_matrices" begin
        # We suppose that `apply_transformation!` works as intended,
        # and only check with DeltaMethod trafo constants and non-inverted input NIM.
        let A1 = reshape(rand(9), 3, 3),
            A2 = reshape(rand(9), 3, 3),
            nim = [collect(UpperTriangular(A1' * A1)), collect(UpperTriangular(A2' * A2))],
            nim_bkp = deepcopy(nim),
            pk = PriorSample([TestPar3(1, 2, 3), TestPar3(4, 5, 6)]), # dummy values
            t = DeltaMethod(p -> diagm([0.5, 2.0, 4.0])),
            tc = Kirstine.precalculate_trafo_constants(t, pk),
            (tnim, is_inv) = Kirstine.transformed_information_matrices(nim, false, pk, tc)

            @test length(tnim) == 2
            # with the DeltaMethod, the output is always inverted
            @test is_inv == true
            # results should not be the same reference
            @test tnim[1] !== tnim[2]
            # input nim not modified
            @test nim == nim_bkp
            # actual matrices
            @test Symmetric(tnim[1]) ≈ tc.jm[1]' * inv(Symmetric(nim[1])) * tc.jm[1]
            @test Symmetric(tnim[2]) ≈ tc.jm[2]' * inv(Symmetric(nim[2])) * tc.jm[2]
        end
    end

    @testset "allocate_initialize_covariates" begin
        let d = DesignMeasure([[0], [1], [2]], [0.2, 0.3, 0.5]),
            m = EmaxModel(1),
            cp = CopyDose(),
            c = Kirstine.allocate_initialize_covariates(d, m, cp)

            @test length(c) == numpoints(d)
            @test c[1].dose == points(d)[1][1]
            @test c[2].dose == points(d)[2][1]
            @test c[3].dose == points(d)[3][1]
        end
    end

    @testset "update_covariates!" begin
        let d1 = DesignMeasure([[0], [1], [2]], [0.2, 0.3, 0.5]),
            d2 = DesignMeasure([[3], [4], [5]], [0.4, 0.4, 0.2]),
            m = EmaxModel(1),
            cp = CopyDose(),
            c = Kirstine.allocate_initialize_covariates(d1, m, cp),
            res = Kirstine.update_covariates!(c, d2, m, cp)

            @test res === c
            @test c[1].dose == points(d2)[1][1]
            @test c[2].dose == points(d2)[2][1]
            @test c[3].dose == points(d2)[3][1]
        end
    end

    @testset "update_model_workspace!(NonlinearRegression)" begin
        let m = EmaxModel(36), # note: this is `sigma_squared`!,
            pk = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 2)]),
            mw = Kirstine.allocate_model_workspace(2, m, pk),
            c = [Dose(1), Dose(2)],
            res = Kirstine.update_model_workspace!(mw, m, c)

            # cholesky factors of 1-dimensional matrices contain the sqrt
            @test mw.m_x_m[1] == [6.0;;]
            @test mw.m_x_m[2] == [6.0;;]
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
