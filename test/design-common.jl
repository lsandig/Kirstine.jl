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
        # The `FisherMatrix` normal approximation should just give the average Fisher
        # information matrix as NIM.
        let _ = seed!(1234),
            afm = rand(3, 3),
            afm_copy = deepcopy(afm),
            na = FisherMatrix(),
            res = Kirstine.informationmatrix!(afm, na)

            @test res === afm
            @test res == afm_copy
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
            ctid = Kirstine.trafo_constants(tid, pk),
            tsc = DeltaMethod(p -> diagm([0.5, 2.0, 4.0])),
            ctsc = Kirstine.trafo_constants(tsc, pk),
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
            _ = setfield!(nw1, :r_is_inv, false),
            r1 = Kirstine.apply_transformation!(nw1, tid, ctid.jm[1]),
            _ = setfield!(nw2, :r_is_inv, true),
            r2 = Kirstine.apply_transformation!(nw2, tid, ctid.jm[1]),
            # scaling parameters should be able to be pulled out
            _ = setfield!(nw3, :r_is_inv, false),
            r3 = Kirstine.apply_transformation!(nw3, tsc, ctsc.jm[1]),
            _ = setfield!(nw4, :r_is_inv, true),
            r4 = Kirstine.apply_transformation!(nw4, tsc, ctsc.jm[1])

            @test Symmetric(nw1.t_x_t) ≈ Symmetric(inv_nim)
            @test r1 === nw1
            @test r1.t_is_inv
            @test r1.r_is_inv == false # should not touch input matrix flag
            @test Symmetric(nw2.t_x_t) ≈ Symmetric(inv_nim)
            @test r2 === nw2
            @test r2.t_is_inv
            @test r2.r_is_inv
            @test Symmetric(nw3.t_x_t) ≈ Symmetric(nw4.t_x_t)
            @test r3 === nw3
            @test r4 === nw4
            @test det(Symmetric(nw3.t_x_t)) ≈ 4^2 * det(Symmetric(inv_nim))
        end

        # The Identity transformation simply passes through the input matrix and its
        # inversion flag unchanged.
        let pk = PriorSample([TestPar3(1, 2, 3)]),
            trafo = Identity(),
            tc = Kirstine.trafo_constants(trafo, pk),
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
            _ = setfield!(nw1, :r_is_inv, false),
            _ = setfield!(nw2, :r_is_inv, true),
            r1 = Kirstine.apply_transformation!(nw1, trafo, diagm(ones(3))),
            r2 = Kirstine.apply_transformation!(nw2, trafo, diagm(ones(3)))

            @test Symmetric(nw1.t_x_t) ≈ Symmetric(nim)
            @test r1 === nw1
            @test r1.t_is_inv == false
            @test r1.r_is_inv == false
            @test Symmetric(nw2.t_x_t) ≈ Symmetric(inv_nim)
            @test r2 === nw2
            @test r2.t_is_inv == true
            @test r2.r_is_inv == true
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
end
end
