# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignNonlinearRegressionTests
using Test
using Kirstine
using LinearAlgebra: Symmetric

include("example-emax.jl")
include("example-vector.jl")

@testset "model-nonlinear-regression.jl" begin
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
end
end
