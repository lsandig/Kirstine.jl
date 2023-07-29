# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module TypesTest
using Test
using Kirstine

include("example-testpar.jl")

@testset "types.jl" begin
    @testset "PriorSample" begin
        # error handling in constructors
        let pars = [TestPar2(1, 2), TestPar2(3, 4)]
            @test_throws "must be equal" PriorSample(pars, [0])
            @test_throws "non-negative" PriorSample(pars, [-0.5, 1.5])
            @test_throws "sum to one" PriorSample(pars, [0.5, 1.5])
        end

        # constructor with default uniform weights
        let p = PriorSample([TestPar2(1, 2), TestPar2(3, 4)])
            @test p.weight == [0.5, 0.5]
            @test p.p == [TestPar2(1, 2), TestPar2(3, 4)]
        end
    end

    @testset "DesignMeasure" begin
        # Note: the outer constructors are covered in designmeasure.jl.
        @test_throws "must be equal" DesignMeasure([[1], [2]], [0])
        @test_throws "identical lengths" DesignMeasure([[1], [2, 3]], [0.5, 0.5])
        @test_throws "non-negative" DesignMeasure([[1], [2]], [-0.5, 1.5])
        @test_throws "sum to one" DesignMeasure([[1], [2]], [0.1, 0.2])

        let d = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5])
            @test d.weight == [0.2, 0.3, 0.5]
            @test d.designpoint == [[1], [42], [9]]
        end
    end

    @testset "SignedMeasure" begin
        @test_throws "must be equal" Kirstine.SignedMeasure([[1], [2]], [0])
        @test_throws "identical lengths" Kirstine.SignedMeasure([[1], [2, 3]], [1, -2])

        let s = Kirstine.SignedMeasure([[1, 4], [2, 3], [5, 6]], [1, -2, 0.5])
            @test s.weight == [1, -2, 0.5]
            @test s.atom == [[1, 4], [2, 3], [5, 6]]
        end
    end

    @testset "DesignInterval" begin
        @test_throws "must be identical" DesignInterval([:a], [1, 2], [3, 4])
        @test_throws "must be identical" DesignInterval([:a, :b], [2], [3, 4])
        @test_throws "must be identical" DesignInterval([:a, :b], [1, 2], [4])
        @test_throws "strictly larger" DesignInterval([:a, :b], [1, 2], [0, 4])
        @test_throws "strictly larger" DesignInterval([:a, :b], [1, 2], [1, 4])

        # inner constructor: check conversion to Float64 and tuples
        let dr = DesignInterval([:a, :b], [0, 0], [1, 2])
            @test dr.name === (:a, :b)
            @test dr.lowerbound === (0.0, 0.0)
            @test dr.upperbound === (1.0, 2.0)
        end
    end

    @testset "DesignConstraints" begin
        let dr = DesignInterval(:a => (0, 1)), dcon = Kirstine.DesignConstraints
            @test_throws DimensionMismatch dcon(dr, [true], [true, false])
        end
    end

    @testset "WorkMatrices" begin
        let m = 2, r = 4, t = 3, wm = Kirstine.WorkMatrices(m, r, t)
            @test size(wm.r_x_r) == (r, r)
            @test size(wm.t_x_t) == (t, t)
            @test size(wm.r_x_t) == (r, t)
            @test size(wm.t_x_r) == (t, r)
            @test size(wm.m_x_r) == (m, r)
        end
    end
end
end
