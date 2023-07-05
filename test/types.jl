module TypesTest
using Test
using Kirstine

@testset "types.jl" begin
    @testset "DiscretePrior" begin
        # error handling in constructors
        @test_throws "must be equal" DiscretePrior([0], [(a = 1, b = 2), (a = 3, b = 4)])
        @test_throws "non-negative" DiscretePrior([-0.5, 1.5], [(a = 1,), (a = 3,)])
        @test_throws "sum to one" DiscretePrior([0.5, 1.5], [(a = 1,), (a = 3,)])

        # constructor with default uniform weights
        let p = DiscretePrior([(a = 1, b = 2), (a = 3, b = 4)])
            @test p.weight == [0.5, 0.5]
            @test p.p == [(a = 1, b = 2), (a = 3, b = 4)]
        end

        # constructor for dirac measure
        let p = DiscretePrior((a = 1, b = 2))
            @test p.weight == [1.0]
            @test p.p == [(a = 1, b = 2)]
        end
    end

    @testset "DesignMeasure" begin
        # Note: the outer constructors are covered in designmeasure.jl.
        @test_throws "must be equal" DesignMeasure([0], [[1], [2]])
        @test_throws "identical lengths" DesignMeasure([0.5, 0.5], [[1], [2, 3]])
        @test_throws "non-negative" DesignMeasure([-0.5, 1.5], [[1], [2]])
        @test_throws "sum to one" DesignMeasure([0.1, 0.2], [[1], [2]])

        let d = DesignMeasure([0.2, 0.3, 0.5], [[1], [42], [9]])
            @test d.weight == [0.2, 0.3, 0.5]
            @test d.designpoint == [[1], [42], [9]]
        end
    end

    @testset "SignedMeasure" begin
        @test_throws "must be equal" Kirstine.SignedMeasure([0], [[1], [2]])
        @test_throws "identical lengths" Kirstine.SignedMeasure([1, -2], [[1], [2, 3]])

        let s = Kirstine.SignedMeasure([1, -2, 0.5], [[1, 4], [2, 3], [5, 6]])
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
        let ds = DesignInterval([:a, :b], [0, 0], [1, 2])
            @test ds.name === (:a, :b)
            @test ds.lowerbound === (0.0, 0.0)
            @test ds.upperbound === (1.0, 2.0)
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
