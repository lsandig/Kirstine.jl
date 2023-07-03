module TypesTest
using Test
using Kirstine

@testset "types.jl" begin
    @testset "DiscretePrior" begin
        # error handling in constructors
        @test_throws "must be equal" DiscretePrior([0], [[1], [2]])
        @test_throws "non-negative" DiscretePrior([-0.5, 1.5], [[1], [2]])
    end

    @testset "DesignMeasure" begin
        # error handling in constructors
        @test_throws "must be equal" DesignMeasure([0], [[1], [2]])
        @test_throws "identical lengths" DesignMeasure([0.5, 0.5], [[1], [2, 3]])
        @test_throws "non-negative" DesignMeasure([-0.5, 1.5], [[1], [2]])
        @test_throws "sum to one" DesignMeasure([0.1, 0.2], [[1], [2]])
        @test_throws "at least two rows" DesignMeasure([1 2 3])

        # outer constructors
        let d = DesignMeasure([1] => 0.2, [42] => 0.3, [9] => 0.5),
            ref = DesignMeasure([0.2, 0.3, 0.5], [[1], [42], [9]])

            @test all(d.weight .== ref.weight)
            @test all(d.designpoint .== ref.designpoint)
        end
    end

    @testset "DesignSpace" begin
        @test_throws "must be identical" DesignSpace([:a], [1, 2], [3, 4])
        @test_throws "must be identical" DesignSpace([:a, :b], [2], [3, 4])
        @test_throws "must be identical" DesignSpace([:a, :b], [1, 2], [4])
        @test_throws "strictly larger" DesignSpace([:a, :b], [1, 2], [0, 4])
        @test_throws "strictly larger" DesignSpace([:a, :b], [1, 2], [1, 4])
        let ds = DesignSpace(:a => (0, 1), :b => (0, 2)),
            ref = DesignSpace((:a, :b), (0, 0), (1, 2))

            @test all(ref.name .== ds.name)
            @test all(ref.lowerbound .== ds.lowerbound)
            @test all(ref.upperbound .== ds.upperbound)
        end
    end

    @testset "DeltaMethod" begin
        let pk = DiscretePrior([(a = 1, b = 2), (a = -1, b = -2)]),
            dt1 = p -> [p.a; p.b], # too few columns
            D1 = DeltaMethod(dt1),
            dt2 = p -> p.a > 0 ? [p.a p.b] : [p.a p.b; p.a p.b], # different number of rows
            D2 = DeltaMethod(dt2)

            @test_throws "2 columns" Kirstine.precalculate_trafo_constants(D1, pk)
            @test_throws "identical" Kirstine.precalculate_trafo_constants(D2, pk)
        end
    end
end
end
