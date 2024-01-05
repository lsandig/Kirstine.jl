# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignRegionTests
using Test
using Kirstine
using Random

@testset "designregion.jl" begin
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

        # outer constructor
        let dr = DesignInterval(:a => (0, 1), :b => (0, 2)),
            ref = DesignInterval((:a, :b), (0, 0), (1, 2))

            @test all(ref.name .== dr.name)
            @test all(ref.lowerbound .== dr.lowerbound)
            @test all(ref.upperbound .== dr.upperbound)
        end
    end

    @testset "isinside" begin
        let dr = DesignInterval(:a => (0, 1), :b => (2, 3)),
            # inside
            dp1 = [0.5, 2.1],
            # on boundary
            dp2 = [1, 2],
            # first outside
            dp3 = [-1, 2.5],
            # both outside
            dp4 = [-1, 4]

            @test Kirstine.isinside(dp1, dr)
            @test Kirstine.isinside(dp2, dr)
            @test !Kirstine.isinside(dp3, dr)
            @test !Kirstine.isinside(dp4, dr)
        end
    end

    @testset "random_designpoint!" begin
        let dr = DesignInterval(:a => (0, 2), :b => (2, 2.5)),
            dp = zeros(2),
            _ = Random.seed!(4711),
            res = Kirstine.random_designpoint!(dp, dr)

            # should return a reference to its first argument
            @test res === dp
            # something should have happened
            @test dp != zeros(2)
            # generated point should be inside
            @test Kirstine.isinside(res, dr)
        end
    end

    @testset "move_desingpoint_how_far" begin
        # 0   1   2   3   4   5   6
        # |---|---|---|---|---|---|
        #    lb       x      ub
        let dr = DesignInterval(:a => (1, 5))

            # stays inside
            @test Kirstine.move_designpoint_how_far([3], [1.5], dr) == 1
            # stop at upper boundary
            @test Kirstine.move_designpoint_how_far([3], [4.0], dr) == 0.5
            @test Kirstine.move_designpoint_how_far([3], [3.0], dr) == 2 / 3

            # stays inside
            @test Kirstine.move_designpoint_how_far([3], [-1.5], dr) == 1
            # stop at lower boundary
            @test Kirstine.move_designpoint_how_far([3], [-4.0], dr) == 0.5
            @test Kirstine.move_designpoint_how_far([3], [-3.0], dr) == 2 / 3
        end
    end

    @testset "move_designpoint!" begin
        # 0   1   2   3   4   5   6
        # |---|---|---|---|---|---|
        #    lb       x      ub
        let dr = DesignInterval(:a => (1, 5)),
            dp1 = [3.0],
            res1 = Kirstine.move_designpoint!(dp1, 1, [1.5], dr),
            dp2 = [3.0],
            res2 = Kirstine.move_designpoint!(dp2, 0.5, [4.0], dr),
            dp3 = [3.0],
            res3 = Kirstine.move_designpoint!(dp3, 2 / 3, [-3.0], dr)

            @test dp1 === res1 # returns reference to first argument
            @test res1 == [4.5]
            @test dp2 === res2
            @test res2 == [5.0]
            @test dp3 === res3
            @test res3 == [1.0]
        end
    end

    @testset "boundingbox" begin
        let dr = DesignInterval(:a => (-2, 3), :b => (5, 7)),
            (lb, ub) = Kirstine.boundingbox(dr)

            @test lb == (-2.0, 5.0)
            @test ub == (3.0, 7.0)
        end
    end
end
end
