# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignRegionTests
using Test
using Kirstine

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
end
end
