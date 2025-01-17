# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module PriorknowledgeTests
using Test
using Kirstine

include("example-testpar.jl")

@testset "priorknowledge.jl" begin
    @testset "PriorSample" begin
        # error handling in constructors
        let pars = [TestPar2(1, 2), TestPar2(3, 4)]
            @test_throws "must be equal" PriorSample(pars, [0])
            @test_throws "non-negative" PriorSample(pars, [-0.5, 1.5])
            @test_throws "sum to one" PriorSample(pars, [0.5, 1.5])
            @test_throws "same concrete" PriorSample([TestPar2(1, 2), TestPar3(1, 2, 3)])
        end

        # constructor with default uniform weights
        let p = PriorSample([TestPar2(1, 2), TestPar2(3, 4)])
            @test p.weight == [0.5, 0.5]
            @test p.p == [TestPar2(1, 2), TestPar2(3, 4)]

            # round trip via textual representation
            @test p == eval(Meta.parse(repr(p)))
        end
    end

    @testset "==" begin
        let p1 = TestPar2(1, 2),
            p2 = TestPar2(3, 4),
            pk1 = PriorSample([p1, p2]),
            pk2 = PriorSample([p2, p1]),
            pk3 = PriorSample([p1, p2], [0.2, 0.8]),
            pk4 = PriorSample([p1])

            @test pk1 != pk2
            @test pk1 != pk3
            @test pk1 != pk4
            @test pk1 == PriorSample(parameters(pk2)[[2, 1]], weights(pk2))
        end
    end

    @testset "hash" begin
        let p1 = TestPar2(1, 2),
            p2 = TestPar2(3, 4),
            pk1 = PriorSample([p1]),
            pk2 = PriorSample([p2]),
            pk3 = PriorSample([p1])

            # implicitly constructs a Set, which calls hash()
            @test unique([pk1, pk2, pk3]) == [pk1, pk2]
        end
    end

    @testset "weights" begin
        let p = PriorSample([TestPar2(1, 2), TestPar2(3, 4)])

            # check that this returns a reference
            @test weights(p) === p.weight
        end
    end

    @testset "parameters" begin
        let p = PriorSample([TestPar2(1, 2), TestPar2(3, 4)])

            # check that this returns a reference
            @test parameters(p) === p.p
        end
    end
end
end
