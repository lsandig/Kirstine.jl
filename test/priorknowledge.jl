# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
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
        end

        # constructor with default uniform weights
        let p = PriorSample([TestPar2(1, 2), TestPar2(3, 4)])
            @test p.weight == [0.5, 0.5]
            @test p.p == [TestPar2(1, 2), TestPar2(3, 4)]
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
