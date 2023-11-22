# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module TransformationTests
using Test
using Kirstine

include("example-testpar.jl")

@testset "transformation.jl" begin
    @testset "trafo_constants" begin
        # DeltaMethod
        let pk = PriorSample([TestPar2(1, 2), TestPar2(-1, -2)]),
            dt1 = p -> [p.a; p.b], # too few columns
            D1 = DeltaMethod(dt1),
            dt2 = p -> p.a > 0 ? [p.a p.b] : [p.a p.b; p.a p.b], # different number of rows
            D2 = DeltaMethod(dt2),
            dt3 = p -> [3 * p.a -p.b^2],
            D3 = DeltaMethod(dt3),
            tc = Kirstine.trafo_constants(D3, pk)

            @test_throws "2 columns" Kirstine.trafo_constants(D1, pk)
            @test_throws "identical" Kirstine.trafo_constants(D2, pk)
            @test isa(tc, Kirstine.TCDeltaMethod)
            @test Kirstine.codomain_dimension(tc) == 1
            @test tc.jm == [[3 -4], [-3 -4]]
        end

        # Identity
        let pk = PriorSample([TestPar2(1, 2), TestPar2(-1, -2)]),
            id = Identity(),
            tc = Kirstine.trafo_constants(id, pk)

            @test isa(tc, Kirstine.TCIdentity)
            @test Kirstine.codomain_dimension(tc) == 2
            @test tc.idmat == [1 0; 0 1]
        end
    end
end
end
