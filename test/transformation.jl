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
            tc = Kirstine.trafo_constants(D3, pk),
            D4 = DeltaMethod(p -> [1 2; 3 4; 5 6]),
            D5 = DeltaMethod(p -> [p.a 2*p.b; 3*p.a 6*p.b])

            @test_throws "2 columns" Kirstine.trafo_constants(D1, pk)
            @test_throws "identical" Kirstine.trafo_constants(D2, pk)
            @test_warn "more rows than cols" Kirstine.trafo_constants(D4, pk)
            @test_warn "not full rank" Kirstine.trafo_constants(D5, pk)
            @test isa(tc, Kirstine.TCDeltaMethod)
            @test tc.jm == [[3 -4], [-3 -4]]
        end

        # Identity
        let pk = PriorSample([TestPar2(1, 2), TestPar2(-1, -2)]),
            id = Identity(),
            tc = Kirstine.trafo_constants(id, pk)

            @test isa(tc, Kirstine.TCIdentity)
            @test tc.idmat == [1 0; 0 1]
        end
    end

    @testset "codomain_dimension(Identity)" begin
        let pk = PriorSample([TestPar2(1, 2)]), trafo = Identity()
            @test Kirstine.codomain_dimension(trafo, pk) == 2
        end
    end

    @testset "codomain_dimension(DeltaMethod)" begin
        let pk = PriorSample([TestPar3(1, 2, 3)]), trafo = DeltaMethod(p -> [1 0 0; 0 0 1])
            @test Kirstine.codomain_dimension(trafo, pk) == 2
        end
    end
end
end
