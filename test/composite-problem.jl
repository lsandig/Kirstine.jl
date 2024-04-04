# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module CompositeDesignProblemTests
using Test
using Kirstine
using Random: seed!

include("example-emax.jl")
include("example-compartment.jl")

@testset "composite-problem.jl" begin
    @testset "CompositeDesignProblem" begin
        let dc = DCriterion(),
            dr1 = DesignInterval(:dose => (0, 10)),
            dr2 = DesignInterval(:dose => (1, 5)),
            m = EmaxModel(1),
            cp = CopyDose(),
            pk1 = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            pk2 = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 3)]),
            dp1 = DesignProblem(;
                criterion = dc,
                region = dr1,
                model = m,
                covariate_parameterization = cp,
                prior_knowledge = pk1,
            ),
            dp2 = DesignProblem(;
                criterion = dc,
                region = dr2,
                model = m,
                covariate_parameterization = cp,
                prior_knowledge = pk1,
            ),
            dp3 = DesignProblem(;
                criterion = dc,
                region = dr1,
                model = m,
                covariate_parameterization = cp,
                prior_knowledge = pk2,
            )

            @test_throws "must match number" CompositeDesignProblem([dp1, dp2], [0.5])
            @test_throws "must match number" CompositeDesignProblem([dp1], [0.5, 2])
            @test_throws "identical regions" CompositeDesignProblem([dp1, dp2], [0.5, 2])
            @test_throws "non-negative" CompositeDesignProblem([dp1, dp3], [0.5, -1])

            cdp1 = CompositeDesignProblem([dp1, dp3], [0.5, 2])
            # Note: this nested construction implicitly tests region(::CompositeDesignProblem).
            cdp2 = CompositeDesignProblem([dp1, cdp1], [42, 3.14])
            @test_throws "identical regions" CompositeDesignProblem([dp2, cdp1], [42, 3.14])
            @test typeof(problems(cdp1)) <: Vector{<:DesignProblem}
            @test typeof(problems(cdp2)) <: Vector{Kirstine.AbstractDesignProblem}
        end
    end
end
end
