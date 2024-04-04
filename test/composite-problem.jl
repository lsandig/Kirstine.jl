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

    @testset "allocate_workspace(::CompositeDesignProblem)" begin
        let dc = DCriterion(),
            dr = DesignInterval(:dose => (0, 10)),
            cp = CopyDose(),
            pk = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            dp1 = DesignProblem(;
                criterion = dc,
                region = dr,
                model = EmaxModel(1),
                covariate_parameterization = cp,
                prior_knowledge = pk,
            ),
            dp2 = DesignProblem(;
                criterion = dc,
                region = dr,
                model = EmaxModel(2),
                covariate_parameterization = cp,
                prior_knowledge = pk,
            ),
            cdp = CompositeDesignProblem([dp1, dp2], [1, 1]),
            d = equidistant_design(region(cdp), 4)

            wsp = Kirstine.allocate_workspace(d, cdp)

            @test wsp isa Kirstine.CDPWorkspace
            @test length(wsp.component_workspaces) == 2
            @test eltype(wsp.component_workspaces) == Kirstine.DPWorkspace
        end
    end

    @testset "objective_constants(::CompositeDesignProblem)" begin
        let dc = DCriterion(),
            dr = DesignInterval(:dose => (0, 10)),
            cp = CopyDose(),
            pk = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            dp1 = DesignProblem(;
                criterion = dc,
                region = dr,
                model = EmaxModel(1),
                covariate_parameterization = cp,
                prior_knowledge = pk,
            ),
            dp2 = DesignProblem(;
                criterion = dc,
                region = dr,
                model = EmaxModel(2),
                covariate_parameterization = cp,
                prior_knowledge = pk,
            ),
            cdp = CompositeDesignProblem([dp1, dp2], [1, 1])

            oc = Kirstine.objective_constants(cdp)

            @test oc isa Kirstine.CDPObjectiveConstants
            @test length(oc.component_constants) == 2
        end
    end

    @testset "objective(::CompositeDesignProblem)" begin
        # Note: here we essentially test objective!(),
        # but don't want so set up constants and workspace manually.
        let dc = DCriterion(),
            dr = DesignInterval(:dose => (0, 10)),
            m = EmaxModel(1),
            cp = CopyDose(),
            p1 = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            p2 = EmaxPar(; e0 = 1, emax = 10, ec50 = 3),
            p3 = EmaxPar(; e0 = 1, emax = 10, ec50 = 7),
            dp_helper(ps, ws) = DesignProblem(;
                criterion = dc,
                region = dr,
                model = m,
                covariate_parameterization = cp,
                prior_knowledge = PriorSample(ps, ws),
            ),
            dp1 = dp_helper([p1, p2, p3], [0.1, 0.3, 0.6]),
            dp2 = dp_helper([p2, p3], [1 / 3, 2 / 3]),
            cdp1 = CompositeDesignProblem([dp1], [1]),
            cdp2 = CompositeDesignProblem(
                [dp_helper([p2], [1]), dp_helper([p3], [1])],
                [1 / 3, 2 / 3],
            ),
            cdp3 = CompositeDesignProblem([dp_helper([p1], [1]), cdp2], [0.1, 0.9]),
            d = equidistant_design(dr, 4)

            # one component is simply a wrapper
            @test objective(d, cdp1) == objective(d, dp1)
            # two components differing only in prior samples are a Baysian problem
            @test objective(d, cdp2) == objective(d, dp2)
            # similarly, but nested
            @test objective(d, cdp3) == objective(d, dp1)
        end
    end
end
end
