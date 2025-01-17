# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module CompositeDesignProblemTests
using Test
using Kirstine
using Random: seed!

include("test-noalloc.jl")
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

    @testset "==" begin
        let dp1 = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            dp2 = deepcopy(dp1),
            cdp1 = CompositeDesignProblem([dp1, dp2], [1, 2]),
            cdp2 = deepcopy(cdp1)

            @test cdp1 == cdp2
        end
    end

    @testset "hash" begin
        let dp1 = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            dp2 = DesignProblem(dp1; criterion = ACriterion()),
            cdp1 = CompositeDesignProblem([dp1, dp2], [1, 2]),
            cdp2 = CompositeDesignProblem([dp1, dp2], [1, 3]),
            cdp3 = CompositeDesignProblem([dp1, dp2], [1, 3])

            # implicitly constructs a Set, which calls hash()
            @test unique([cdp1, cdp2, cdp3]) == [cdp1, cdp2]
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
            @test eltype(wsp.component_workspaces) <: Kirstine.DPWorkspace
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

    @testset "derivative_constants(::CompositeDesignProblem)" begin
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
            at = equidistant_design(dr, 4)

            dcon = Kirstine.derivative_constants(at, cdp)

            @test dcon isa Kirstine.CDPDerivativeConstants
            @test length(dcon.component_constants) == 2
            @test dcon.component_constants isa
                  Vector{<:Kirstine.DPDerivativeConstants{Kirstine.GCPriorSample}}
        end
    end

    @testset "derivative_constants!" begin
        let dp1 = DesignProblem(;
                criterion = ACriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            dp2 = DesignProblem(dp1; model = EmaxModel(2)),
            cdp = CompositeDesignProblem([dp1, dp2], [1, 1]),
            at = equidistant_design(region(cdp), 4),
            somewhere_else = DesignMeasure([[1], [2], [3], [4]], [0.1, 0.2, 0.3, 0.4]),
            w = Kirstine.allocate_workspace(at, cdp),
            oc = Kirstine.objective_constants(cdp),
            dc = Kirstine.derivative_constants(at, cdp),
            dc_copy = deepcopy(dc)

            Kirstine.derivative_constants!(dc, w, somewhere_else, cdp, oc)

            cc = dc.component_constants
            cc_copy = dc_copy.component_constants

            @test cc[1].gc.A[1] != cc_copy[1].gc.A[1]
            @test cc[1].gc.tr_B[1] != cc_copy[1].gc.tr_B[1]
            @test cc[2].gc.A[1] != cc_copy[2].gc.A[1]
            @test cc[2].gc.tr_B[1] != cc_copy[2].gc.tr_B[1]
        end
    end

    @testset "objective!" begin
        # The correct implementation of the criteria is checked in criterion-{a,d}.jl.
        # Here we test whether there are spurious memory allocations
        # that could point to type instability.
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
            d = equidistant_design(region(dp1), 4),
            ws = Kirstine.allocate_workspace(d, cdp),
            oc = Kirstine.objective_constants(cdp)

            @test_noalloc Kirstine.objective!($ws, $d, $cdp, $oc)
        end
    end

    @testset "gateauxderivative!" begin
        # The correct implementation of the criteria is checked in criterion-{a,d}.jl.
        # Here we test whether there are spurious memory allocations
        # that could point to type instability.
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
            at = equidistant_design(region(dp1), 4),
            dir = one_point_design([0.5]),
            ws = Kirstine.allocate_workspace(dir, cdp),
            dcon = Kirstine.derivative_constants(at, cdp)

            @test_noalloc Kirstine.gateauxderivative!($ws, $dir, $cdp, $dcon)
        end
    end

    @testset "sensitivity!" begin
        # Indirectly check sensitivity function by comparing to the gateaux derivative.
        # Also check for spurious memory allocations.
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
            dp2 = DesignProblem(dp1; model = EmaxModel(2)),
            cdp = CompositeDesignProblem([dp1, dp2], [3, 4]),
            at = equidistant_design(region(dp1), 4),
            dir = one_point_design([0.5]),
            ws = Kirstine.allocate_workspace(dir, cdp),
            dcon = Kirstine.derivative_constants(at, cdp)

            s = Kirstine.sensitivity!(ws, dir, cdp, dcon)
            @test s == only(gateauxderivative(at, [dir], cdp)) + 3 * (3 + 4)
            @test_noalloc Kirstine.gateauxderivative!($ws, $dir, $cdp, $dcon)
        end
    end

    @testset "gateauxderivative(::CompositeDesignProblem)" begin
        # Note: we're essentially testing gateauxderivative!() here.
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
            d = equidistant_design(dr, 4),
            dir = [one_point_design([4.2])]

            # one component is simply a wrapper
            @test gateauxderivative(d, dir, cdp1) ≈ gateauxderivative(d, dir, dp1)
            # two components differing only in prior samples are a Baysian problem
            @test gateauxderivative(d, dir, cdp2) ≈ gateauxderivative(d, dir, dp2)
            # similarly, but nested
            @test gateauxderivative(d, dir, cdp3) ≈ gateauxderivative(d, dir, dp1)
        end
    end

    @testset "simplify(::CompositeDesignProblem)" begin
        let dp_helper(ps) = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 1)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample(ps),
            ),
            p1 = EmaxPar(; e0 = 1, emax = 10, ec50 = 5),
            p2 = EmaxPar(; e0 = 1, emax = 10, ec50 = 3),
            p3 = EmaxPar(; e0 = 1, emax = 10, ec50 = 7),
            cdp = CompositeDesignProblem(
                [dp_helper([p1, p2]), dp_helper([p2, p3])],
                [0.8, 0.2],
            ),
            d = equidistant_design(region(cdp), 4)

            @test_logs(
                (:warn, "uargs [:foo => 42] ignored for a composite design problem"),
                simplify(d, cdp; foo = 42)
            )
        end
    end

    # This is Example 1 from
    # [CW94]: https://dx.doi.org/10.1080/01621459.1994.10476794
    # where the idea is to find an optimal design
    # that balances A-optimal estimation of c in
    #   μ(x, θ) = a + bx + cx^2
    # with D-optimal estimation of a and b in
    #   μ(x, θ) = a + bx
    #
    # To obtain Kirstine.jl's problem weights (w1, w2) from  [CW94]'s lambda,
    # we need to account for the slightly different definitions of the A- and D-criterion in (7) and (8).
    # To match the article,
    # we must multiply Kirstine.jl's A/D-criterion by 0.25 and 0.5, respectively.
    # We move these factors to the problem weights
    #   (w1, w2) = (1/4, 1/2) .* (lambda, 1 - lambda)
    #
    # However, the optimal designs themselves are published in
    # [S82]: https://doi.org/10.2307/2287327
    # where the notation is different.
    # The component designs are not mixed in terms of lambda,
    # but depend on an efficiency value rho,
    # which is called e1 in [CW94].
    # Here, Equation (12) gives the conversion as
    lambda_from_e1(e1) = e1^2 / (4 + 4 * sqrt(1 - e1) - 4 * e1 + e1^2)
    # Hence with
    problem_weights_from_rho(rho) = [lambda_from_e1(rho) / 4, (1 - lambda_from_e1(rho)) / 2]
    # we can specify a problem from Table 1 of [S82] in a way Kirstine.jl understands
    # and then compare the results.
    # All solutions have points {-1, 0, 1} with equal weights on the boundary.
    # The weight at 0 is given by 1-2α₁ in Table 1 from [S82].
    # This number uniquely characterizes the solution.
    # For rho = 0.5 this weight is given as 0.146 (to 3 decimal points).

    # Note: This implementation is somewhat lazy.
    @simple_model Quadratic x
    @simple_model Linear x
    @simple_parameter QPoly a b c
    @simple_parameter LPoly a b
    function Kirstine.jacobianmatrix!(
        jm,
        m::QuadraticModel,
        c::QuadraticCovariate,
        p::QPolyParameter,
    )
        jm[1, 1] = 1
        jm[1, 2] = c.x
        jm[1, 3] = c.x^2
    end
    function Kirstine.jacobianmatrix!(
        jm,
        m::LinearModel,
        c::LinearCovariate,
        p::LPolyParameter,
    )
        jm[1, 1] = 1
        jm[1, 2] = c.x
    end

    @testset "solve(::CompoundDesignProblem)" begin
        let dr = DesignInterval(:x => (-1, 1)),
            cp = JustCopy(:x),
            dp1 = DesignProblem(;
                criterion = ACriterion(),
                model = QuadraticModel(; sigma = 1),
                prior_knowledge = PriorSample([QPolyParameter(; a = 1, b = 1, c = 1)]),
                region = dr,
                covariate_parameterization = cp,
                transformation = DeltaMethod(p -> [0 0 1]),
            ),
            dp2 = DesignProblem(;
                criterion = DCriterion(),
                model = LinearModel(; sigma = 1),
                prior_knowledge = PriorSample([LPolyParameter(; a = 1, b = 1)]),
                region = dr,
                covariate_parameterization = cp,
            ),
            cdp = CompositeDesignProblem([dp1, dp2], problem_weights_from_rho(0.5))

            str = DirectMaximization(;
                optimizer = Pso(; swarmsize = 50, iterations = 50),
                # start with neither optimal points nor weights
                prototype = DesignMeasure([-0.9 0.2 0.8], [0.1, 0.7, 0.2]),
            )
            seed!(135)
            sc, _ = solve(cdp, str)
            opd = one_point_design.(points(sc))

            @test reduce(hcat, points(sc)) ≈ [-1 0 1] rtol = 1e-4
            @test weights(sc)[2] ≈ 0.146 atol = 1e-3
            @test gateauxderivative(sc, opd, cdp) ≈ [0, 0, 0] atol = 1e-4
            @test first(gateauxderivative(sc, [one_point_design([0.5])], cdp)) < 0
        end
    end

    @testset "check_numerically_singular" begin
        let dp = DesignProblem(;
                criterion = DCriterion(),
                region = DesignInterval(:dose => (0, 10)),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 1, emax = 10, ec50 = 5)]),
            ),
            dpa = DesignProblem(dp; criterion = ACriterion()),
            cdp = CompositeDesignProblem([dp, dpa], [3, 4]),
            s = uniform_design([[0], [5], [10]]),
            mix = mixture(1e-14, s, one_point_design([2])),
            cns = Kirstine.check_numerically_singular

            @test_logs cns(s, cdp)
            @test_logs (:warn, r"numerically singular") match_mode = :any cns(mix, cdp)
        end
    end
end
end
