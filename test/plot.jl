# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module PlotTests
using Test
using Kirstine
using Plots: plot, xlims, ylims

include("example-emax.jl")
include("example-dtr.jl")

@testset "plot.jl" begin
    @testset "plot" begin
        @test_throws "1- or 2-dimensional" plot(one_point_design([1, 2, 3]))
    end

    @testset "plot_gateauxderivative" begin
        # 1d example
        let p = EmaxPar(; e0 = 0, emax = 1, ec50 = 0.2),
            dr = DesignInterval(:dose => (0, 1)),
            d = emax_solution(p, dr),
            dp = DesignProblem(;
                criterion = DCriterion(),
                region = dr,
                prior_knowledge = PriorSample([p]),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
            ),
            pt1 = plot_gateauxderivative(d, dp),
            pt2 = plot_gateauxderivative(d, dp; n_grid = 11),
            pt3 = plot_gateauxderivative(d, dp; label_formatter = (k, pt, w) -> k),
            pt4 = plot_gateauxderivative(d, dp; overlay_points = false)

            # axes are widened, so they should be strictly smaller
            @test xlims(pt1)[1] < lowerbound(dr)[1]
            @test xlims(pt1)[2] > upperbound(dr)[1]

            # x-axis: grid with 11 point plus the three design points
            @test length(pt2.subplots[1].series_list[1].plotattributes[:x]) == 14

            # check for a few labels
            @test pt1.subplots[1].attr[:xaxis].plotattributes[:guide] == :dose
            @test pt3.subplots[1].series_list[2].plotattributes[:label] == "1"

            # plot/don't plot design points
            # 1 graph + 3 points
            @test length(pt1[1].series_list) == 4
            @test length(pt4[1].series_list) == 1

            @test_throws "here 1" plot_gateauxderivative(d, dp; n_grid = (11, 11))
            @test_throws ".>= 2" plot_gateauxderivative(d, dp; n_grid = -10)

            # we should be able to pass a 1-tuple for n_grid without raising an exception
            @test try
                plot_gateauxderivative(d, dp; n_grid = (11,))
                true
            catch
                false
            end
        end

        # 2d example
        let p = DTRParameter(; a = 1, e = 0.1, e0 = 0, emax = 1, ec50 = 0.2),
            dr = DesignInterval(:dose => (0, 1), :time => (0, 1)),
            d = uniform_design([[0, 0], [1, 0.05], [1, 0.4], [0.2, 0.8], [1, 1]]),
            dp = DesignProblem(;
                criterion = DCriterion(),
                region = dr,
                prior_knowledge = PriorSample([p]),
                model = DTRMod(; sigma = 1, m = 1),
                covariate_parameterization = CopyBoth(),
            ),
            pt1 = plot_gateauxderivative(d, dp; n_grid = (11, 21))

            # check axes (heatmap requires a transposition that is easy to get wrong)
            @test xlims(pt1)[1] < lowerbound(dr)[1]
            @test xlims(pt1)[2] > upperbound(dr)[1]
            @test ylims(pt1)[1] < lowerbound(dr)[2]
            @test ylims(pt1)[2] > upperbound(dr)[2]

            @test length(pt1.subplots[1].series_list[1].plotattributes[:x]) == 11
            @test length(pt1.subplots[1].series_list[1].plotattributes[:y]) == 21

            # check for symmetric color limits
            @test sum(pt1.subplots[1].attr[:clims]) == 0

            # check labels
            @test pt1.subplots[1].attr[:xaxis].plotattributes[:guide] == :dose
            @test pt1.subplots[1].attr[:yaxis].plotattributes[:guide] == :time

            @test_throws "here 2" plot_gateauxderivative(d, dp; n_grid = 11)
            @test_throws ".>= 2" plot_gateauxderivative(d, dp; n_grid = (1, 10))
        end

        # singular design measure with all-nan Gateaux derivative
        let p = DTRParameter(; a = 1, e = 0.1, e0 = 0, emax = 1, ec50 = 0.2),
            dr = DesignInterval(:dose => (0, 1), :time => (0, 1)),
            d = one_point_design([1, 0.05]),
            dp = DesignProblem(;
                criterion = DCriterion(),
                region = dr,
                prior_knowledge = PriorSample([p]),
                model = DTRMod(; sigma = 1, m = 1),
                covariate_parameterization = CopyBoth(),
            )

            @test try
                plot_gateauxderivative(d, dp)
                true
            catch
                false
            end
        end

        # pseudo 3d example, only to test the error message
        let p = EmaxPar(; e0 = 0, emax = 1, ec50 = 0.2),
            dr = DesignInterval(:dose => (0, 1), :foo => (0, 1), :bar => (0, 1)),
            d = emax_solution(p, dr),
            dp = DesignProblem(;
                criterion = DCriterion(),
                region = dr,
                prior_knowledge = PriorSample([p]),
                model = EmaxModel(1),
                covariate_parameterization = CopyDose(),
            )

            @test_throws "got 3" plot_gateauxderivative(d, dp)
        end

        # composite 1d problem
        let dp_helper(cr) = DesignProblem(;
                criterion = cr,
                region = DesignInterval(:dose => (0, 1)),
                prior_knowledge = PriorSample([EmaxPar(; e0 = 0, emax = 1, ec50 = 0.2)]),
                model = EmaxModel(1),
                covariate_parameterization = CopyTo(:dose),
            ),
            dp1 = dp_helper(DCriterion()),
            dp2 = dp_helper(ACriterion()),
            cdp = CompositeDesignProblem([dp1, dp2], [0.5, 0.5]),
            d = equidistant_design(region(cdp), 5)

            @test try
                plot_gateauxderivative(d, cdp)
                true
            catch
                false
            end
        end
    end

    @testset "expected_function_graph_data" begin
        let p1 = DTRParameter(; a = 1, e = 0.1, e0 = 0, emax = 1, ec50 = 0.2),
            p2 = DTRParameter(; a = 0.5, e = 0.2, e0 = 0, emax = 2, ec50 = 0.3),
            pk1 = PriorSample([p1]),
            pk2 = PriorSample([p1, p2]),
            pk3 = PriorSample([p1, p2], [0.2, 0.8]),
            dr = DesignInterval(:dose => (0, 1), :time => (0, 1)),
            mod = DTRMod(; sigma = 1, m = 1),
            cp = CopyBoth(),
            d = uniform_design([[0, 0], [1, 0.05], [1, 0.4], [0.2, 0.8], [1, 1]]),
            x = collect(0:0.2:1),
            f(x, c, p) = response(c.dose, x, p),
            efgd = Kirstine.expected_function_graph_data

            m1, l1, u1 = efgd(f, x, d, mod, cp, pk1, false, 0.8)
            @test size(m1) == (6, 3)
            @test size(l1) == (6, 1)
            @test size(u1) == (6, 1)
            @test all(isnan.(l1))
            @test all(isnan.(u1))

            m2, l2, u2 = efgd(f, x, d, mod, cp, pk2, true, 0.8)
            @test size(m2) == (6, 3)
            @test size(l2) == (6, 3)
            @test size(u2) == (6, 3)
            @test !any(isnan.(l2))
            @test !any(isnan.(u2))

            @test_throws "uniform prior" efgd(f, x, d, mod, cp, pk3, true, 0.8)
            @test_throws "[0, 1]" efgd(f, x, d, mod, cp, pk2, true, 2)
            @test_throws "[0, 1]" efgd(f, x, d, mod, cp, pk2, true, -0.1)
        end
    end

    @testset "expected_function_point_data" begin
        let p1 = DTRParameter(; a = 1, e = 0.1, e0 = 0, emax = 1, ec50 = 0.2),
            p2 = DTRParameter(; a = 0.5, e = 0.2, e0 = 0, emax = 2, ec50 = 0.3),
            pk = PriorSample([p1, p2], [0.1, 0.9]),
            dr = DesignInterval(:dose => (0, 1), :time => (0, 1)),
            mod = DTRMod(; sigma = 1, m = 1),
            cp = CopyBoth(),
            d = uniform_design([[0, 0], [1, 0.05], [1, 0.4], [0.2, 0.8], [1, 1]]),
            g(c) = c.time,
            h(c, p) = [response(c.dose, t, p) for t in c.time],
            efpd = Kirstine.expected_function_point_data

            g1, h1 = efpd(g, h, d, mod, cp, pk)
            @test size(g1) == (1, 5)
            @test size(h1) == (1, 5)
        end
    end
end
end
