# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
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
            pt3 = plot_gateauxderivative(d, dp; label_formatter = (k, pt, w) -> k)

            # axes are widened, so they should be strictly smaller
            @test xlims(pt1)[1] < lowerbound(dr)[1]
            @test xlims(pt1)[2] > upperbound(dr)[1]

            # x-axis: grid with 11 point plus the three design points
            @test length(pt2.subplots[1].series_list[1].plotattributes[:x]) == 14

            # check for a few labels
            @test pt1.subplots[1].attr[:xaxis].plotattributes[:guide] == :dose
            @test pt3.subplots[1].series_list[2].plotattributes[:label] == "1"

            @test_throws "single" plot_gateauxderivative(d, dp; n_grid = (11, 11))
            @test_throws ">= 2" plot_gateauxderivative(d, dp; n_grid = -10)
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

            @test_throws "2-tuple" plot_gateauxderivative(d, dp; n_grid = 11)
            @test_throws ">= 2" plot_gateauxderivative(d, dp; n_grid = (1, 10))
        end
    end
end
end
