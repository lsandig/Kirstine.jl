# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

@recipe function f(r::OptimizationResult)
    niter = length(r.trace_fx)
    iter = collect(0:(niter - 1))
    xguide --> "iteration"
    yguide --> "objective"
    xlims --> (0, niter)
    label --> ""
    iter, r.trace_fx
end

@recipe function f(r::DirectMaximizationResult)
    # just unwrap the OptimizationResult
    r.or
end

@recipe function f(rs::AbstractVector{<:OptimizationResult})
    nsteps = length(rs)
    steps = collect(1:nsteps)
    fx = [r.maximum for r in rs]
    xguide --> "step"
    yguide --> "best objective value"
    xlims --> (1, nsteps)
    label --> ""
    steps, fx
end

@recipe function f(r::ExchangeResult)
    layout --> @layout [derivative; objective]
    @series begin
        subplot := 1
        yguide --> "max derivative"
        r.ord
    end
    @series begin
        subplot := 2
        r.orw
    end
end

@recipe function f(
    d::DesignMeasure;
    label_formatter = (k, dp, w) -> "$(round(100 * w; sigdigits=3))%",
)
    N = length(designpoints(d)[1])
    if N != 1 && N != 2
        throw(ArgumentError("only implemented for 1- or 2-dimensional design points"))
    end
    markersize --> permutedims(max.(2, sqrt.(100 .* weights(d))))
    # scatter each design point explicitly in its own series because grouping can't be used
    # in a recipe: https://github.com/JuliaPlots/Plots.jl/issues/1167
    mat = as_matrix(d)
    for k in 1:size(mat, 2)
        @series begin
            seriestype := :scatter
            markercolor --> k
            label --> (label_formatter(k, mat[2:end, k], mat[1, k]))
            y = (N == 1) ? 0 : mat[3, k]
            [mat[2, k]], [y]
        end
    end
end

@recipe function f(d::DesignMeasure, dr::DesignInterval{2})
    lb = lowerbound(dr)
    ub = upperbound(dr)
    xguide --> dimnames(dr)[1]
    yguide --> dimnames(dr)[2]
    xlims --> (lb[1], ub[1])
    ylims --> (lb[2], ub[2])
    widen --> true
    d
end

@recipe function f(d::DesignMeasure, dr::DesignInterval{1})
    lb = lowerbound(dr)
    ub = upperbound(dr)
    xguide --> dimnames(dr)[1]
    xlims --> (lb[1], ub[1])
    widen --> true
    d
end

struct DerivativePlot{
    N,
    Tdc<:DesignCriterion,
    Tdr<:DesignRegion{N},
    Tm<:Model,
    Tcp<:CovariateParameterization,
    Tpk<:PriorKnowledge,
    Tt<:Transformation,
    Tna<:NormalApproximation,
}
    d::DesignMeasure
    dp::DesignProblem{Tdc,Tdr,Tm,Tcp,Tpk,Tt,Tna}
end

@recipe function f(
    dplot::DerivativePlot{
        1,
        <:DesignCriterion,
        <:DesignInterval{1},
        <:NonlinearRegression,
        <:CovariateParameterization,
        <:PriorKnowledge,
        <:Transformation,
        <:NormalApproximation,
    };
    subdivisions = 101,
)
    (; d, dp) = dplot
    range_x = range(lowerbound(dp.dr)[1], upperbound(dp.dr)[1]; length = subdivisions)
    dsgpts = collect(Iterators.flatten(designpoints(d)))
    all_x = sort(vcat(range_x, dsgpts))
    directions = [one_point_design([d]) for d in all_x]
    gd = gateauxderivative(d, directions, dp)

    seriestype := :line
    markershape := :none
    label --> ""
    xguide --> dimnames(dp.dr)[1]

    all_x, gd
end

@recipe function f(
    dplot::DerivativePlot{
        2,
        <:DesignCriterion,
        <:DesignInterval{2},
        <:NonlinearRegression,
        <:CovariateParameterization,
        <:PriorKnowledge,
        <:Transformation,
        <:NormalApproximation,
    };
    subdivisions = (51, 51),
)
    (; d, dp) = dplot
    # calculate gateaux derivative
    lb = lowerbound(dp.dr)
    ub = upperbound(dp.dr)
    range_x = range(lb[1], ub[1]; length = subdivisions[1])
    range_y = range(lb[2], ub[2]; length = subdivisions[2])
    xy_grid = collect(Iterators.product(range_x, range_y))
    directions = [one_point_design([d...]) for d in xy_grid]
    gd = gateauxderivative(d, directions, dp)
    ex = extrema(gd)

    xguide --> (dimnames(dp.dr)[1])
    yguide --> (dimnames(dp.dr)[2])
    # perceptually uniform gradient from blue via white to red
    fillcolor --> :diverging_bwr_55_98_c37_n256
    max_abs_gd = max(abs(ex[1]), abs(ex[2]))
    clims --> ((-1, 1) .* max_abs_gd)
    seriestype := :heatmap
    label := ""

    range_x, range_y, permutedims(gd)
end

"""
    plot_gateauxderivative(d::DesignMeasure, dp::DesignProblem; <keyword arguments>)

Plot the [`gateauxderivative`](@ref) in directions from a grid on the design region of `dp`,
overlaid with the design points of `d`.

For 2-dimensional design regions,
the default color gradient for the heat map is `:diverging_bwr_55_98_c37_n256`,
which is perceptually uniform from blue via white to red.
Custom gradients can be set via the `fillcolor` keyword argument.
The standard plotting keyword arguments
(`markershape`, `markercolor`, `markersize`, `linecolor`, `linestyle`, `clims`, ...)
are supported.
By default, `markersize` indicates the design weights.

# Additional Keyword Arguments

  - `subdivisions::Union{Integer, Tuple{Integer, Integer}}`: number of points in the grid.
    Must match the dimension of the design region.
  - `label_formatter::Function`: a function for mapping a triple `(k, pt, w)`
    of an index, design point, and weight to a string for use as a label in the legend.
    Default: weight in percentages rounded to 3 significant digits.

!!! note

    Currently only implemented for 1- and 2-dimensional [`DesignInterval`](@ref)s.
"""
function plot_gateauxderivative(d::DesignMeasure, dp::DesignProblem; kw...)
    plt_gd = RecipesBase.plot(DerivativePlot(d, dp); kw...)
    return RecipesBase.plot!(plt_gd, d; kw...)
end
