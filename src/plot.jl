# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

@recipe function f(r::OptimizationResult)
    niter = length(r.trace_fx)
    xguide --> "iteration"
    yguide --> "objective"
    label --> ""
    1:niter, r.trace_fx
end

@recipe function f(r::DirectMaximizationResult)
    # just unwrap the OptimizationResult
    optimization_result(r)
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
        optimization_results_direction(r)
    end
    @series begin
        subplot := 2
        optimization_results_weight(r)
    end
end

@recipe function f(
    d::DesignMeasure;
    label_formatter = (k, dp, w) -> "$(round(100 * w; sigdigits=3))%",
    minmarkersize = 2,
    maxmarkersize = 10,
)
    N = length(points(d)[1])
    if N != 1 && N != 2
        throw(ArgumentError("only implemented for 1- or 2-dimensional design points"))
    end
    if minmarkersize > maxmarkersize
        throw(ArgumentError("minmarkersize must be <= maxmarkersize"))
    end
    markersize --> permutedims(max.(minmarkersize, maxmarkersize .* sqrt.(weights(d))))
    # scatter each design point explicitly in its own series because grouping can't be used
    # in a recipe: https://github.com/JuliaPlots/Plots.jl/issues/1167
    pt = reduce(hcat, points(d))
    w = weights(d)
    for k in 1:numpoints(d)
        @series begin
            seriestype := :scatter
            markercolor --> k
            label --> (label_formatter(k, pt[:, k], w[k]))
            y = (N == 1) ? 0 : pt[2, k]
            [pt[1, k]], [y]
        end
    end
end

@recipe function f(d::DesignMeasure, dr::DesignRegion{2})
    check_compatible(d, dr)
    lb, ub = boundingbox(dr)
    xguide --> dimnames(dr)[1]
    yguide --> dimnames(dr)[2]
    xlims --> (lb[1], ub[1])
    ylims --> (lb[2], ub[2])
    widen --> true
    d
end

@recipe function f(d::DesignMeasure, dr::DesignRegion{1})
    check_compatible(d, dr)
    lb, ub = boundingbox(dr)
    xguide --> dimnames(dr)[1]
    xlims --> (lb[1], ub[1])
    widen --> true
    d
end

struct DerivativePlot{
    Tdc<:DesignCriterion,
    Tdr<:DesignRegion,
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
        <:DesignCriterion,
        <:DesignRegion{1},
        <:Model,
        <:CovariateParameterization,
        <:PriorKnowledge,
        <:Transformation,
        <:NormalApproximation,
    };
    n_grid = 101,
)
    if length(n_grid) != 1
        throw(ArgumentError("n_grid must be a single integer"))
    end
    if n_grid < 2
        throw(ArgumentError("n_grid must be >= 2"))
    end
    (; d, dp) = dplot
    lb, ub = boundingbox(region(dp))
    range_x = range(lb[1], ub[1]; length = n_grid[1])
    dsgpts = collect(Iterators.flatten(points(d)))
    x_grid = sort(vcat(range_x, dsgpts))
    dp_grid = [[x] for x in x_grid]
    inside_dr = isinside.(dp_grid, Ref(region(dp)))
    directions = [one_point_design(dp) for dp in dp_grid]
    gd_grid = fill(NaN, length(x_grid))
    gd = gateauxderivative(d, directions, dp)
    gd_grid[inside_dr] .= gd

    seriestype := :line
    markershape := :none
    label --> ""
    xguide --> dimnames(region(dp))[1]
    yguide --> "Gateaux derivative"
    linecolor --> :black

    x_grid, gd_grid
end

@recipe function f(
    dplot::DerivativePlot{
        <:DesignCriterion,
        <:DesignRegion{2},
        <:Model,
        <:CovariateParameterization,
        <:PriorKnowledge,
        <:Transformation,
        <:NormalApproximation,
    };
    n_grid = (51, 51),
)
    if length(n_grid) != 2
        throw(ArgumentError("n_grid must be a 2-tuple"))
    end
    if any(n_grid .< 2)
        throw(ArgumentError("all elements of n_grid must be >= 2"))
    end
    (; d, dp) = dplot
    # Calculate Gateaux derivative on a grid over the bounding box of region(dp).
    # Evaluate it only on those points that are inside and fill the rest with NaN,
    # which renders transparently in the heatmap.
    lb, ub = boundingbox(region(dp))
    range_x = range(lb[1], ub[1]; length = n_grid[1])
    range_y = range(lb[2], ub[2]; length = n_grid[2])
    xy_grid = collect.(Iterators.product(range_x, range_y))
    inside_dr = isinside.(xy_grid, Ref(region(dp)))
    gd_grid = fill(NaN, n_grid[1], n_grid[2])
    # note: indexing produces a vector
    directions = [one_point_design([d...]) for d in xy_grid[inside_dr]]
    gd = gateauxderivative(d, directions, dp)
    ex = extrema(gd)
    gd_grid[inside_dr] .= gd

    xguide --> (dimnames(region(dp))[1])
    yguide --> (dimnames(region(dp))[2])
    # perceptually uniform gradient from blue via white to red
    fillcolor --> :diverging_bwr_55_98_c37_n256
    max_abs_gd = max(abs(ex[1]), abs(ex[2]))
    clims --> ((-1, 1) .* max_abs_gd)
    seriestype := :heatmap
    label := ""

    range_x, range_y, permutedims(gd_grid)
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

  - `n_grid::Union{Integer, Tuple{Integer, Integer}}`: number of points in the grid.
    Must match the dimension of the design region.
  - Keyword arguments for plotting `DesignMeasure`s:
    `label_formatter`, `minmarkersize`, `maxmarkersize`

!!! note

    Currently only implemented for 1- and 2-dimensional [`DesignRegion`](@ref)s.
"""
function plot_gateauxderivative(d::DesignMeasure, dp::DesignProblem; kw...)
    plt_gd = RecipesBase.plot(DerivativePlot(d, dp); kw...)
    return RecipesBase.plot!(plt_gd, d; kw...)
end

struct ExpectedFunctionPlot{
    Tf<:Function,
    Tg<:Function,
    Th<:Function,
    Tm<:Model,
    Tcp<:CovariateParameterization,
    Tpk<:PriorKnowledge,
}
    f::Tf
    g::Tg
    h::Th
    x::Vector{Float64}
    d::DesignMeasure
    m::Tm
    cp::Tcp
    pk::Tpk
end

@recipe function f(
    efplot::ExpectedFunctionPlot;
    label_formatter = (k, dp, w) -> "$(round(100 * w; sigdigits=3))%",
    minmarkersize = 2,
    maxmarkersize = 10,
)
    if minmarkersize > maxmarkersize
        throw(ArgumentError("minmarkersize must be <= maxmarkersize"))
    end
    (; f, g, h, x, d, m, cp, pk) = efplot
    K = numpoints(d)
    # graphs
    fx = zeros(length(x), K)
    c = implied_covariates(d, m, cp)
    for k in 1:K
        for i in 1:length(x)
            fx[i, k] = mapreduce(+, weights(pk), parameters(pk)) do w, p
                return w * f(x[i], c[k], p)
            end
        end
    end
    @series begin
        linecolor --> :black
        label --> nothing
        x, unique(fx; dims = 2)
    end
    # points
    pt = points(d)
    wt = weights(d)
    for k in 1:K
        gx = g(c[k])
        hx = mapreduce((w, p) -> w * h(c[k], p), +, weights(pk), parameters(pk))
        @series begin
            label --> (label_formatter(k, pt[k], wt[k]))
            markersize --> max(minmarkersize, maxmarkersize * sqrt(wt[k]))
            markercolor --> k
            seriestype := :scatter
            gx, hx
        end
    end
end

"""
    plot_expected_function(
        f,
        g,
        h,
        xrange::AbstractVector{<:Real},
        d::DesignMeasure,
        m::Model,
        cp::CovariateParameterization,
        pk::PriorSample;
        <keyword arguments>
    )

Plot a graph of the pointwise expected value of a function
and design points on that graph.

For user types `C<:Covariate` and `p<:Parameter`,
the functions `f`, `g`, and `h` should have the following signatures:

  - `f(x::Real, c::C, p::P)::Real`.
    This function is for plotting a graph.
  - `g(c::C)::AbstractVector{<:Real}`.
    This function is for plotting points,
    and should return the x coordinates corresponding to `c`.
  - `h(c::C, p::P)::AbstractVector{<:Real}`.
    This function is for plotting points,
    and should return the y coordinates corresponding to `c`.

For each value of the covariate `c` implied by the design measure `d`,
the following things will be plotted:

  - A graph of the average of `f` with respect to `pk` as a function of `x` over `xrange`.
    If different `c` result in identical graphs, only one will be plotted.

  - Points at `g(c)` and `h(c, p)` averaged with respect to `pk`.
    By default, the `markersize` attribute is used to indicate the weight of the design point.

# Additional Keyword Arguments

  - Keyword arguments for plotting `DesignMeasure`s:
    `label_formatter`, `minmarkersize`, `maxmarkersize`

# Examples

For usage examples see the [tutorial](tutorial.md)
and [dose-time-response](dtr.md) vignettes.
"""
function plot_expected_function(
    f,
    g,
    h,
    xrange::AbstractVector{<:Real},
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample;
    kw...,
)
    efp = ExpectedFunctionPlot(f, g, h, collect(xrange), d, m, cp, pk)
    return RecipesBase.plot(efp; kw...)
end
