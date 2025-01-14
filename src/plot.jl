# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

@recipe function f(r::OptimizationResult)
    niter = length(r.trace_fx)
    seriestype --> (niter == 1 ? :scatter : :line)
    xticks --> (niter == 1 ? [1] : :auto)
    xlims --> (niter == 1 ? (0.5, 1.5) : :auto)
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
    seriestype --> (nsteps == 1 ? :scatter : :line)
    xticks --> (nsteps == 1 ? [1] : :auto)
    xlims --> (nsteps == 1 ? (0.5, 1.5) : (1, nsteps))
    fx = [r.maximum for r in rs]
    xguide --> "step"
    yguide --> "best objective value"
    label --> ""
    steps, fx
end

@recipe function f(r::ExchangeResult)
    layout --> @layout [derivative; objective]
    @series begin
        subplot := 1
        yguide --> "max derivative"
        ylims --> (0, NaN)
        widen --> true
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

struct DerivativePlot{N,Tdr<:DesignRegion{N}}
    dr::Tdr
    marginal_grids::NTuple{N,Vector{Float64}}
    gd::Array{Float64,N}
    function DerivativePlot(
        d::DesignMeasure,
        adp::AbstractDesignProblem,
        n_grid::NTuple{N,Integer},
    ) where N
        if any(n_grid .< 2)
            throw(ArgumentError("n_grid must be .>= 2"))
        end
        dr = region(adp)
        # Calculate Gateaux derivative on a grid over the bounding box of dr.
        # Evaluate it only on those points that are inside and fill the rest with NaN,
        # which renders transparently in the heatmap.
        mg = marginal_grids(dr, n_grid)
        if N == 1
            append!(mg[1], Iterators.flatten(points(d)))
            sort!(mg[1])
        end
        product_dp_grid = collect.(Iterators.product(mg...))
        inside_dr = isinside.(product_dp_grid, Ref(dr))
        gd_grid = fill(NaN, size(product_dp_grid))
        # note: indexing and array with another array produces a vector
        directions = [one_point_design(dp) for dp in product_dp_grid[inside_dr]]
        gd = gateauxderivative(d, directions, adp)
        gd_grid[inside_dr] .= gd
        return new{N,typeof(dr)}(dr, mg, gd_grid)
    end
end

@recipe function f(dplot::DerivativePlot{1,<:DesignRegion{1}})
    seriestype := :line
    markershape := :none
    label --> ""
    xguide --> dimnames(dplot.dr)[1]
    yguide --> "Gateaux derivative"
    linecolor --> :black

    dplot.marginal_grids[1], dplot.gd
end

@recipe function f(dplot::DerivativePlot{2,<:DesignRegion{2}})
    gd_not_nan = filter(!isnan, dplot.gd)
    ex = isempty(gd_not_nan) ? (-1.0, 1.0) : extrema(gd_not_nan)
    xguide --> (dimnames(dplot.dr)[1])
    yguide --> (dimnames(dplot.dr)[2])
    # perceptually uniform gradient from blue via white to red
    fillcolor --> :diverging_bwr_55_98_c37_n256
    max_abs_gd = max(abs(ex[1]), abs(ex[2]))
    clims --> ((-1, 1) .* max_abs_gd)
    seriestype := :heatmap
    label := ""

    dplot.marginal_grids[1], dplot.marginal_grids[2], permutedims(dplot.gd)
end

"""
    plot_gateauxderivative(d::DesignMeasure, adp::Kirstine.AbstractDesignProblem; <keyword arguments>)

Plot the [`gateauxderivative`](@ref) in directions from a grid on `region(adp)`,
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

  - `n_grid::Union{Integer, Tuple{Integer}, Tuple{Integer, Integer}}`:
    number of points in the grid.
    Must match the dimension of the design region.
    Defaults to `101` for one dimension, and to `(51, 51)` for two.
  - `overlay_points::Bool`: draw also `points(d)` on top of the graph.
    Defaults to `true`.
  - Keyword arguments for plotting `DesignMeasure`s:
    `label_formatter`, `minmarkersize`, `maxmarkersize`.
    See the [API reference](api.md#Plotting) for what they do.

!!! note

    Currently only implemented for 1- and 2-dimensional [`DesignRegion`](@ref)s.
"""
function plot_gateauxderivative(
    d::DesignMeasure,
    adp::AbstractDesignProblem;
    n_grid = missing,
    overlay_points::Bool = true,
    kw...,
)
    N = dimension(region(adp))
    if ismissing(n_grid)
        if N == 1
            n_grid = 101
        elseif N == 2
            n_grid = (51, 51)
        else
            msg = "only implemented for 1 or 2 dimensional regions, got $N"
            throw(ArgumentError(msg))
        end
    end
    if length(n_grid) != N
        msg = "length(n_grid) must match dimension of design region, here $N"
        throw(ArgumentError(msg))
    end
    n_grid_tuple = n_grid isa Integer ? (n_grid,) : n_grid
    plt_gd = RecipesBase.plot(DerivativePlot(d, adp, n_grid_tuple); kw...)
    if overlay_points
        RecipesBase.plot!(plt_gd, d; kw...)
    end
    return plt_gd
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

function expected_function_graph_data(
    f::Function,
    x::AbstractVector{<:Real},
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    bands::Bool,
    probability::Real,
)
    if bands && any(weights(pk) .!= first(weights(pk)))
        msg = "credibility bands are only implemented for uniform prior weights"
        throw(ArgumentError(msg))
    elseif probability < 0 || probability > 1
        throw(ArgumentError("probability must be in in interval [0, 1]"))
    end
    K = numpoints(d)
    num_draws = length(weights(pk))
    fx = zeros(num_draws)
    fxm = zeros(length(x), K)
    fxl = fill(NaN, length(x), K)
    fxu = fill(NaN, length(x), K)
    c = implied_covariates(d, m, cp)
    qprobs = ((1 - probability) / 2, (1 + probability) / 2)
    pars = parameters(pk)
    for k in 1:K
        for i in 1:length(x)
            for s in 1:num_draws
                fx[s] = f(x[i], c[k], pars[s])
            end
            fxm[i, k] = mapreduce((w, fxs) -> w * fxs, +, weights(pk), fx)
            if bands
                fxl[i, k], fxu[i, k] = quantile(fx, qprobs)
            end
        end
    end
    unique_fxm = unique(fxm; dims = 2)
    # lower/upper bounds should be unique as _pairs_
    unique_pairs = unique(vcat(fxl, fxu); dims = 2)
    unique_fxl = unique_pairs[1:length(x), :]
    unique_fxu = unique_pairs[(length(x) + 1):end, :]
    return unique_fxm, unique_fxl, unique_fxu
end

function expected_function_point_data(
    g::Function,
    h::Function,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
)
    c = implied_covariates(d, m, cp)
    gx = mapreduce(g, hcat, c)
    hx = similar(gx)
    for k in 1:numpoints(d)
        hx[:, k] .= mapreduce((w, p) -> w * h(c[k], p), +, weights(pk), parameters(pk))
    end
    return gx, hx
end

@recipe function f(
    efplot::ExpectedFunctionPlot;
    label_formatter = (k, dp, w) -> "$(round(100 * w; sigdigits=3))%",
    minmarkersize = 2,
    maxmarkersize = 10,
    bands = false,
    probability = 0.8,
)
    (; f, g, h, x, d, m, cp, pk) = efplot
    if minmarkersize > maxmarkersize
        throw(ArgumentError("minmarkersize must be <= maxmarkersize"))
    end
    fxm, fxl, fxu = expected_function_graph_data(f, x, d, m, cp, pk, bands, probability)
    if bands
        @series begin
            label --> nothing
            fillcolor --> :black
            fillalpha --> 0.1
            linecolor := nothing
            fillrange := fxl
            x, fxu
        end
    end
    # graphs
    @series begin
        linecolor --> :black
        label --> nothing
        x, fxm
    end
    # points
    pt = points(d)
    wt = weights(d)
    gx, hx = expected_function_point_data(g, h, d, m, cp, pk)
    for k in 1:numpoints(d)
        @series begin
            label --> (label_formatter(k, pt[k], wt[k]))
            # TODO: refactor into separate function?
            markersize --> max(minmarkersize, maxmarkersize * sqrt(wt[k]))
            markercolor --> k
            seriestype := :scatter
            gx[:, k], hx[:, k]
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
        m::Kirstine.Model,
        cp::CovariateParameterization,
        pk::PriorSample;
        <keyword arguments>
    )

Plot a graph of the pointwise expected value of a function
and design points on that graph.

For user types `C<:Covariate` and `p<:Kirstine.Parameter`,
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

  - `bands::Bool=false`: plot point-wise credible bands.
    Currently only implemented for uniform `weights(d)`.
    Color and opacity can be controlled with the standard keyword arguments
    `fillcolor` and `fillalpha`.
  - `probability::Real=0.8`: probability to use for the credible band.
  - Keyword arguments for plotting `DesignMeasure`s:
    `label_formatter`, `minmarkersize`, `maxmarkersize`.
    See the [API reference](api.md#Plotting) for what they do.

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
