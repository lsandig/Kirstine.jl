# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

@recipe function f(r::OptimizationResult)
    niter = length(r.trace_fx)
    seriestype --> (niter == 1 ? :scatter : :line)
    xticks --> (niter == 1 ? [1] : :auto)
    xlims --> (niter == 1 ? (0.5, 1.5) : :auto)
    xguide --> "iteration"
    yguide --> "objective"
    label --> ""
    return 1:niter, r.trace_fx
end

@recipe function f(r::DirectMaximizationResult)
    # just unwrap the OptimizationResult
    return optimization_result(r)
end

@recipe function f(rs::AbstractVector{<:OptimizationResult})
    nsteps = length(rs)
    steps = collect(1:nsteps)
    seriestype --> (nsteps == 1 ? :scatter : :line)
    xticks --> steps
    xlims --> (nsteps == 1 ? (0.5, 1.5) : (1, nsteps))
    fx = [r.maximum for r in rs]
    xguide --> "step"
    yguide --> "best objective value"
    label --> ""
    return steps, fx
end

@recipe function f(r::ExchangeResult)
    layout --> @layout [panel_objective, panel_derivative]
    @series begin
        subplot := 1
        widen --> true
        optimization_results_weight(r)
    end
    @series begin
        subplot := 2
        yguide --> "max derivative"
        ylims --> (0, NaN)
        widen --> true
        optimization_results_direction(r)
    end
end

@recipe function f(
    d::DesignMeasure;
    label_formatter = (k, dp, w) -> "$(round(100 * w; sigdigits=3))%",
    minmarkersize = 2,
    maxmarkersize = 10,
)
    N = length(points(d)[1])
    err_dim = ArgumentError("only implemented for 1- or 2-dimensional design points")
    N == 1 || N == 2 || throw(err_dim)
    err_markersize = ArgumentError("minmarkersize must be <= maxmarkersize")
    minmarkersize <= maxmarkersize || throw(err_markersize)
    markersize --> permutedims(max.(minmarkersize, maxmarkersize .* sqrt.(weights(d))))
    # scatter each design point explicitly in its own series because grouping can't be used
    # in a recipe: https://github.com/JuliaPlots/Plots.jl/issues/1167
    pt = reduce(hcat, points(d))
    for k in 1:numpoints(d)
        @series begin
            seriestype := :scatter
            markercolor --> k
            label --> (label_formatter(k, pt[:, k], weights(d)[k]))
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
    return d
end

@recipe function f(d::DesignMeasure, dr::DesignRegion{1})
    check_compatible(d, dr)
    lb, ub = boundingbox(dr)
    xguide --> dimnames(dr)[1]
    xlims --> (lb[1], ub[1])
    widen --> true
    return d
end

struct DerivativePlot{N,Tdr<:DesignRegion{N}}
    dr::Tdr
    mg::NTuple{N,Vector{Float64}}
    gd::Array{Float64,N}
end

function DerivativePlot(
    d::DesignMeasure,
    adp::AbstractDesignProblem,
    n_grid::NTuple{N,Integer},
) where N
    mg, gd_grid = gd_grid_helper(d, adp, n_grid; include_1d_support = true)
    return DerivativePlot(region(adp), mg, gd_grid)
end

@recipe function f(dplot::DerivativePlot{1,<:DesignRegion{1}})
    gd_not_nan = filter(!isnan, dplot.gd)
    gd_max = isempty(gd_not_nan) ? NaN : maximum(gd_not_nan)
    seriestype := :line
    markershape := :none
    label --> ""
    xguide --> dimnames(dplot.dr)[1]
    yguide --> "Gateaux derivative"
    linecolor --> :black
    title --> "maximum: $(round(gd_max; sigdigits=3))"
    return dplot.mg[1], dplot.gd
end

@recipe function f(dplot::DerivativePlot{2,<:DesignRegion{2}})
    # Note that NaN values in gd will renders transparently in the heatmap.
    gd_not_nan = filter(!isnan, dplot.gd)
    gd_max = isempty(gd_not_nan) ? NaN : maximum(gd_not_nan)
    ex = isempty(gd_not_nan) ? (-1.0, 1.0) : extrema(gd_not_nan)
    xguide --> (dimnames(dplot.dr)[1])
    yguide --> (dimnames(dplot.dr)[2])
    # perceptually uniform gradient from blue via white to red
    fillcolor --> :diverging_bwr_55_98_c37_n256
    max_abs_gd = max(abs(ex[1]), abs(ex[2]))
    clims --> ((-1, 1) .* max_abs_gd)
    seriestype := :heatmap
    label := ""
    colorbar_title --> "Gateaux derivative"
    title --> "maximum: $(round(gd_max; sigdigits=3))"
    return dplot.mg[1], dplot.mg[2], permutedims(dplot.gd)
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
        msg = "only implemented for 1 or 2 dimensional regions, got $N"
        N in [1, 2] || throw(ArgumentError(msg))
        n_grid = N == 1 ? 101 : (51, 51)
    end
    n_grid_tuple = n_grid isa Integer ? (n_grid,) : n_grid
    plt_gd = RecipesBase.plot(DerivativePlot(d, adp, n_grid_tuple); kw...)
    overlay_points && RecipesBase.plot!(plt_gd, d; kw...)
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
    msg_weight = "credibility bands are only implemented for uniform prior weights"
    !bands || allequal(weights(pk)) || throw(ArgumentError(msg_weight))
    err_prob = ArgumentError("probability must be in in interval [0, 1]")
    (0 <= probability <= 1) || throw(err_prob)
    num_draws = length(weights(pk))
    fx = zeros(num_draws)
    fxm = zeros(length(x), numpoints(d))
    fxl = fill(NaN, length(x), numpoints(d))
    fxu = fill(NaN, length(x), numpoints(d))
    c = implied_covariates(d, m, cp)
    qprobs = ((1 - probability) / 2, (1 + probability) / 2)
    for k in 1:numpoints(d)
        for i in 1:length(x)
            for s in 1:num_draws
                fx[s] = f(x[i], c[k], parameters(pk)[s])
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
    err_markersize = ArgumentError("minmarkersize must be <= maxmarkersize")
    minmarkersize <= maxmarkersize || throw(err_markersize)
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
    gx, hx = expected_function_point_data(g, h, d, m, cp, pk)
    for k in 1:numpoints(d)
        @series begin
            label --> (label_formatter(k, points(d)[k], weights(d)[k]))
            markersize --> max(minmarkersize, maxmarkersize * sqrt(weights(d)[k]))
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
        xgrid::AbstractVector{<:Real},
        d::DesignMeasure,
        m::Kirstine.Model,
        cp::Kirstine.CovariateParameterization,
        pk::PriorSample;
        <keyword arguments>
    )

Plot a graph of the pointwise expected value of a function
and design points on that graph.

For user types `C<:Kirstine.Covariate` and `p<:Kirstine.Parameter`,
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

  - A graph of the average of `f` with respect to `pk` as a function of `x`
    evaluated at the elements of `xgrid`.
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
    xgrid::AbstractVector{<:Real},
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample;
    kw...,
)
    efp = ExpectedFunctionPlot(f, g, h, collect(xgrid), d, m, cp, pk)
    return RecipesBase.plot(efp; kw...)
end
