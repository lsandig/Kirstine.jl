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

@recipe function f(d::DesignMeasure)
    N = length(designpoints(d)[1])
    if N != 1 && N != 2
        throw(ArgumentError("only implemented for 1- or 2-dimensional design points"))
    end
    legend --> :outerleft
    markersize --> permutedims(max.(2, sqrt.(100 .* weights(d))))
    # scatter each design point explicitly in its own series because grouping can't be used
    # in a recipe: https://github.com/JuliaPlots/Plots.jl/issues/1167
    mat = as_matrix(d)
    for k in 1:size(mat, 2)
        @series begin
            seriestype := :scatter
            markercolor --> k
            label --> k
            y = (N == 1) ? 0 : mat[3, k]
            [mat[2, k]], [y]
        end
    end
end

@recipe function f(d::DesignMeasure, ds::DesignInterval{2})
    lb = lowerbound(ds)
    ub = upperbound(ds)
    xguide --> dimnames(ds)[1]
    yguide --> dimnames(ds)[2]
    xlims --> (lb[1], ub[1])
    ylims --> (lb[2], ub[2])
    widen --> true
    d
end

@recipe function f(d::DesignMeasure, ds::DesignInterval{1})
    lb = lowerbound(ds)
    ub = upperbound(ds)
    xguide --> dimnames(ds)[1]
    xlims --> (lb[1], ub[1])
    widen --> true
    d
end

struct DerivativePlot{
    N,
    Tdc<:DesignCriterion,
    Tds<:DesignSpace{N},
    Tm<:Model,
    Tcp<:CovariateParameterization,
    Tpk<:PriorKnowledge,
    Tt<:Transformation,
    Tna<:NormalApproximation,
}
    d::DesignMeasure
    dp::DesignProblem{Tdc,Tds,Tm,Tcp,Tpk,Tt,Tna}
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
    range_x = range(lowerbound(dp.ds)[1], upperbound(dp.ds)[1]; length = subdivisions)
    dsgpts = collect(Iterators.flatten(designpoints(d)))
    all_x = sort(vcat(range_x, dsgpts))
    directions = [one_point_design([d]) for d in all_x]
    gd = gateauxderivative(d, directions, dp)

    seriestype := :line
    markershape := :none
    label --> ""
    xguide --> dimnames(dp.ds)[1]

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
    subdivisions = (11, 11),
)
    (; d, dp) = dplot
    # calculate gateaux derivative
    lb = lowerbound(dp.ds)
    ub = upperbound(dp.ds)
    range_x = range(lb[1], ub[1]; length = subdivisions[1])
    range_y = range(lb[2], ub[2]; length = subdivisions[2])
    xy_grid = collect(Iterators.product(range_x, range_y))
    directions = [one_point_design([d...]) for d in xy_grid]
    gd = gateauxderivative(d, directions, dp)
    ex = extrema(gd)

    xguide --> (dimnames(dp.ds)[1])
    yguide --> (dimnames(dp.ds)[2])
    # perceptually uniform gradient from blue via white to red
    fillcolor --> :diverging_bwr_55_98_c37_n256
    max_abs_gd = max(abs(ex[1]), abs(ex[2]))
    clims --> ((-1, 1) .* max_abs_gd)
    seriestype := :heatmap
    label := ""

    range_x, range_y, permutedims(gd)
end

"""
    plot_gateauxderivative(d::DesignMeasure, dp::DesignProblem)

Plot the [`gateauxderivative`](@ref) at candidate solution `d` in directions taken from a
grid over design space of the given [`DesignProblem`](@ref), together with the design points of `d`.

Currently only implemented for 1- and 2-dimensional [`DesignInterval`](@ref)s.

The default color gradient for the heat map is `:diverging_bwr_55_98_c37_n256`, which is
perceptually uniform from blue via white to red. Custom gradients can be set via the
`fillcolor` keyword argument. The standard plotting keyword arguments (`markershape`,
`markercolor`, `markersize`, `linecolor`, `linestyle`, `clims`, ...) are supported. By
default, `markersize` indicates the design weights.

# Additional Keyword Arguments

  - `subdivisions::Union{Integer, Tuple{Integer, Integer}}`: number of points in the grid.
    Must match the dimension of the design space.
"""
function plot_gateauxderivative(d::DesignMeasure, dp::DesignProblem; kw...)
    plt_gd = RecipesBase.plot(DerivativePlot(d, dp); kw...)
    return RecipesBase.plot!(plt_gd, d; kw...)
end
