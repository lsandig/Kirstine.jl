# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## solve design problem by directly maximizing the objective function ##

"""
    DirectMaximization <: Kirstine.ProblemSolvingStrategy

Find an optimal design by directly maximizing a criterion's objective function.
"""
struct DirectMaximization{To<:Optimizer} <: ProblemSolvingStrategy
    optimizer::To
    prototype::DesignMeasure
    fixedweights::Vector{Int64}
    fixedpoints::Vector{Int64}
    # Note: We don't use @kwdef because fixed{weights,points} should be able to accept an
    # AbstractVector, and esp. also a unit range
    @doc """
        DirectMaximization(;
            optimizer::Kirstine.Optimizer,
            prototype::DesignMeasure,
            fixedweights = [],
            fixedpoints = [],
        )

    Initialize the `optimizer` with a single `prototype`
    and attempt to directly maximize the objective function.

    For every index in `fixedweights`,
    the corresponding weight of `prototype` is held constant during optimization.
    The indices in `fixedpoints` do the same for the design points of the `prototype`.
    These additional constraints can be used to speed up computation
    in cases where some weights or design points are know analytically.

    For more details on how the `prototype` is used,
    see the specific [`Optimizer`](@ref)s.

    The return value of [`solve`](@ref) for this strategy is a [`DirectMaximizationResult`](@ref).
    """
    function DirectMaximization(;
        optimizer::To,
        prototype::DesignMeasure,
        fixedweights::AbstractVector{<:Integer} = Int64[],
        fixedpoints::AbstractVector{<:Integer} = Int64[],
    ) where To<:Optimizer
        new{To}(optimizer, prototype, fixedweights, fixedpoints)
    end
end

"""
    DirectMaximizationResult <: Kirstine.ProblemSolvingResult

Wraps the [`OptimizationResult`](@ref) from direct maximization.

See also [`solution(::DirectMaximizationResult)`](@ref), [`optimization_result`](@ref).
"""
struct DirectMaximizationResult{S<:OptimizerState{DesignMeasure,SignedMeasure}} <:
       ProblemSolvingResult
    or::OptimizationResult{DesignMeasure,SignedMeasure,S}
end

"""
    solution(dmr::DirectMaximizationResult)

Return the best candidate found.
"""
function solution(dmr::DirectMaximizationResult)
    return dmr.or.maximizer
end

"""
    optimization_result(dmr::DirectMaximizationResult)

Get the full [`OptimizationResult`](@ref).
"""
function optimization_result(dmr::DirectMaximizationResult)
    return dmr.or
end

function solve_with(
    adp::AbstractDesignProblem,
    strategy::DirectMaximization{<:ZerothOrderOptimizer},
    trace_state::Bool,
)
    constraints = DesignConstraints(
        strategy.prototype,
        region(adp),
        strategy.fixedweights,
        strategy.fixedpoints,
    )
    oc = objective_constants(adp)
    w = allocate_workspace(strategy.prototype, adp)
    or = optimize(
        d -> objective!(w, d, adp, oc),
        strategy.optimizer,
        [strategy.prototype],
        constraints;
        trace_state = trace_state,
    )
    return DirectMaximizationResult(or)
end

function solve_with(
    adp::AbstractDesignProblem,
    strategy::DirectMaximization{<:FirstOrderOptimizer},
    trace_state::Bool,
)
    (; optimizer, prototype, fixedpoints, fixedweights) = strategy
    K = numpoints(prototype)
    constraints = DesignConstraints(prototype, region(adp), fixedweights, fixedpoints)
    !all(constraints.fixp) && error("only implemented for fixed points")
    oc = objective_constants(adp)
    dc = derivative_constants(prototype, adp)
    opd = one_point_design(first(points(prototype)))
    wf = allocate_workspace(prototype, adp)
    wg = allocate_workspace(opd, adp)
    f = d -> objective!(wf, d, adp, oc)
    g! = (g, d) -> objective_gradient_weights!(g, opd, wg, wf, d, adp, dc, oc)
    or = optimize(f, g!, optimizer, [prototype], constraints; trace_state = trace_state)
    return DirectMaximizationResult(or)
end
