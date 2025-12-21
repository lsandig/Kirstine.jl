# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
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
end

"""
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
    return DirectMaximization{To}(optimizer, prototype, fixedweights, fixedpoints)
end

"""
    Kirstine.DirectMaximizationResult <: Kirstine.ProblemSolvingResult

Wraps the [`OptimizationResult`](@ref) from direct maximization.

See also [`solution(::DirectMaximizationResult)`](@ref), [`optimization_result`](@ref).
"""
struct DirectMaximizationResult{S<:OptimizerState{DesignMeasure,SignedMeasure}} <:
       ProblemSolvingResult
    or::OptimizationResult{DesignMeasure,SignedMeasure,S}
end

"""
    solution(dmr::Kirstine.DirectMaximizationResult)

Return the best candidate found.
"""
solution(dmr::DirectMaximizationResult) = dmr.or.maximizer

"""
    optimization_result(dmr::Kirstine.DirectMaximizationResult)

Get the full [`OptimizationResult`](@ref).
"""
optimization_result(dmr::DirectMaximizationResult) = dmr.or

"""
    total_runtime(dmr::Kirstine.DirectMaximizationResult)

Get the total time spent during optimization.
"""
total_runtime(dmr::DirectMaximizationResult) = dmr.or.seconds_elapsed

function Base.show(io::IO, ::MIME"text/plain", dmr::DirectMaximizationResult)
    println(io, "DirectMaximizationResult")
    println(io, "* final objective: ", dmr.or.maximum)
    println(io, "* runtime total: ", prettydifftime(total_runtime(dmr)))
    println(io, "* raw solution: ")
    show(io, MIME("text/plain"), solution(dmr))
    print(io, "\n* diagnostic information: optimization_result(<this object>)")
end

function solve_with(
    adp::AbstractDesignProblem,
    strategy::DirectMaximization{<:ZerothOrderOptimizer},
    trace_state::Bool,
)
    (; optimizer, prototype, fixedpoints, fixedweights) = strategy
    cstr = DesignConstraints(prototype, region(adp), fixedweights, fixedpoints)
    oc = objective_constants(adp)
    w = allocate_workspace(prototype, adp)
    f = d -> objective!(w, d, adp, oc)
    or = optimize(f, optimizer, [prototype], cstr; trace_state = trace_state)
    return DirectMaximizationResult(or)
end

function solve_with(
    adp::AbstractDesignProblem,
    strategy::DirectMaximization{<:FirstOrderOptimizer},
    trace_state::Bool,
)
    (; optimizer, prototype, fixedpoints, fixedweights) = strategy
    cstr = DesignConstraints(prototype, region(adp), fixedweights, fixedpoints)
    !all(cstr.fixp) && error("only implemented for fixed points")
    oc = objective_constants(adp)
    dc = derivative_constants(prototype, adp)
    opd = one_point_design(first(points(prototype)))
    wf = allocate_workspace(prototype, adp)
    wg = allocate_workspace(opd, adp)
    f = d -> objective!(wf, d, adp, oc)
    g! = (g, d) -> objective_gradient_weights!(g, opd, wg, wf, d, adp, dc, oc)
    or = optimize((f, g!), optimizer, [prototype], cstr; trace_state = trace_state)
    return DirectMaximizationResult(or)
end
