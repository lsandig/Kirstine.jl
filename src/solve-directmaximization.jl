# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## solve design problem by directly maximizing the objective function ##

"""
    DirectMaximization <: ProblemSolvingStrategy

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
            optimizer::Optimizer,
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
    DirectMaximizationResult <: ProblemSolvingResult

Contains an [`OptimizationResult`](@ref) in the `or` field.
"""
struct DirectMaximizationResult{S<:OptimizerState{DesignMeasure,SignedMeasure}} <:
       ProblemSolvingResult
    or::OptimizationResult{DesignMeasure,SignedMeasure,S}
end

function maximizer(dmr::DirectMaximizationResult)
    return dmr.or.maximizer
end

function solve_with(dp::DesignProblem, strategy::DirectMaximization, trace_state::Bool)
    constraints = DesignConstraints(
        strategy.prototype,
        dp.dr,
        strategy.fixedweights,
        strategy.fixedpoints,
    )
    tc = precalculate_trafo_constants(dp.trafo, dp.pk)
    wm = WorkMatrices(unit_length(dp.m), parameter_dimension(dp.pk), codomain_dimension(tc))
    c = allocate_initialize_covariates(strategy.prototype, dp.m, dp.cp)
    f = d -> objective!(wm, c, dp.dc, d, dp.m, dp.cp, dp.pk, tc, dp.na)
    or = optimize(
        strategy.optimizer,
        f,
        [strategy.prototype],
        constraints;
        trace_state = trace_state,
    )
    return DirectMaximizationResult(or)
end
