# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## abstract types for user-supplied models ##

"""
    Model

Supertype for statistical models.
"""
abstract type Model end

"""
    NonlinearRegression

Supertype for user-defined nonlinear regression models.
"""
abstract type NonlinearRegression <: Model end

"""
    Parameter

Supertype for [`Model`](@ref) parameters.
"""
abstract type Parameter end

"""
    Covariate

Supertype for user-defined model covariates.

# Implementation

Subtypes of `Covariate` should be mutable, or have at least a mutable field. This is
necessary for being able to modify them in [`update_model_covariate!`](@ref).
"""
abstract type Covariate end

"""
    CovariateParameterization

Supertype for user-defined mappings from design points to model covariates.
"""
abstract type CovariateParameterization end

## abstract types for package code ##

"""
    DesignRegion{N}

Supertype for design regions. A design region is a compact subset of ``\\Reals^N``.

The design points of a [`DesignMeasure`](@ref) are taken from this set.

See also [`DesignInterval`](@ref), [`dimnames`](@ref).
"""
abstract type DesignRegion{N} end

"""
    PriorKnowledge{T<:Parameter}

Supertype for representing prior knowledge of the model [`Parameter`](@ref).

See also [`PriorSample`](@ref).
"""
abstract type PriorKnowledge{T<:Parameter} end

"""
    NormalApproximation

Supertype for different possible normal approximations to the posterior distribution.
"""
abstract type NormalApproximation end

@doc raw"""
    Transformation

Supertype of posterior transformations.

See also [`Identity`](@ref), [`DeltaMethod`](@ref),
and the [mathematical background](math.md#Objective-Function).
"""
abstract type Transformation end

"""
    DesignCriterion

Supertype for optimal experimental design criteria.

See also [`DOptimality`](@ref), [`AOptimality`](@ref).
"""
abstract type DesignCriterion end

## helper types for precomputed constants ##

abstract type TrafoConstants end

abstract type GateauxConstants end

## types related to particle-based optimization ##

abstract type AbstractPoint end
abstract type AbstractPointDifference end

"""
    Optimizer

Supertype for particle-based optimization algorithms.

See also [`Pso`](@ref).
"""
abstract type Optimizer end
abstract type OptimizerState{T<:AbstractPoint,U<:AbstractPointDifference} end

abstract type AbstractConstraints end

## types related to solving design problems

"""
    ProblemSolvingStrategy

Supertype for algorithms for solving a [`DesignProblem`](@ref).
"""
abstract type ProblemSolvingStrategy end

"""
    ProblemSolvingResult

Supertype for results of a [`ProblemSolvingStrategy`](@ref).
"""
abstract type ProblemSolvingResult end
