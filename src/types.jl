# == abstract supertypes for user-defined models == #

"""
    Model

Abstract supertype for statistical models.
"""
abstract type Model end

"""
    NonlinearRegression

Abstract supertype for user-defined nonlinear regression models.
"""
abstract type NonlinearRegression <: Model end

"""
    Covariate

Abstract supertype for user-defined model covariates.
"""
abstract type Covariate end

"""
    CovariateParameterization

Abstract supertype for user-defined mappings from design points to model covariates.
"""
abstract type CovariateParameterization end

# == package types == #

"""
    PriorKnowledge

Abstract supertype for structs representing prior knowlege of the model parameters.

See also: [`PriorSample`](@ref), [`PriorGuess`](@ref), and [`DiscretePrior`](@ref).
"""
abstract type PriorKnowledge end

"""
    PriorSample(p::AbstractVector{T}) where T

Wraps a vector of samples from a prior distribution, which is used for Bayesian optimal
design.

`T` can be any type for which `length(p::T)` can be interpreted as the dimension of the
parameter space in which `p` lives.
"""
struct PriorSample{T} <: PriorKnowledge
    p::Vector{T}
end

"""
    PriorGuess(p::T) where T

Wraps a single best guess at the unknown parameter value, which is used for locally optimal
design.

`T` can be any type for which `length(p::T)` can be interpreted as the dimension of the
parameter space in which `p` lives.
"""
struct PriorGuess{T} <: PriorKnowledge
    p::T
end

"""
    DiscretePrior(weights, p::AbstractVector{T}) where T

Represents a discrete prior distribution with finite support, which is used for Bayesian
optimal design.

`T` can be any type for which `length(p::T)` can be interpreted as the dimension of the
parameter space in which `p` lives.
"""
struct DiscretePrior{T} <: PriorKnowledge
    weight::Vector{Float64}
    p::Vector{T}
    function DiscretePrior(weights, parameters::AbstractVector{T}) where T
        if length(weights) != length(parameters)
            error("number of weights and parameter values must be equal")
        end
        if any(weights .< 0) || !(sum(weights) ≈ 1)
            error("weights must be non-negative and sum to one")
        end
        return new{T}(weights, parameters)
    end
end

@doc raw"""
    Transformation

Abstract supertype of posterior transformations.

Consider the regression model
``y_i | \theta \sim \mathrm{Normal}(\mu(\theta, x_i), \sigma^2)``
with ``\theta\in\Theta\subset\mathbb{R}^q``.
A `Transformation` representing the function ``T: \Theta\to\mathbb{R}^s``
when we want to maximize a [`DesignCriterion`](@ref) for the posterior distribution of
``T(\theta) \mid y``.
"""
abstract type Transformation end

"""
    Identity

Represents the [`Transformation`](@ref) that maps a parameter to itself.
"""
struct Identity <: Transformation end

"""
    DesignCriterion

Abstract supertype of criteria for optimal experimental design.

See also: [`DOptimality`](@ref)
"""
abstract type DesignCriterion end

@doc raw"""
    DOptimality

Criterion for (Bayesian or locally) D-optimal experimental design.

For the normalized information matrix
``\mathrm{M}(\xi, \theta)``
corresponding to design measure ``\xi``
and an [`Identity`](@ref) transformation for ``\theta``
this is
``\int_{\Theta} \log\det \mathrm{M}(\xi, \theta)\,\mathrm{d}\theta``,
or
``\log\det \mathrm{M}``,
respectively.

See p.69 in Fedorov, V. V., & Leonov, S. L. (2013). [Optimal design for
nonlinear response models](https://doi.org/10.1201/b15054).
"""
struct DOptimality <: DesignCriterion end

abstract type AbstractPoint end

"""
    Optimizer

Abstract supertype for particle-based optimization algorithms.

See also [`Pso`](@ref).
"""
abstract type Optimizer end
abstract type OptimizerState{T<:AbstractPoint} end

"""
    OptimizationResult

Wraps results of particle-based optimization.

# Fields

  - `maximizer`: final maximizer
  - `maximum`: final objective value
  - `trace_x`: vector of current maximizer in each iteration
  - `trace_fx`: vector of current objective value in each iteration
  - `trace_state`: vector of internal optimizer states

Note that `trace_state` may only contain the initial state, if saving all
states was not requested explicitly.

See also [`optimize_design`](@ref).
"""
struct OptimizationResult{T<:AbstractPoint,S<:OptimizerState{T}}
    maximizer::T
    maximum::Float64
    trace_x::Vector{T}
    trace_fx::Vector{Float64}
    trace_state::Vector{S}
end

"""
    DesignMeasure

A discrete probability measure with finite support
representing a continuous experimental design.

See p.62 in Fedorov, V. V., & Leonov, S. L. (2013). [Optimal design for
nonlinear response models](https://doi.org/10.1201/b15054).
"""
struct DesignMeasure <: AbstractPoint
    "the weights of the individual design points"
    weight::Vector{Float64}
    "the design points"
    designpoint::Vector{Vector{Float64}}
    @doc """
        DesignMeasure(w, dp)

    Construct a design measure with the given weights and a vector of design
    points.

    ## Examples
    ```jldoctest
    julia> DesignMeasure([0.5, 0.2, 0.3], [[1, 2], [3, 4], [5, 6]])
    DesignMeasure([0.5, 0.2, 0.3], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    ```
    """
    function DesignMeasure(weight, designpoint)
        if length(weight) != length(designpoint)
            error("number of weights and design points must be equal")
        end
        if !allequal(map(length, designpoint))
            error("design points must have identical lengths")
        end
        if any(weight .< 0) || !(sum(weight) ≈ 1)
            error("weights must be non-negative and sum to one")
        end
        new(weight, designpoint)
    end
end

"""
    DesignSpace{N}

A (hyper)rectangular subset of ``\\mathbb{R}^N`` representing the set in which
which the design points of a `DesignMeasure` live.
The dimensions of a `DesignSpace` are named.
"""
struct DesignSpace{N}
    name::NTuple{N,Symbol}
    lowerbound::NTuple{N,Float64}
    upperbound::NTuple{N,Float64}
    @doc """
        DesignSpace(name, lowerbound, upperbound)

    Construct a design space with the given dimension names (supplied as
    symbols) and lower / upper bounds.

    ## Examples
    ```jldoctest
    julia> DesignSpace([:dose, :time], [0, 0], [300, 20])
    DesignSpace{2}((:dose, :time), (0.0, 0.0), (300.0, 20.0))
    ```
    """
    function DesignSpace(name, lowerbound, upperbound)
        n = length(name)
        if !(n == length(lowerbound) == length(upperbound))
            error("lengths of name, upper and lower bounds must be identical")
        end
        if any(upperbound .<= lowerbound)
            error("upper bounds must be strictly larger than lower bounds")
        end
        new{n}(
            tuple(name...),
            tuple(convert.(Float64, lowerbound)...),
            tuple(convert.(Float64, upperbound)...),
        )
    end
end
