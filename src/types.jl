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
    Parameter

Supertype for [`Model`](@ref) parameters.

A user-defined subtype `P` should have a `dimension(p::P)` method
which returns the dimension of the associated parameter space.
"""
abstract type Parameter end

"""
    PriorKnowledge{T<:Parameter}

Abstract supertype for structs representing prior knowledge of the model [`Parameter`](@ref).

See also [`DiscretePrior`](@ref).
"""
abstract type PriorKnowledge{T<:Parameter} end

"""
    DiscretePrior([weights,] p::AbstractVector{<:Parameter})

Represents a sample from a prior distribution, or a discrete prior distribution with finite
support.

If no `weights` are given, a uniform distribution on the elements of `p` is assumed.
"""
struct DiscretePrior{T} <: PriorKnowledge{T}
    weight::Vector{Float64}
    p::Vector{T}
    function DiscretePrior(weights, parameters::AbstractVector{T}) where T<:Parameter
        if length(weights) != length(parameters)
            error("number of weights and parameter values must be equal")
        end
        if any(weights .< 0) || !(sum(weights) ≈ 1)
            error("weights must be non-negative and sum to one")
        end
        return new{T}(weights, parameters)
    end
end

function DiscretePrior(p::AbstractVector{<:Parameter})
    n = length(p)
    return DiscretePrior(fill(1 / n, n), p)
end

"""
    DiscretePrior(p::Parameter)

Construct a one-point (Dirac) prior distribution at `p`.
"""
function DiscretePrior(p::Parameter)
    return DiscretePrior([1.0], [p])
end

"""
    NormalApproximation

Abstract supertype for different possible normal approximations to the posterior
distribution.
"""
abstract type NormalApproximation end

"""
    FisherMatrix

Normal approximation based on the maximum-likelihood approach. The information matrix is
obtained as the average of the Fisher information matrix with respect to the design measure.
Singular information matrices can occur.
"""
struct FisherMatrix <: NormalApproximation end

@doc raw"""
    Transformation

Abstract supertype of posterior transformations.

Consider the regression model
``y_i \mid \theta \sim \mathrm{Normal}(\mu(\theta, x), \sigma^2)``
with ``\theta\in\Theta\subset\mathbb{R}^q``.
A `Transformation` represents the function ``T: \Theta\to\mathbb{R}^s``
when we want to maximize a [`DesignCriterion`](@ref) for the posterior distribution of
``T(\theta) \mid y``.
"""
abstract type Transformation end

"""
    Identity

Represents the [`Transformation`](@ref) that maps a parameter to itself.
"""
struct Identity <: Transformation end

@doc raw"""
    DeltaMethod(jacobian_matrix)

Represents a nonlinear [`Transformation`](@ref) of the model parameter.

The [delta method](https://en.wikipedia.org/wiki/Delta_method)
maps the asymptotic multivariate normal distribution of ``\theta``
to the asymptotic multivariate normal distribution of ``T(\theta)``,
using the Jacobian matrix ``\mathrm{D}T``.
To construct a `DeltaMethod` object,
the argument `jacobian_matrix` must be a function
that maps a parameter value `p`
to the Jacobian matrix of ``T`` evaluated at `p`.

# Example
Suppose `p` has the fields `a` and `b`, and ``T(a, b) = (ab, b/a)'``.
Then the Jacobian matrix of ``T`` is
```math
\mathrm{D}T(a, b) =
  \begin{bmatrix}
    b      & a   \\
    -b/a^2 & 1/a \\
  \end{bmatrix}.
```
In Julia this is equivalent to
```jldoctest; output = false
jm1(p) = [p.b p.a; -p.b/p.a^2 1/p.a]
DeltaMethod(jm1)
# output
DeltaMethod{typeof(jm1)}(jm1)
```
Note that for a scalar quantity,
e.g. ``T(a, b) = \sqrt{ab}``,
the Jacobian matrix is a _row_ vector.
```jldoctest; output = false
jm2(p) = [b a] ./ (2 * sqrt(p.a * p.b))
DeltaMethod(jm2)
# output
DeltaMethod{typeof(jm2)}(jm2)
```
"""
struct DeltaMethod{T<:Function} <: Transformation
    jacobian_matrix::T # parameter -> Matrix{Float64}
end

abstract type TrafoConstants end

struct TCIdentity <: TrafoConstants
    codomain_dimension::Int64
end
struct TCDeltaMethod <: TrafoConstants
    codomain_dimension::Int64
    jm::Vector{Matrix{Float64}}
end

abstract type GateauxConstants end

struct GCDIdentity <: GateauxConstants
    invM::Vector{Matrix{Float64}}
    parameter_length::Int64
end

struct GCDDeltaMethod <: GateauxConstants
    invM_B_invM::Vector{Matrix{Float64}}
    transformed_parameter_length::Int64
end

"""
    DesignCriterion

Abstract supertype of criteria for optimal experimental design.

See also [`DOptimality`](@ref).
"""
abstract type DesignCriterion end

@doc raw"""
    DOptimality

Criterion for (Bayesian or locally) D-optimal experimental design.

Denote the normalized information matrix for a design measure ``\xi`` by ``\mathrm{M}(\xi,
\theta)``. Assume for simplicity we are interested in the whole parameter ``\theta``, ie.
the [`Transformation`](@ref) is [`Identity`](@ref). Then the objective function for Bayesian
D-optimal design with respect to a prior density $p(\theta)$ is

``\xi \mapsto \int_{\Theta} \log\det \mathrm{M}(\xi, \theta)\,p(\theta)\,\mathrm{d}\theta``

and for locally D-optimal design with respect to a prior guess $\theta_0$ it is

``\xi \mapsto \log\det \mathrm{M}(\xi, \theta_0)``

For details see p. 69 in Fedorov/Leonov [^FL13].

[^FL13]: Valerii V. Fedorov and Sergei L. Leonov, "Optimal design for nonlinear response models", CRC Press, 2013. [doi:10.1201/b15054](https://doi.org/10.1201/b15054)
"""
struct DOptimality <: DesignCriterion end

abstract type AbstractPoint end
abstract type AbstractPointDifference end

"""
    Optimizer

Abstract supertype for particle-based optimization algorithms.

See also [`Pso`](@ref).
"""
abstract type Optimizer end
abstract type OptimizerState{T<:AbstractPoint,U<:AbstractPointDifference} end

abstract type AbstractConstraints end

"""
    OptimizationResult

Wraps results of particle-based optimization.

# `OptimizationResult` fields

| Field           | Description                                          |
|:--------------- |:---------------------------------------------------- |
| maximizer       | final maximizer                                      |
| maximum         | final objective value                                |
| trace_x         | vector of current maximizer in each iteration        |
| trace_fx        | vector of current objective value in each iteration  |
| trace_state     | vector of internal optimizer state in each iteration |
| n_eval          | total number of objective evaluations                |
| seconds_elapsed | total runtime                                        |

Note that `trace_state` may only contain the initial state, if saving all
states was not requested explicitly.

See also [`optimize_design`](@ref).
"""
struct OptimizationResult{
    T<:AbstractPoint,
    U<:AbstractPointDifference,
    S<:OptimizerState{T,U},
}
    maximizer::T
    maximum::Float64
    trace_x::Vector{T}
    trace_fx::Vector{Float64}
    trace_state::Vector{S}
    n_eval::Int64
    seconds_elapsed::Float64
end

"""
    DesignMeasure

A discrete probability measure with finite support
representing a continuous experimental design.

For details see p.62 in Fedorov/Leonov [^FL13].

Special kinds of design measures can be constructed with [`one_point_design`](@ref),
[`uniform_design`](@ref), [`equidistant_design`](@ref), [`random_design`](@ref).

See also [`weights`](@ref), [`designpoints`](@ref), [`as_matrix`](@ref),
[`apportion`](@ref).
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

    # Examples
    ```jldoctest
    julia> DesignMeasure([0.5, 0.2, 0.3], [[1, 2], [3, 4], [5, 6]])
    DesignMeasure(
     [1.0, 2.0] => 0.5,
     [3.0, 4.0] => 0.2,
     [5.0, 6.0] => 0.3
    )
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

# Only used internally to represent difference vectors between DesignMeasures in
# particle-based optimizers. Structurally the same as a DesignMeasure, but weights can be
# arbitrary real numbers.
struct SignedMeasure <: AbstractPointDifference
    weight::Vector{Float64}
    atom::Vector{Vector{Float64}}
    function SignedMeasure(weights, atoms)
        if length(weights) != length(atoms)
            error("number of weights and atoms must be equal")
        end
        if !allequal(map(length, atoms))
            error("atoms must have identical lengths")
        end
        new(weights, atoms)
    end
end

"""
    DesignSpace{N}

Supertype for design spaces. A design space is a compact subset of ``\\mathbb{R}^N``.

See also [`DesignInterval`](@ref), [`dimnames`](@ref).
"""
abstract type DesignSpace{N} end

"""
    DesignInterval{N} <: DesignSpace{N}

A (hyper)rectangular subset of ``\\mathbb{R}^N`` representing the set in which
which the design points of a [`DesignMeasure`](@ref) live.

See also [`lowerbound`](@ref), [`upperbound`](@ref), [`dimnames`](@ref).
"""
struct DesignInterval{N} <: DesignSpace{N}
    name::NTuple{N,Symbol}
    lowerbound::NTuple{N,Float64}
    upperbound::NTuple{N,Float64}
    @doc """
        DesignInterval(name, lowerbound, upperbound)

    Construct a design interval with the given dimension names (supplied as
    symbols) and lower / upper bounds.

    # Examples
    ```jldoctest
    julia> DesignInterval([:dose, :time], [0, 0], [300, 20])
    DesignInterval{2}((:dose, :time), (0.0, 0.0), (300.0, 20.0))
    ```
    """
    function DesignInterval(name, lowerbound, upperbound)
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

struct DesignConstraints{N,T<:DesignSpace{N}} <: AbstractConstraints
    ds::T
    fixw::Vector{Bool}
    fixp::Vector{Bool}
end

# preallocated matrices with dimensions that are often used together.
struct WorkMatrices
    r_x_r::Matrix{Float64}
    r_x_t::Matrix{Float64}
    t_x_r::Matrix{Float64}
    t_x_t::Matrix{Float64}
    m_x_r::Matrix{Float64}
    function WorkMatrices(m::Integer, r::Integer, t::Integer)
        new(zeros(r, r), zeros(r, t), zeros(t, r), zeros(t, t), zeros(m, r))
    end
end
