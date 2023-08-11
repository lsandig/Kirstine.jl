# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

# == abstract supertypes for user-defined models == #

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

# == package types == #

"""
    Parameter

Supertype for [`Model`](@ref) parameters.
"""
abstract type Parameter end

"""
    PriorKnowledge{T<:Parameter}

Supertype for representing prior knowledge of the model [`Parameter`](@ref).

See also [`PriorSample`](@ref).
"""
abstract type PriorKnowledge{T<:Parameter} end

"""
    PriorSample{T} <: PriorKnowledge{T}

A sample from a prior distribution, or a discrete prior distribution with finite support.
"""
struct PriorSample{T} <: PriorKnowledge{T}
    weight::Vector{Float64}
    p::Vector{T}
    @doc """
    PriorSample(p::AbstractVector{<:Parameter} [, weights::AbstractVector{<:Real}])

Construct a weighted prior sample on the given parameter draws.

If no `weights` are given, a uniform distribution on the elements of `p` is assumed.
"""
    function PriorSample(
        parameters::AbstractVector{T},
        weights::AbstractVector{<:Real} = fill(1 / length(parameters), length(parameters)),
    ) where T<:Parameter
        if length(weights) != length(parameters)
            error("number of weights and parameter values must be equal")
        end
        if any(weights .< 0) || !(sum(weights) ≈ 1)
            error("weights must be non-negative and sum to one")
        end
        return new{T}(weights, parameters)
    end
end

"""
    NormalApproximation

Supertype for different possible normal approximations to the posterior distribution.
"""
abstract type NormalApproximation end

"""
    FisherMatrix

Normal approximation based on the maximum-likelihood approach.

The information matrix is obtained as the average of the Fisher information matrix with
respect to the design measure. Singular information matrices can occur.

See also the [mathematical background](math.md#Objective-Function).
"""
struct FisherMatrix <: NormalApproximation end

@doc raw"""
    Transformation

Supertype of posterior transformations.

See also [`Identity`](@ref), [`DeltaMethod`](@ref),
and the [mathematical background](math.md#Objective-Function).
"""
abstract type Transformation end

"""
    Identity <: Transformation

The [`Transformation`](@ref) that maps a parameter to itself.
"""
struct Identity <: Transformation end

@doc raw"""
    DeltaMethod <: Transformation
    DeltaMethod(jacobian_matrix::Function)

A nonlinear [`Transformation`](@ref) of the model parameter.

The [delta method](https://en.wikipedia.org/wiki/Delta_method)
uses the Jacobian matrix ``\TotalDiff\Transformation``
to map the asymptotic multivariate normal distribution of ``\Parameter``
to the asymptotic multivariate normal distribution of ``\Transformation(\Parameter)``.

To construct a `DeltaMethod` object,
`jacobian_matrix` must be a function
that maps a [`Parameter`](@ref) `p`
to the Jacobian matrix of ``\Transformation`` evaluated at `p`.

# Examples
Suppose `p` has the fields `a` and `b`, and ``\Transformation(\Parameter) = (ab, b/a)'``.
Then the Jacobian matrix of ``\Transformation`` is
```math
\TotalDiff\Transformation(\Parameter) =
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
e.g. ``\Transformation(\Parameter) = \sqrt{ab}``,
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

struct GCAIdentity <: GateauxConstants
    B::Vector{Matrix{Float64}} # inv(M(at))^2
    tr_C::Vector{Float64}      # tr(inv(M(at)))
end

struct GCADeltaMethod <: GateauxConstants
    B::Vector{Matrix{Float64}} # inv(M(at)) * J' * J * inv(M(at))
    tr_C::Vector{Float64}      # tr(J' * J * inv(M(at)))
end

"""
    DesignCriterion

Supertype for optimal experimental design criteria.

See also [`DOptimality`](@ref), [`AOptimality`](@ref).
"""
abstract type DesignCriterion end

@doc raw"""
    DOptimality

Criterion for D-optimal experimental design.

Log-determinant of the normalized information matrix.

See also the [mathematical background](math.md#D-Criterion).
"""
struct DOptimality <: DesignCriterion end

"""
    AOptimality <: DesignCriterion

Criterion for A-optimal experimental design.

Trace of the inverted normalized information matrix.

See also the [mathematical background](math.md#A-Criterion).
"""
struct AOptimality <: DesignCriterion end

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

"""
    OptimizationResult

Wrapper for results of particle-based optimization.

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

Note that `trace_state` will only contain the initial state when saving all states was not
explicitly requested.

See also [`solve`](@ref).
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

A probability measure with finite support representing a continuous experimental design.

The support points of a design measure are called _design points_.
In Julia, a design point is simply a `Vector{Float64}`.

Special kinds of design measures can be constructed with [`one_point_design`](@ref),
[`uniform_design`](@ref), [`equidistant_design`](@ref), [`random_design`](@ref).

See also [`weights`](@ref), [`designpoints`](@ref), [`as_matrix`](@ref),
[`apportion`](@ref).
"""
struct DesignMeasure <: AbstractPoint
    designpoint::Vector{Vector{Float64}}
    weight::Vector{Float64}
    @doc """
        DesignMeasure(
            designpoints::AbstractVector{<:AbstractVector{<:Real}},
            weights::AbstractVector{<:Real},
        )

    Construct a design measure.

    # Examples
    ```jldoctest
    julia> DesignMeasure([[1, 2], [3, 4], [5, 6]], [0.5, 0.2, 0.3])
    DesignMeasure(
     [1.0, 2.0] => 0.5,
     [3.0, 4.0] => 0.2,
     [5.0, 6.0] => 0.3,
    )
    ```
    """
    function DesignMeasure(
        designpoint::AbstractVector{<:AbstractVector{<:Real}},
        weight::AbstractVector{<:Real},
    )
        if length(weight) != length(designpoint)
            error("number of weights and design points must be equal")
        end
        if !allequal(map(length, designpoint))
            error("design points must have identical lengths")
        end
        if any(weight .< 0) || !(sum(weight) ≈ 1)
            error("weights must be non-negative and sum to one")
        end
        new(designpoint, weight)
    end
end

# Only used internally to represent difference vectors between DesignMeasures in
# particle-based optimizers. Structurally the same as a DesignMeasure, but weights can be
# arbitrary real numbers.
struct SignedMeasure <: AbstractPointDifference
    atom::Vector{Vector{Float64}}
    weight::Vector{Float64}
    function SignedMeasure(atoms, weights)
        if length(weights) != length(atoms)
            error("number of weights and atoms must be equal")
        end
        if !allequal(map(length, atoms))
            error("atoms must have identical lengths")
        end
        new(atoms, weights)
    end
end

"""
    DesignRegion{N}

Supertype for design regions. A design region is a compact subset of ``\\Reals^N``.

The design points of a [`DesignMeasure`](@ref) are taken from this set.

See also [`DesignInterval`](@ref), [`dimnames`](@ref).
"""
abstract type DesignRegion{N} end

"""
    DesignInterval{N} <: DesignRegion{N}

A (hyper)rectangular subset of ``\\Reals^N``.

See also [`lowerbound`](@ref), [`upperbound`](@ref), [`dimnames`](@ref).
"""
struct DesignInterval{N} <: DesignRegion{N}
    name::NTuple{N,Symbol}
    lowerbound::NTuple{N,Float64}
    upperbound::NTuple{N,Float64}
    @doc """
        DesignInterval(names, lowerbounds, upperbounds)

    Construct a design interval.

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

struct DesignConstraints{N,T<:DesignRegion{N}} <: AbstractConstraints
    dr::T
    fixw::Vector{Bool}
    fixp::Vector{Bool}
    function DesignConstraints(dr::T, fixw, fixp) where {N,T<:DesignRegion{N}}
        if length(fixw) != length(fixp)
            throw(DimensionMismatch("fix vectors must have identical lengths"))
        end
        return new{N,T}(dr, fixw, fixp)
    end
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

"""
    DesignProblem

A `DesignProblem` has 7 components:

  - a [`DesignCriterion`](@ref),
  - a [`DesignRegion`](@ref),
  - a [`Model`](@ref),
  - a [`CovariateParameterization`](@ref),
  - some [`PriorKnowledge`](@ref),
  - a [`Transformation`](@ref),
  - and a [`NormalApproximation`](@ref).

See also [`solve`](@ref),
and the [mathematical background](math.md#Design-Problems).
"""
struct DesignProblem{
    Tdc<:DesignCriterion,
    Tdr<:DesignRegion,
    Tm<:Model,
    Tcp<:CovariateParameterization,
    Tpk<:PriorKnowledge,
    Tt<:Transformation,
    Tna<:NormalApproximation,
}
    dc::Tdc
    dr::Tdr
    m::Tm
    cp::Tcp
    pk::Tpk
    trafo::Tt
    na::Tna
    @doc """
        DesignProblem(<keyword arguments>)

    Construct a design problem with some sensible defaults.

    # Arguments
    - `design_criterion::DesignCriterion`
    - `design_region::DesignRegion`
    - `model::Model`
    - `covariate_parameterization::CovariateParameterization`
    - `prior_knowledge::PriorKnowledge`
    - `transformation::Transformation = Identity()`
    - `normal_approximation::NormalApproximation = FisherMatrix()`
    """
    function DesignProblem(;
        design_criterion::Tdc,
        design_region::Tdr,
        model::Tm,
        covariate_parameterization::Tcp,
        prior_knowledge::Tpk,
        transformation::Tt = Identity(),
        normal_approximation::Tna = FisherMatrix(),
    ) where {
        Tdc<:DesignCriterion,
        Tdr<:DesignRegion,
        Tm<:Model,
        Tcp<:CovariateParameterization,
        Tpk<:PriorKnowledge,
        Tt<:Transformation,
        Tna<:NormalApproximation,
    }
        new{Tdc,Tdr,Tm,Tcp,Tpk,Tt,Tna}(
            design_criterion,
            design_region,
            model,
            covariate_parameterization,
            prior_knowledge,
            transformation,
            normal_approximation,
        )
    end
end

"""
    ProblemSolvingStrategy

Supertype for algorithms for solving a [`DesignProblem`](@ref).
"""
abstract type ProblemSolvingStrategy end

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
    Exchange <: ProblemSolvingStrategy

Find an optimal design by ascending the Gateaux derivative of the objective function.

This is a variant of the basic idea in [^YBT13].

[^YBT13]: Min Yang, Stefanie Biedermann, Elina Tang (2013). On optimal designs for nonlinear models: a general and efficient algorithm. Journal of the American Statistical Association, 108(504), 1411–1420. [doi:10.1080/01621459.2013.806268](http://dx.doi.org/10.1080/01621459.2013.806268)
"""
struct Exchange{Tod<:Optimizer,Tow<:Optimizer,Td<:AbstractDict{Symbol,Any}} <:
       ProblemSolvingStrategy
    od::Tod
    ow::Tow
    steps::Int64
    candidate::DesignMeasure
    simplify_args::Td
    @doc """
        Exchange(;
            od::Optimizer,
            ow::Optimizer,
            steps::Integer,
            candidate::DesignMeasure,
            simplify_args::Dict,
        )

    Improve the `candidate` design by repeating the following `steps` times,
    starting with `r = candidate`:

     1. [`simplify`](@ref) the design `r`, passing along `sargs`.
     2. Use the optimizer `od` to find the direction (one-point design / Dirac measure) `d`
        of highest [`gateauxderivative`](@ref) at `r`.
        The vector of `prototypes` that is used for initializing `od`
        is constructed from one-point designs at the design points of `r`.
        See the [`Optimizer`](@ref)s for algorithm-specific details.
     3. Use the optimizer `ow` to re-calculcate optimal weights
        of a [`mixture`](@ref) of `r` and `d` for the [`DesignCriterion`](@ref).
        This is implemented as a call to [`solve`](@ref)
        with the [`DirectMaximization`](@ref) strategy
        and with all of the designpoints of `r` kept fixed.
     4. Set `r` to the result of step 3.

    The return value of [`solve`](@ref) for this strategy is an [`ExchangeResult`](@ref).
    """
    function Exchange(;
        od::Tod,
        ow::Tow,
        steps::Integer,
        candidate::DesignMeasure,
        simplify_args::Td = Dict{Symbol,Any}(),
    ) where {Tod<:Optimizer,Tow<:Optimizer,Td<:AbstractDict{Symbol,Any}}
        new{Tod,Tow,Td}(od, ow, steps, candidate, simplify_args)
    end
end

"""
    ProblemSolvingResult

Supertype for results of a [`ProblemSolvingStrategy`](@ref).
"""
abstract type ProblemSolvingResult end

"""
    DirectMaximizationResult <: ProblemSolvingResult

Contains an [`OptimizationResult`](@ref) in the `or` field.
"""
struct DirectMaximizationResult{S<:OptimizerState{DesignMeasure,SignedMeasure}} <:
       ProblemSolvingResult
    or::OptimizationResult{DesignMeasure,SignedMeasure,S}
end

"""
    ExchangeResult <: ProblemSolvingResult

Contains vectors `ord` and `orw` of [`OptimizationResult`](@ref)s,
one for each direction finding and reweighting step.
"""
struct ExchangeResult{
    S<:OptimizerState{DesignMeasure,SignedMeasure},
    T<:OptimizerState{DesignMeasure,SignedMeasure},
} <: ProblemSolvingResult
    ord::Vector{OptimizationResult{DesignMeasure,SignedMeasure,S}}
    orw::Vector{OptimizationResult{DesignMeasure,SignedMeasure,T}}
end
