# API Reference

## Design Problems

```@docs
DesignProblem
DesignProblem()
criterion
region
model
covariate_parameterization
prior_knowledge
transformation
normal_approximation
```

Different high-level strategies are available for trying to solve a `DesignProblem`.

```@docs
solve
ProblemSolvingStrategy
ProblemSolvingResult
DirectMaximization
DirectMaximization()
DirectMaximizationResult
solution(::DirectMaximizationResult)
optimization_result
Exchange
Exchange()
ExchangeResult
solution(::ExchangeResult)
optimization_results_direction
optimization_results_weight
```

Particle-based optimizers are operating on a lower level,
and are used as part of a `ProblemSolvingStrategy`.

```@docs
Optimizer
OptimizationResult
Pso
Pso()
```

## Design Criteria

```@docs
DesignCriterion
DOptimality
AOptimality
objective
gateauxderivative
efficiency
shannon_information
```

## Design Regions

```@docs
DesignRegion
DesignInterval
DesignInterval(::Any, ::Any, ::Any)
DesignInterval()
dimension(::DesignRegion)
dimnames
upperbound
lowerbound
```

## Nonlinear Regression Models

Regression models are implemented
by first subtyping `Model`, `Covariate`, `CovariateParameterization`, and `Parameter`,
and then defining several helper methods on them.
See the [tutorial](tutorial.md) for a hands-on example.

### Supertypes

```@docs
Model
NonlinearRegression
Covariate
CovariateParameterization
Parameter
```

### Implementing a Nonlinear Regression Model

Suppose the following subtypes have been defined:

  - `M <: NonlinearRegression`
  - `C <: Covariate`
  - `Cp <: CovariateParameterization`
  - `P <: Parameter`

Then methods need to be added for the following package-internal functions:

```@docs
Kirstine.allocate_covariate
Kirstine.jacobianmatrix!
Kirstine.map_to_covariate!
Kirstine.update_model_vcov!
Kirstine.unit_length
dimension
```

!!! note
    
    Some of these methods only require `M <: Model` and not `M <: NonlinearRegression`.
    This is intentional.
    These methods can in principle also be implemented for `M <: Ma`
    where `abstract type Ma <: Model end` is also user-defined.

### Helpers

To reduce boilerplate code in the common cases of a one-dimensional unit of observation,
and for a vector parameter without any additional structure,
the following helper macros can be used:

```@docs
@simple_model
@simple_parameter
```

In problems where the design variables coincide with the model covariates,
the following `CovariateParameterization` can be used:

```@docs
JustCopy
JustCopy()
```

## Prior Knowledge

```@docs
PriorKnowledge
PriorSample
PriorSample(::AbstractVector{T}, ::AbstractVector{<:Real}) where T <: Parameter
```

## Transformations

```@docs
Transformation
Identity
DeltaMethod
```

## Normal Approximations

```@docs
NormalApproximation
FisherMatrix
```

## Design Measures

```@docs
DesignMeasure
Base.:(==)(::DesignMeasure, ::DesignMeasure)
DesignMeasure(::AbstractMatrix{<:Real}, ::AbstractVector{<:Real})
DesignMeasure(::AbstractVector{<:AbstractVector{<:Real}}, ::AbstractVector{<:Real})
DesignMeasure(::Pair...)
one_point_design
uniform_design
equidistant_design
random_design
points
weights
numpoints
sort_points
sort_weights
mixture
apportion
simplify
simplify_drop
simplify_unique
simplify_merge
informationmatrix
```

## Plotting

```@docs
plot_gateauxderivative
plot_expected_function
```

In addition, objects of the following types can be plotted directly.
Note that all plots involving a `DesignMeasure` can take the `label_formatter` keyword argument.

```julia
plot(d::DesignMeasure)
plot(d::DesignMeasure, dr::DesignInterval)
plot(r::OptimizationResult)
plot(rs::AbstractVector{<:OptimizationResult})
plot(r::DirectMaximizationResult)
plot(r::ExchangeResult)
```
