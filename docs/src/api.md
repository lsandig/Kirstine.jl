# API Reference

## Design Problems

```@docs
DesignProblem
DesignProblem()
```

Different high-level strategies are available for trying to solve a `DesignProblem`.

```@docs
solve
ProblemSolvingStrategy
ProblemSolvingResult
DirectMaximization
DirectMaximization()
DirectMaximizationResult
Exchange
Exchange()
ExchangeResult
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
Kirstine.update_model_covariate!
Kirstine.invcov
Kirstine.unit_length
dimension
```

### Helper Macros

To reduce boilerplate code in the common cases of a one-dimensional unit of observation,
and for a vector parameter without any additional structure,
the following helper macros can be used:

```@docs
@define_scalar_unit_model
@define_vector_parameter
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
DesignMeasure(::AbstractVector{<:AbstractVector{<:Real}}, ::AbstractVector{<:Real})
DesignMeasure(::Pair...)
DesignMeasure(::AbstractMatrix{<:Real})
as_matrix
one_point_design
uniform_design
equidistant_design
random_design
weights
designpoints
sort_designpoints
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
```

In addition, objects of the following types can be plotted directly:

```julia
plot(d::DesignMeasure)
plot(d::DesignMeasure, dr::DesignInterval)
plot(r::OptimizationResult)
plot(rs::AbstractVector{<:OptimizationResult})
plot(r::DirectMaximizationResult)
plot(r::ExchangeResult)
```
