# Kirstine.jl

*Optimal Designs for Nonlinear Regression.*

```@eval
import Pkg
ver = Pkg.project().version
"Version $(ver)"
```

## Design Problems

```@docs
DesignProblem
DesignProblem()
```

Different high-level algorithms are available for trying to solve a `DesignProblem`.

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
```

## Design Criteria

```@docs
DesignCriterion
DOptimality
objective
gateauxderivative
efficiency
```

## Design Spaces

```@docs
DesignSpace
DesignInterval
DesignInterval()
Kirstine.dimension
dimnames
upperbound
lowerbound
```

## Nonlinear Regression Models

Regression models are implemented
by first subtyping `Model`, `Covariate`, `CovariateParameterization`, and `Parameter`,
and then defining several helper methods on them.
See [“Getting Started”](getting-started.md) for a hands-on example.

```@docs
Model
NonlinearRegression
Covariate
@define_scalar_unit_model
CovariateParameterization
Parameter
@define_vector_parameter
```

## Prior Knowledge

```@docs
PriorKnowledge
DiscretePrior
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
```

## Plotting

```@docs
plot_gateauxderivative
```
