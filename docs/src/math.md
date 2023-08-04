# Mathematical Background

Terms and notations in the field of experimental design vary between authors.
This pages documents the notation for `Kirstine.jl`.

## Design Problems

A design problem is a tuple

```math
(
  \DesignCriterion,
  \DesignRegion,
  \MeanFunction,
  \CovariateParameterization,
  \PriorDensity,
  \Transformation
)
```

with the following elements:

  - A _design criterion_ ``\DesignCriterion : \SNNDMatrices{\DimTransformedParameter} → \Reals``.
    This is the functional that maps the normalized information matrix to a real number.
    See [D-Optimality](@ref) and [A-Optimality](@ref) for specific instances.

  - A compact _design region_ ``\DesignRegion ⊂ \Reals^{\DimDesignRegion}``.
  - A _mean function_ ``\MeanFunction : \CovariateSet × \ParameterSet → \Reals^{\DimUnit}``.
    The nonlinear regression model is defined as
    
    ```math
    \Unit_{\IndexUnit} \mid \Parameter
    \simiid
    \MvNormDist(\MeanFunction(\Covariate_{\IndexDesignPoint}, \Parameter), \UnitCovariance)
    \quad \text{ for } \IndexUnit ∈ \IndexSet_{\IndexDesignPoint}
    \text{ and } \IndexDesignPoint = 1,…,\NumDesignPoints
    ```
    
    with ``\SampleSize`` units of observation ``\Unit_{\IndexUnit} ∈ \Reals^{\DimUnit}``
    across ``\NumDesignPoints`` groups of size
    ``\SampleSize_1,…,\SampleSize_{\NumDesignPoints}``
    with corresponding covariates
    ``\Covariate_{\IndexDesignPoint} ∈ \CovariateSet ⊂ \Reals^{\DimCovariate}``.
    In simple cases, ``\DimUnit=1`` or ``\DimCovariate=1``.
    The mean function ``\MeanFunction`` must be continuous in ``\Covariate``
    and continuously differentiable in ``\Parameter``.
    The covariance matrix ``\UnitCovariance ∈ \SNNDMatrices{\DimUnit}``
    is assumed to be known and constant.
  - A _covariate parameterization_ ``\CovariateParameterization : \DesignRegion → \CovariateSet``.
    This maps a design point to a model covariate.
  - A _prior density_ ``\PriorDensity : \ParameterSet → [0, ∞)``.
    This encodes the prior knowledge about the model parameter ``\Parameter``.
  - A _posterior transformation_ ``\Transformation : \ParameterSet → \Reals^{\DimTransformedParameter}``.
    This is used to indicate when we are not interested in finding a design for the parameter ``\Parameter``,
    but for some transformation of it.
    ``\Transformation`` must be continuously differentiable.
    In simple cases, ``\Transformation(\Parameter)=\Parameter``.

!!! info
    
    The [`DesignProblem`](@ref) type has an additional seventh field for a [`NormalApproximation`](@ref).
    This is because the precision matrix of the underlying posterior normal approximation
    can be obtained in different ways
    (see e.g. p. 224 in Berger[^B85]).
    However, `Kirstine.jl` currently only supports using the likelihood-based
    Fisher information matrix.

[^B85]: James O. Berger. Statistical decision theory and Bayesian analysis. Springer, 1985.
## Design Measures

A probability measure ``\DesignMeasure`` on
(some sigma algebra on) the design region ``\DesignRegion``
is called a _design measure_.
For a discrete design measure

```math
\DesignMeasure = \sum_{\IndexDesignPoint=1}^{\NumDesignPoints}
  \DesignWeight_{\IndexDesignPoint} \DiracDist(\DesignPoint_{\IndexDesignPoint})
```

the points ``\DesignPoint_{\IndexDesignPoint}`` of its support are called _design points_.

## Objective Function

The following function is to be maximized over the set ``\AllDesignMeasures`` of all design measures:

```math
\Objective(\DesignMeasure) = \IntD{\ParameterSet}{\DesignCriterion(\TNIMatrix(\DesignMeasure, \Parameter))}{\PriorDensity(\Parameter)}{\Parameter}
```

## Gateaux Derivative

## D-Optimality

```math
\DesignCriterion_D(\SomeMatrix) = \log\det\SomeMatrix
```

## A-Optimality

```math
\DesignCriterion_A(\SomeMatrix) = -\Trace\SomeMatrix^{-1}
```
