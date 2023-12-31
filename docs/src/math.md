# Mathematical Background

Terms and notations in the field of experimental design vary between authors.
This page documents the notation for `Kirstine.jl`.

## Preliminaries

  - The set of ``a×a`` symmetric non-negative definite matrices is denoted ``\SNNDMatrices{a}``.

  - The total derivative / Jacobian matrix operator is denoted ``\TotalDiff``.
  - The Hessian matrix operator is denoted ``\Hessian``.
  - The prime symbol ``(·)'`` denotes the transpose of a matrix, not a first derivative.
  - The matrix-valued derivative
    ``\MatDeriv{φ}{A}{·}``
    of a differentiable function ``φ : \Reals^{a×b} → \Reals``
    organizes the partial derivatives of ``φ`` with respect to the elements of ``A``
    in the same structure as the input matrix ``A``:
    
    ```math
    \MatDeriv{φ}{A}{A}
    =
    \biggl[
    \frac{∂φ}{∂A_{ij}}(A)
    \biggr]_{i=1,…,a; j=1,…,b}
    ```
    
    For information on finding ``\MatDeriv{φ}{A}{·}`` for certain ``φ``,
    see the book by Magnus and Neudecker[^MN99].
  - The log-likelihood of a statistical model with
    observation ``\Unit``,
    covariate ``\Covariate``,
    and parameter ``\Parameter``
    is denoted ``\LogLikelihood(\Unit, \Covariate, \Parameter)``.

[^MN99]: Jan R. Magnus and Heinz Neudecker (1999). Matrix differential calculus with applications in statistics and econometrics. Wiley. [doi:10.1002/9781119541219](https://doi.org/10.1002/9781119541219)
## Design Problems

A design problem is a tuple

```math
(
  \DesignCriterion,
  \DesignRegion,
  \MeanFunction,
  \UnitCovariance,
  \CovariateParameterization,
  \PriorDensity,
  \Transformation
)
```

with the following elements:

  - A _design criterion_ ``\DesignCriterion : \SNNDMatrices{\DimTransformedParameter} → \Reals``.
    This is the functional that maps the normalized information matrix to a real number.
    ``\DesignCriterion`` must be concave, monotonically non-decreasing and differentiable.
    See also [D-Criterion](@ref) and [A-Criterion](@ref).

  - A _design region_ ``\DesignRegion ⊂ \Reals^{\DimDesignRegion}``.
    ``\DesignRegion`` must be compact.
  - The _mean function_ ``\MeanFunction : \CovariateSet × \ParameterSet → \Reals^{\DimUnit}``
    and _covariance matrix_ ``\UnitCovariance : \CovariateSet → \SNNDMatrices{\DimUnit}``
    of a nonlinear regression model
    
    ```math
    \Unit_{\IndexUnit} \mid \Parameter
    \simiid
    \MvNormDist(\MeanFunction(\Covariate_{\IndexDesignPoint}, \Parameter),
                \UnitCovariance(\Covariate_{\IndexDesignPoint}))
    \quad \text{ for } \IndexUnit ∈ \IndexSet_{\IndexDesignPoint}
    \text{ and } \IndexDesignPoint = 1,…,\NumDesignPoints
    ```
    
    with ``\SampleSize`` units of observation ``\Unit_{\IndexUnit} ∈ \Reals^{\DimUnit}``
    across ``\NumDesignPoints`` groups of size
    ``\SampleSize_1,…,\SampleSize_{\NumDesignPoints}``
    with corresponding covariates
    ``\Covariate_{\IndexDesignPoint} ∈ \CovariateSet ⊂ \Reals^{\DimCovariate}``
    and parameter set ``\ParameterSet ⊂ \Reals^{\DimParameter}``.
    The _mean function_ ``\MeanFunction`` must be continuous in ``\Covariate``
    and continuously differentiable in ``\Parameter``.
    The covariance matrix ``\UnitCovariance``
    may continuously depend on the covariate,
    but is otherwise assumed to be known.
    In simple cases,
    ``\DimUnit=1``,
    ``\DimCovariate=1``,
    with ``\UnitCovariance(\Covariate) = \ScalarUnitVariance`` constant.
  - A _covariate parameterization_ ``\CovariateParameterization : \DesignRegion → \CovariateSet``.
    This maps a design point to a model covariate.
  - A _prior density_ ``\PriorDensity : \ParameterSet → [0, ∞)``.
    This encodes the prior knowledge about the model parameter ``\Parameter``.
    The dominating measure is implicit,
    and can be either the ``\DimParameter``-dimensional Lebesgue measure
    or a counting measure.
  - A _posterior transformation_ ``\Transformation : \ParameterSet → \Reals^{\DimTransformedParameter}``.
    This is used to indicate when we are not interested in finding a design for the original parameter ``\Parameter``,
    but for some transformation of it.
    ``\Transformation`` must be continuously differentiable.
    In simple cases, ``\Transformation(\Parameter)=\Parameter``.

!!! info
    
    The [`DesignProblem`](@ref) type has an additional seventh field for a [`NormalApproximation`](@ref).
    This is because the precision matrix of the underlying posterior normal approximation
    can be obtained in different ways
    (see e.g. p. 224 in Berger[^B85]).
    However, `Kirstine.jl` officially only supports the likelihood-based
    Fisher information matrix.
    If your problem requires ad-hoc matrix regularization,
    you need to [implement it yourself](extend-approximation.md).

[^B85]: James O. Berger (1985). Statistical decision theory and Bayesian analysis. Springer.
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

The set ``\AllDesignMeasures`` denotes the set of all design measures on (some sigma algebra on) the design region.

## Objective Function

The following function is to be maximized over the set ``\AllDesignMeasures`` of all design measures:

```math
\Objective(\DesignMeasure) = \IntD{\ParameterSet}{\DesignCriterion(\TNIMatrix(\DesignMeasure, \Parameter))}{\PriorDensity(\Parameter)}{\Parameter}
```

Here, ``\TNIMatrix`` denotes the normalized information matrix for ``\Transformation(\Parameter)``,

```math
\TNIMatrix(\DesignMeasure, \Parameter)
=
\bigl[
 (\TotalDiff\Transformation(\Parameter))
  \NIMatrix^{-1}(\DesignMeasure, \Parameter)
 (\TotalDiff\Transformation(\Parameter))'
\bigr]^{-1},
```

and ``\TotalDiff\Transformation`` denotes the Jacobian matrix of the posterior transformation ``\Transformation``.

It is obtained from the normalized information matrix for ``\Parameter``
by applying multivariate delta method to the underlying normal approximation.

The normalized information matrix for the original parameter ``\Parameter``
is given by the average

```math
\NIMatrix(\DesignMeasure, \Parameter)
=
\AverageFisherMatrix(\DesignMeasure, \Parameter)
=
\IntM{\DesignRegion}{
 \FisherMatrix(\CovariateParameterization(\DesignPoint), \Parameter)
}{\DesignMeasure}{\DesignPoint},
```

over the model's Fisher information matrix

```math
\FisherMatrix(\Covariate, \Parameter)
=
- \Expectation[\Hessian\LogLikelihood(\Unit,\Covariate,\Parameter)\mid\Parameter]
```

where ``\Hessian`` is the Hessian matrix with respect to ``\Parameter``,
and the expectation is taken with respect to ``\Unit``.
With the log-likelihood for nonlinear regression

```math
\LogLikelihood(\Unit,\Covariate,\Parameter)
=
\log\MvNormDist(\Unit\mid\MeanFunction(\Covariate,\Parameter), \UnitCovariance(\Covariate))
```

the Fisher information matrix turns out to be

```math
\FisherMatrix(\Covariate, \Parameter)
=
(\TotalDiff\MeanFunction(\Covariate, \Parameter))'
\UnitCovariance(\Covariate)^{-1}
\TotalDiff\MeanFunction(\Covariate, \Parameter),
```

where ``\TotalDiff\MeanFunction`` denotes the Jacobian matrix of the mean function ``\MeanFunction``
with respect to ``\Parameter``.

## Gateaux Derivative

The infinite-dimensional directional derivative of ``\Objective``
at ``\DesignMeasure ∈ \AllDesignMeasures``
into the direction of ``\DesignMeasureDirection ∈ \AllDesignMeasures``
is called the _Gateaux derivative_ of ``\Objective``.
For a general design criterion ``\DesignCriterion``,
the Gateaux derivative is given by

```math
\begin{aligned}
\GateauxDerivative(\DesignMeasure, \DesignMeasureDirection)
&=
\lim_{α→0} \frac{1}{α}(\Objective((1 - α)\DesignMeasure + α\DesignMeasureDirection) - \Objective(\DesignMeasure))\\
&=
\IntD{\ParameterSet}{
\biggl\{
\Trace\biggl[
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
 (\TotalDiff \Transformation(\Parameter))'
  \TNIMatrix(\DesignMeasure, \Parameter)
   \MatDeriv{\DesignCriterion}{\SomeMatrix}{\TNIMatrix(\DesignMeasure, \Parameter)}
  \TNIMatrix(\DesignMeasure, \Parameter)
 (\TotalDiff \Transformation(\Parameter))
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
\NIMatrix(\DesignMeasureDirection, \Parameter)
\biggr] \\
&\qquad
-
\Trace\biggl[
\MatDeriv{\DesignCriterion}{\SomeMatrix}{\TNIMatrix(\DesignMeasure, \Parameter)}
\TNIMatrix(\DesignMeasure, \Parameter)
\biggr]
\biggr\}
}{\PriorDensity(\Parameter)}{\Parameter}
.
\end{aligned}
```

Note that the direction ``\DesignMeasureDirection`` enters the Gateaux derivative
only through the final matrix in the first trace.

If ``\Transformation(\Parameter)=\Parameter``, the expression above simplifies to

```math
\GateauxDerivative(\DesignMeasure, \DesignMeasureDirection)
=
\IntD{\ParameterSet}{
\biggl\{
\Trace\biggl[
\MatDeriv{\DesignCriterion}{\SomeMatrix}{\NIMatrix(\DesignMeasure, \Parameter)}
\NIMatrix(\DesignMeasureDirection, \Parameter)
\biggr]
-
\Trace\biggl[
\MatDeriv{\DesignCriterion}{\SomeMatrix}{\NIMatrix(\DesignMeasure, \Parameter)}
\NIMatrix(\DesignMeasure, \Parameter)
\biggr]
\biggr\}
}{\PriorDensity(\Parameter)}{\Parameter}
.
```

In both cases, the general form of the Gateaux derivative is

```math
\GateauxDerivative(\DesignMeasure, \DesignMeasureDirection)
=
\IntD{\ParameterSet}{
\bigl\{
\Trace\bigl[
A(\DesignMeasure, \Parameter)
\NIMatrix(\DesignMeasureDirection, \Parameter)
\bigr]
-
\Trace\bigl[
B(\DesignMeasure, \Parameter)
\bigr]
\bigr\}
}{\PriorDensity(\Parameter)}{\Parameter}
,
```

where the matrices ``A`` and ``B`` are specific to the design criterion and transformation,
but do not depend on the direction ``\DesignMeasureDirection``.
Since they are constant in ``\DesignMeasureDirection``
they only need to be computed once.

Note that the function

```math
\Sensitivity : \AllDesignMeasures × \DesignRegion → \Reals,
\quad
\Sensitivity(\DesignMeasure, \DesignPoint)
=
\IntD{\ParameterSet}{
\Trace\bigl[
A(\DesignMeasure, \Parameter)
\NIMatrix(\DiracDist(\DesignPoint), \Parameter)
\bigr]
}{\PriorDensity(\Parameter)}{\Parameter}
```

is sometimes known as the _sensitivity function_.

An equivalence theorem states:
a design measure ``\DesignMeasure^*`` maximizes ``\Objective`` iff

```math
\GateauxDerivative(\DesignMeasure^*, \DiracDist(\DesignPoint)) ≤ 0
```

for all ``\DesignPoint ∈ \DesignRegion``.

## D-Criterion

```math
\DesignCriterion_D(\SomeMatrix) = \log\det\SomeMatrix
```

A design which is optimal with respect to ``\DesignCriterion_D``
maximizes the approximate expected posterior Shannon information about ``\Transformation(\Parameter)``.

The Gateaux derivative is

```math
\GateauxDerivative(\DesignMeasure, \DesignMeasureDirection)
=
\IntD{\ParameterSet}{
\Trace\bigl[
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
 (\TotalDiff \Transformation(\Parameter))'
  \TNIMatrix(\DesignMeasure, \Parameter)
 (\TotalDiff \Transformation(\Parameter))
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
\NIMatrix(\DesignMeasureDirection, \Parameter)
\bigr]
}{\PriorDensity(\Parameter)}{\Parameter} - \DimTransformedParameter
.
```

For ``\Transformation(\Parameter) = \Parameter`` this simplifies to

```math
\GateauxDerivative(\DesignMeasure, \DesignMeasureDirection)
=
\IntD{\ParameterSet}{
\Trace\bigl[
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
\NIMatrix(\DesignMeasureDirection, \Parameter)
\bigr]
}{\PriorDensity(\Parameter)}{\Parameter} - \DimParameter
.
```

## A-Criterion

```math
\DesignCriterion_A(\SomeMatrix) = -\Trace\SomeMatrix^{-1}
```

A design which is optimal with respect to ``\DesignCriterion_A``
minimizes the approximate average posterior variance for the elements of ``\Transformation(\Parameter)``.

The Gateaux derivative is

```math
\GateauxDerivative(\DesignMeasure, \DesignMeasureDirection)
=
\IntD{\ParameterSet}{
\bigl\{
\Trace\bigl[
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
(\TotalDiff\Transformation(\Parameter))'
(\TotalDiff\Transformation(\Parameter))
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
\NIMatrix(\DesignMeasureDirection, \Parameter)
\bigr]
-
\Trace\bigl[
\TNIMatrix^{-1}(\DesignMeasure, \Parameter)
\bigr]
\bigr\}
}{\PriorDensity(\Parameter)}{\Parameter}
.
```

For ``\Transformation(\Parameter) = \Parameter`` this simplifies to

```math
\GateauxDerivative(\DesignMeasure, \DesignMeasureDirection)
=
\IntD{\ParameterSet}{
\bigl\{
\Trace\bigl[
\NIMatrix^{-2}(\DesignMeasure, \Parameter)
\NIMatrix(\DesignMeasureDirection, \Parameter)
\bigr]
-
\Trace\bigl[
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
\bigr]
\bigr\}
}{\PriorDensity(\Parameter)}{\Parameter}
.
```

## Shannon Information

The _approximate expected posterior Shannon information_ for an experiment with the design measure ``\DesignMeasure``
and a sample size of ``\SampleSize`` is

```math
\frac{\DimTransformedParameter}{2}\log(\SampleSize)
- \frac{\DimTransformedParameter}{2}(1 + \log(2π))
+ \frac{1}{2} \IntD{\ParameterSet}{
\log\det(\TNIMatrix(\DesignMeasure, \Parameter))
}{\PriorDensity(\Parameter)}{\Parameter}
.
```

## Efficiency

Consider two experiments with design measures ``\DesignMeasure_1`` and ``\DesignMeasure_2``,
and samples sizes ``\SampleSize^{(1)}`` and ``\SampleSize^{(2)}``.
Equating their approximate expected posterior Shannon information
and solving for the ratio of sample sizes yields

```math
\frac{n^{(2)}}{n^{(1)}}
=
\exp\biggl(
\frac{1}{\DimTransformedParameter}
\IntD{\ParameterSet}{
\log\frac{\det \TNIMatrix(\DesignMeasure_1, \Parameter)}{\det \TNIMatrix(\DesignMeasure_2, \Parameter)}
}{\PriorDensity(\Parameter)}{\Parameter}
\biggr)
=:
\RelEff(\DesignMeasure_1, \DesignMeasure_2)
.
```

This quantity is called the _expected relative D-efficiency_
of the two designs.
A relative efficiency smaller than ``1`` means
that an experiment with ``\DesignMeasure_1`` on average needs more observations
than an experiment with ``\DesignMeasure_2`` to achieve the same accuracy.
In this sense, ``\DesignMeasure_1`` is less efficient than ``\DesignMeasure_2``.
Conversely, a relative efficiency larger than ``1`` means
that ``\DesignMeasure_1`` is more efficient than ``\DesignMeasure_2``.
When ``\DesignMeasure_2`` is optimal for the design problem,
relative efficiency is bounded from above by ``1``.

In the special case of locally optimal design,
where the prior distribution is a point mass at ``\Parameter_0``,
this simplifies to

```math
\RelEff(\DesignMeasure_1, \DesignMeasure_2)
=
\biggl(
\frac{\det \TNIMatrix(\DesignMeasure_1, \Parameter_0)}{\det \TNIMatrix(\DesignMeasure_2, \Parameter_0)}
\biggr)^{1/\DimTransformedParameter}
.
```

Note that it is also possible to compute a relative efficiency under two different design problems

```math
(
  \DesignCriterion_D,
  \DesignRegion^{(1)},
  \MeanFunction^{(1)},
  \CovariateParameterization^{(1)},
  \PriorDensity^{(1)},
  \Transformation^{(1)}
)
\text{ and }
(
  \DesignCriterion_D,
  \DesignRegion^{(2)},
  \MeanFunction^{(2)},
  \CovariateParameterization^{(2)},
  \PriorDensity^{(2)},
  \Transformation^{(2)}
).
```

The only requirement is that both ``\Transformation^{(1)}`` and ``\Transformation^{(2)}``
map into spaces with a common dimension ``\DimTransformedParameter``.
In this general case, the two integrals must be calculated separately:

```math
\begin{aligned}
\RelEff(\DesignMeasure_1, \DesignMeasure_2)
&=
\exp\biggl(
\frac{1}{\DimTransformedParameter}\biggl\{
\IntD{\ParameterSet^{(1)}}{
\log\det \TNIMatrix^{(1)}(\DesignMeasure_1, \Parameter^{(1)})
}{\PriorDensity^{(1)}(\Parameter^{(1)})}{\Parameter^{(1)}} \\
&\qquad
-
\IntD{\ParameterSet^{(2)}}{
\log\det \TNIMatrix^{(2)}(\DesignMeasure_2, \Parameter^{(2)})
}{\PriorDensity^{(2)}(\Parameter^{(2)})}{\Parameter^{(2)}}
\biggr\}
\biggr)
\end{aligned}
```

This more general quantity of course now also depends on the potentially different prior densities!
