# Frequently and not so Frequently Asked Questions

```@contents
Pages=["faq.md"]
Depth=2:3
```

## General

### How fast is it?

There are no rigorous benchmarks against other [experimental design packages](https://cran.r-project.org/view=ExperimentalDesign) yet.
Anecdotally,
Kirstine.jl is between 10 and 100 times faster
than comparable R implementations.

### Why the name?

Kirstine Smith (1878-1939) was a Danish statistician, hydrographer and teacher.
Her [1918 article](https://dx.doi.org/10.1093/biomet/12.1-2.1)
_On the standard deviations of adjusted and interpolated values of an observed polynomial function and its constants and the guidance they give towards a proper choice of the distribution of observations_
can be considered the first treatment of experimental design as a problem in statistics.
You can find a longer [biography at the internet archive](https://web.archive.org/web/20090810040534/http://www.webdoe.cc/publications/kirstine.php).

### How can I contribute?

See the [contributing section](index.md#Contributing) on the main page.

Further development of Kirstine.jl will focus on the following topics.
Feedback, ideas, and discussion on them are highly appreciated.

  - *Abstract Integration.*
    Investigate whether the internals can reworked
    to support additional numerical integration methods like HCubature.
  - *Benchmark.*
    How does Kirstine.jl compare to other
    [experimental design packages](https://cran.r-project.org/view=ExperimentalDesign)?
    This should take into account
    that they use different optimization algorithms.
  - *Composed Design Problems.*
    Figure out an elegant way to support more abstract design problems
    that are themselves built from several concrete design problems,
    e.g. for compound objectives or two-stage design.
  - *Diagnostics.*
    Plots for high- and low-level optimizers.
  - *Examples Repository.*
    Collect further usage examples with more complicated models
    and reproduce already published results.
  - *Usability Improvements.*
    Improve the user interface
    whenever this is possible without adding too much internal complexity
    or dependencies on packages outside the standard library.

## Supported Design Problems

### Can I use other regression models?

Yes, with a bit of work.
[Generalized linear models](https://en.wikipedia.org/wiki/Generalized_linear_model) (GLMs)
or [multilevel models](https://en.wikipedia.org/wiki/Multilevel_model)
are not supported out of the box.
However,
you can define your own arbitrary abstract `Model` subtype
as long as you can figure out its (approximate) Fisher information matrix.
Take a look a the [logistic regression example](extend-model.md)
to see what needs to be implemented.

*Linear regression* is a special kind of nonlinear regression,
so it is already supported implicitly.

### Can I solve compound design problems?

No.
Compound design problems,
i.e. maximizing a weighted sum

```math
\DesignMeasure ↦ \sum_{j=1}^J λ_j \Objective_j(\DesignMeasure)
```

of different [objective functions](math.md#Objective-Function),
are currently not supported.

### Can a design region have more than 2 dimensions?

Yes.
A [`DesignRegion`](@ref) can have an arbitrary number of dimensions.
However,
plotting is currently only implemented for one- and two-dimensional regions.

### Which alphabetical criteria are supported?

The [A-criterion](math.md#A-Criterion) and the [D-criterion](math.md#D-Criterion) are supported out of the box.
Together with certain transformations this gets you the following criteria “for free”:

  - *Ds-optimal* design.
    This is equivalent to the D-criterion design with a transformation
    that projects to a subset of the parameter's elements,
    i.e. a [`DeltaMethod`](@ref) where each row of the Jacobian matrix has exactly one 1
    and is 0 everywhere else.
  - *c-optimal* design.
    This is the A-criterion with transformation ``\Transformation(\Parameter)=c'\Parameter``,
    i.e. a [`DeltaMethod`](@ref) with Jacobian matrix ``c'``.
  - *L-optimal* design.
    This is the A-criterion with transformation ``\Transformation(\Parameter)=L\Parameter``,
    i.e. a [`DeltaMethod`](@ref) with Jacobian matrix ``L``.
  - *I-optimal* design.
    Supported with an A-criterion [work-around](#Are-there-different-variants-of-A-optimality?).
  - *V-optimal* design.
    Supported with an A-criterion [work-around](#Are-there-different-variants-of-A-optimality?).

Some other alphabetical criteria are not supported
because they involve nested optimization problems,
and because they do not have an equivalence theorem
that is based on the Gateaux derivative.
These include:

  - *E-optimal* design. Minimum eigenvalue of the information matrix.
  - *G-optimal* design. Maximum predictive variance across the design region.
  - *T-optimal* design. Model discrimination via residuals.

### Are there different variants of A-optimality?

Yes, and Kirstine.jl only implements one of them.
Some authors define the [objective function](math.md#Objective-Function) for locally A-optimal design as

```math
\Objective(\DesignMeasure) = -\Trace[\tilde{A}(\Parameter_0)\NIMatrix^{-1}(\DesignMeasure,\Parameter_0)]
```

with a given positive definite matrix ``\tilde{A}``
that may optionally depend on ``\Parameter``.
In Kirstine.jl,
the matrix is given implicitly by
``\tilde{A}(\Parameter) = \TotalDiff\Transformation'(\Parameter)\TotalDiff\Transformation(\Parameter)``,
the Gram product of the transformation's Jacobian matrix.
In some problems,
however,
``\tilde{A}`` does not come from a transformation,
notably in I-optimal and V-optimal design.

Here, a slightly hacky work-around is to use Kirstine.jl's [A-criterion](math.md#A-Criterion)
together with the “pseudo-transformation”

```julia
DeltaMethod(θ -> cholesky(A_tilde(θ)).U)
```

which is nonsense semantically,
but produces the desired objective function.

## Troubleshooting

### Unused keyword arguments

This warning occurs when you have not defined [`simplify_unique`](@ref) for your model,
and have called [`simplify`](@ref) with unknown additional keyword arguments.
You should check that you don't have misspelled one of the other keyword arguments.

### Objective function produces NaNs

This is probably a bug in your [`Kirstine.jacobianmatrix!`](@ref) or [`DeltaMethod`](@ref).
Check that you handle potential divisions by zero
by evaluating them at and near potential singularities.
Make sure that the [`Identity`](@ref) transformation works
before debugging the `DeltaMethod`.
You could also compare derivatives to their autodiff versions.

### Weight increases despite fixing it

With [`DirectMaximization`](@ref),
a `fixedweight` effectively just sets a lower bound.
This is because during optimization,
design points are not forced to be unique.
When the solution is simplified afterwards,
additional weight can be merged onto the initially fixed point.
