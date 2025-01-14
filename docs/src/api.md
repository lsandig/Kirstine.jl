# API Reference

## Design Problems

### Specifying and Composing Design Problems

```@docs
Kirstine.AbstractDesignProblem
DesignProblem
DesignProblem()
DesignProblem(::DesignProblem)
criterion
region(::DesignProblem)
model
covariate_parameterization
prior_knowledge
transformation
normal_approximation
```

Several `AbstractDesignProblems` can be combined to form a composite design problem,
where a weighted sum of the component objectives is maximized.
Composite design problems can be nested within each other.

```@docs
CompositeDesignProblem
CompositeDesignProblem(::AbstractVector{<:Kirstine.AbstractDesignProblem}, ::AbstractVector{<:Real})
problems(::CompositeDesignProblem)
weights(::CompositeDesignProblem)
region(::CompositeDesignProblem)
```

### Solving Design Problems

Different high-level strategies are available for trying to solve any `AbstractDesignProblem`.

```@docs
solve
Kirstine.ProblemSolvingStrategy
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
and are used as part of a `Kirstine.ProblemSolvingStrategy`.

```@docs
Kirstine.Optimizer
Kirstine.ZerothOrderOptimizer
Kirstine.FirstOrderOptimizer
Kirstine.OptimizationResult
Pso
Pso()
GridSearch
GridSearch()
GridSpec
ProductGrid
Kirstine.FrankWolfe
Kirstine.FrankWolfe(::Integer, ::Kirstine.StepSizeStrategy, ::Bool)
Kirstine.StepSizeStrategy
Kirstine.OpenLoop
Kirstine.ConstNumIter
Kirstine.Adaptive
```

## Design Criteria

```@docs
Kirstine.DesignCriterion
DCriterion
ACriterion
objective
gateauxderivative
efficiency
shannon_information
```

## Design Regions

```@docs
Kirstine.DesignRegion
DesignInterval
DesignInterval(::Any, ::Any, ::Any)
DesignInterval()
dimension(::Kirstine.DesignRegion)
dimnames
upperbound
lowerbound
```

## Nonlinear Regression Models

Regression models are implemented
by first subtyping `Kirstine.Model`, `Kirstine.Covariate`, `Kirstine.CovariateParameterization`, and `Kirstine.Parameter`,
and then defining several helper methods on them.
See the [tutorial](tutorial.md) for a hands-on example.

### Supertypes

```@docs
Kirstine.Model
Kirstine.NonlinearRegression
Kirstine.Covariate
Kirstine.CovariateParameterization
Kirstine.Parameter
```

### Implementing a Nonlinear Regression Model

Suppose the following subtypes have been defined:

  - `M <: Kirstine.NonlinearRegression`
  - `C <: Kirstine.Covariate`
  - `Cp <: Kirstine.CovariateParameterization`
  - `P <: Kirstine.Parameter`

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
    
    Some of these methods only require `M <: Kirstine.Model` and not `M <: Kirstine.NonlinearRegression`.
    This is intentional.
    These methods can in principle also be implemented for `M <: Ma`
    where `abstract type Ma <: Kirstine.Model end` is also user-defined.

### Helpers

To reduce boilerplate code in the common cases of a one-dimensional unit of observation,
and for a vector parameter without any additional structure,
the following helper macros can be used:

```@docs
@simple_model
@simple_parameter
```

In problems where the design variables coincide with the model covariates,
the following `Kirstine.CovariateParameterization` can be used:

```@docs
JustCopy
JustCopy()
```

## Prior Knowledge

```@docs
Kirstine.PriorKnowledge
PriorSample
PriorSample(::AbstractVector{T}, ::AbstractVector{<:Real}) where T <: Kirstine.Parameter
Base.:(==)(::PriorSample, ::PriorSample)
parameters(::PriorSample)
weights(::PriorSample)
```

## Transformations

```@docs
Kirstine.Transformation
Identity
DeltaMethod
```

## Normal Approximations

```@docs
Kirstine.NormalApproximation
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
weights(::DesignMeasure)
numpoints
sort_points
sort_weights
mixture
apportion
simplify(::DesignMeasure, ::DesignProblem)
simplify(::DesignMeasure, ::CompositeDesignProblem)
simplify_drop
simplify_unique
simplify_merge
informationmatrix
apply_transformation
implied_covariates
```

## Plotting

Kirstine.jl provides [plot recipes](https://docs.juliaplots.org/latest/recipes/)
for several (combinations of) types,
instances of which can simply be `plot()`ted.
For Gateaux derivatives and expected functions there are separate methods.

```julia
plot(d::DesignMeasure)
plot(d::DesignMeasure, dr::Kirstine.DesignRegion)
plot(r::Kirstine.OptimizationResult)
plot(rs::AbstractVector{<:OptimizationResult})
plot(r::DirectMaximizationResult)
plot(r::ExchangeResult)
```

The method for plotting `DesignMeasure`s has several options.

  - When a `DesignRegion` is given,
    `xlims` (and for 2-dimensional design points also `ylims`) is set to the boundaries of the region.
  - The `label_formatter` keyword argument accepts a function
    that maps a triple `(k, pt, w)` of an index, design point, and weight to a string.
    By default, the weight is indicated in percentage points rounded to 3 significant digits.
    Note that due to rounding the weights do not necessarily add up to 100%.
    To achieve this,
    consider plotting an [`apportion`](@ref)ed version of your design measure like this:
    `DesignMeasure(points(d), apportion(weights(d), 1000) ./ 1000)`
  - The `maxmarkersize` keyword argument indicates the `markersize` (radius in pixels)
    of a design point with weight 1.
    Other points are scaled down according to the `sqrt` of their weight.
    It will be overridden by an explicitly given `markersize`.
    The default is `maxmarkersize=10`.
  - The `minmarkersize` keyword argument gives the absolute lower bound for the `markersize` of a point.
    This prevents points with very little weight from becoming illegible.
    It will be overridden by an explicitly given `markersize`.
    The default is `minmarkersize=2`.

In all cases,
standard [plotting attributes](https://docs.juliaplots.org/latest/attributes/) will be passed through.

The next plots illustrate these options:

```@example plot
using Kirstine, Plots
dr = DesignInterval(:a => (0, 1), :b => (-2.5, 3))
d = DesignMeasure(
    [0.5, -1] => 0.1,
    [0.2, 2] => 0.2,
    [0.9, -2] => 0.2,
    [0.1, -0.1] => 0.05,
    [0.3, 0.4] => 0.45,
)
p1 = plot(d, title = "without region")
p2 = plot(d, dr, title = "with region")
p3 = plot(
    d,
    dr,
    markercolor = :orange,
    markershape = [:star :diamond :hexagon :x :+],
    grid = nothing,
    legend = :outerright,
    title = "standard attributes",
)
pa = plot(p1, p2, p3)
savefig(pa, "api-pa.png") # hide
nothing # hide
```

![](api-pa.png)

```@example plot
p5 = plot(d, dr; maxmarkersize = 20, title = "maxmarkersize 20")
p6 = plot(d, dr; minmarkersize = 4, title = "minmarkersize 4")
p7 = plot(d, dr; markersize = 5, title = "markersize 5")
p8 = plot(d, dr; label_formatter = (k, p, w) -> "$k: a = $(p[1])", title = "custom labels")
pb = plot(p5, p6, p7, p8)
savefig(pb, "api-pb.png") # hide
nothing # hide
```

![](api-pb.png)

Since the design points are plotted in the given order,
large points may be plotted over small ones.
Unfortunately, the z-order of points can not be set independently in Plots.jl.
Possible workarounds are sorting the points by decreasing weight before plotting,
or choosing an unfilled marker shape.

```julia
d = DesignMeasure([0 1 1.01 2], [0.1, 0.1, 0.7, 0.1])
plot(d) # point at 1 not visible
plot(sort_weights(d; rev = true)) # point at 1 visible, but legend re-ordered
plot(d; markershape = :+) # all visible, but colors harder to distinguish
```

```@docs
plot_gateauxderivative
plot_expected_function
```
