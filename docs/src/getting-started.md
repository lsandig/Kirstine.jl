# A Simple Dose-Response Model

```@setup main
# we can't do the `savefig(); nothing # hide` trick when using JuliaFormatter
function savefig_nothing(plot, filename)
	savefig(plot, filename)
	return nothing
end
```

This vignette describes how to set up a simple non-linear regression model
in order to find a Bayesian D-optimal design.

## Model

Suppose we want to investigate the [dose-response relationship](https://en.wikipedia.org/wiki/Dose%E2%80%93response_relationship)
between a drug and some clinical outcome.
A commonly used model for this task is the _sigmoid Emax_ model:
an s-shaped curve with a parameter for controlling the steepness of the slope.
Assuming independent measurement errors,
the corresponding regression model for a total of ``n`` observations
at ``K`` different dose levels ``x_1,\dots,x_k`` is

```math
y_i \overset{\mathrm{iid}}{\sim} \mathrm{Normal}(\mu(x_k, θ), \sigma^2) \quad\text{for all } i \in I_k, k = 1,\dots, K,
```

with expected response at dose ``x``

```math
\mu(x, θ) = E_0 + E_{\max} \frac{x^h}{\mathrm{ED}_{50}^h + x^h}
```

and a four-element parameter vector
``θ = (E_0, E_{\max}, \mathrm{ED}_{50}, h)``.

```@example main
using Plots
θ = (e0 = 1, emax = 2, ed50 = 4, h = 5)
plot(
    x -> θ.e0 + θ.emax * x^θ.h / (θ.ed50^θ.h + x^θ.h);
    xlims = (0, 10),
    xguide = "dose",
    yguide = "response",
    label = "μ(dose, θ)",
)
savefig_nothing(ans, "getting-started-sigemax.png") # hide
```

![](getting-started-sigemax.png)

In order to find an optimal design,
we need to know the
[Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
of ``\mu(x, θ)`` with respect to ``θ``.
Calculating manually,
or using a computer algebra system such as [Maxima](https://maxima.sourceforge.io/)
or [WolframAlpha](https://www.wolframalpha.com/input?i=Jacobian+matrix++a+%2B+b+x%5Eh+%2F%28c%5Eh+%2B+x%5Eh%29+w.r.t.+a%2C+b%2C+c%2C+h),
we find

```math
\begin{aligned}
\partial_{E_0} \mu(x, θ)              &= 1 \\
\partial_{E_{\max}} \mu(x, θ)         &= \frac{x^h}{\mathrm{ED}_{50}^h + x^h}\\
\partial_{\mathrm{ED}_{50}} \mu(x, θ) &= \frac{h E_{\max} x^h\mathrm{ED}_{50}^{h-1}}{(\mathrm{ED}_{50}^h + x^h)^2} \\
\partial_{h} \mu(x, θ)                &= \frac{E_{\max} x^h \mathrm{ED}_{50}^h (\log(x / \mathrm{ED}_{50}))}{(\mathrm{ED}_{50}^h + x^h)^2} \\
\end{aligned}
```

for ``x \neq 0``, and the limit

```math
\lim_{x\to0} \partial_{h} \mu(x, θ) = 0.
```

## Setup

### Model and Design Region

In `Kirstine.jl`, the regression model is defined by subtyping
[`NonlinearRegression`](@ref),
[`Covariate`](@ref),
[`Parameter`](@ref),
and [`CovariateParameterization`](@ref),
and implementing a handful of methods for them.

In the sigmoid Emax model,
a single unit of observations is just a real number `y_i`.
For such a case,
we can use the helper macro [`@define_scalar_unit_model`](@ref)
to declare a model type named `SigEmax`,
and a covariate type named `SigEmaxCovariate` with the single field `dose`.

```@example main
using Kirstine
@define_scalar_unit_model Kirstine SigEmax dose
```

The model parameter ``θ`` is just a vector with no additional structure,
which is why we can use the helper macro [`@define_vector_parameter`](@ref)
to define a parameter type `SigEmaxPar`
with the fields fields `e0`, `emax`, `ed50`, and `h`.

```@example main
@define_vector_parameter Kirstine SigEmaxPar e0 emax ed50 h
```

Next we implement the Jacobian matrix.
We do this by defining a method for `Kirstine.jacobianmatrix!`
that specializes on our newly defined model, covariate and parameter types.
This method will later be called from the package internals with a pre-allocated matrix `jm`
(of the correct size `(1, 4)`).
Now our job is to fill in the correct values:

```@example main
function Kirstine.jacobianmatrix!(jm, m::SigEmax, c::SigEmaxCovariate, p::SigEmaxPar)
    dose_pow_h = c.dose^p.h
    ed50_pow_h = p.ed50^p.h
    A = dose_pow_h / (dose_pow_h + ed50_pow_h)
    B = ed50_pow_h * p.emax / (dose_pow_h + ed50_pow_h)
    jm[1, 1] = 1.0
    jm[1, 2] = A
    jm[1, 3] = -A * B * p.h / p.ed50
    jm[1, 4] = c.dose == 0 ? 0.0 : A * B * log(c.dose / p.ed50)
    return jm
end
```

(We follow [the conventions](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention)
and also return the modified argument `jm`.)

Next, we set up the design region.
Let's say we want to allow doses between `0` and `1` (in some unit).

```@example main
dr = DesignInterval(:dose => (0, 1))
nothing # hide
```

Note that the name `:dose` does not necessarily have to match the field name of the `SigEmaxCovariate`.
A design will then be a discrete probability measure with atoms (design points) from `ds`,
and a design point is represented by a `Vector{Float64}`.
In our simple model, this vector has length `1`.

Finally, we need to specify how a design point maps to a `SigEmaxCovariate`.
Here, the design region is simply the interval of possible doses.
This means we can just copy the only element of the design point
into the covariate's `dose` field.
To do this, we subtype [`CovariateParameterization`](@ref)
and define a method for `Kirstine.update_model_covariate!`.

```@example main
struct CopyDose <: CovariateParameterization end

function Kirstine.update_model_covariate!(c::SigEmaxCovariate, dp, m::SigEmax, cp::CopyDose)
    c.dose = dp[1]
    return c
end
```

### Prior Knowledge

For Bayesian optimal design of experiments we need to specify a prior distribution on ``θ``.
`Kirstine.jl` then needs a sample from this distribution.

For this example we use independent normal priors on the elements of ``θ``.
We first draw a vector of `SigEmaxPar`s
which we then wrap into a [`PriorSample`](@ref).

```@example main
using Random
Random.seed!(31415)

theta_mean = [1, 2, 0.4, 5]
theta_sd = [0.5, 0.5, 0.05, 0.5]
sep_draws = map(1:1000) do i
    rnd = theta_mean .+ theta_sd .* randn(4)
    return SigEmaxPar(e0 = rnd[1], emax = rnd[2], ed50 = rnd[3], h = rnd[4])
end
prior_sample = PriorSample(sep_draws)
nothing # hide
```

Note that the `SigEmaxPar` constructor takes keyword arguments.

### Design Problem

Now we collect all the parts in a [`DesignProblem`](@ref).

```@example main
dp = DesignProblem(
    design_criterion = DOptimality(),
    design_region = dr,
    model = SigEmax(1),
    covariate_parameterization = CopyDose(),
    prior_knowledge = prior_sample,
)
nothing # hide
```

Note that the `SigEmax` constructor takes the inverse variance ``1/σ^2`` as an argument.
For D-optimality, this only scales the objective function and has no influence on the optimal design.
This is why we simply set it to `1` here.

## Optimal Design

`Kirstine.jl` provides several high-level [`ProblemSolvingStrategy`](@ref)s
for solving the design problem `dp`.
A very simple, but often quite effective one,
is direct maximization with a [particle swarm optimizer](https://en.wikipedia.org/wiki/Particle_swarm_optimization).

Apart from the [`Pso`](@ref) parameters,
the [`DirectMaximization`](@ref) strategy needs to know
how many design points the design measures it searches over should have.
This is indirectly specified by the `prototype` argument:
it takes a [`DesignMeasure`](@ref) which is then used for initializing the swarm.
One particle is set exactly to `prototype`.
The remaining ones have the same number of design points,
but their weights and design points are completely randomized.

For our example,
let's use a `prototype` with `8` equally spaced design points.

```@example main
strategy = DirectMaximization(
    optimizer = Pso(iterations = 50, swarmsize = 100),
    prototype = equidistant_design(dr, 8),
)

Random.seed!(31415)
s1, r1 = solve(dp, strategy)
nothing # hide
```

The [`solve`](@ref) function returns two objects:

  - `s1`: a slightly post-processed [`DesignMeasure`](@ref)
  - `r1`: a [`DirectMaximizationResult`](@ref)
    that contains additional information about the optimization run.

We can display the solution as `designpoint => weight` pairs,
or as a matrix with the weights in the first row.

```@example main
s1
```

```@example main
as_matrix(s1)
```

Looking closely at `s1`,
we notice that three design points are nearly identical:

```julia
[0.4720457753718496] => 0.0003693803605209383
[0.4994069801758983] => 0.07074703398360908
[0.5005657108099189] => 0.15038996534460683
```

It seems plausible that they would merge into a single one
if we ran the optimizer for some more iterations.
But we can also do this after the fact by calling [`simplify`](@ref) on the solution.
This way we remove all design points with negligible weight,
and merge all design points that are less than some minimum distance apart.

```@example main
s2 = simplify(s1, dp, minweight = 1e-3, mindist = 1e-2)
as_matrix(s2)
```

Because this issue occurs frequently
we can directly pass the simplification arguments to `solve`:

```julia
s2, r2 = solve(dp, strategy; minweight = 1e-3, mindist = 1e-2)
```

The original, unsimplified solution can still be accessed at `maximizer(r2)`.

In order to confirm that we have actually found the solution
we visually check that the Gateaux derivative is non-positive over the whole design region.

```@example main
gd = plot_gateauxderivative(s2, dp)
savefig_nothing(gd, "getting-started-gd.png") # hide
```

![](getting-started-gd.png)

Finally, we can also look at a plot of the optimization progress
and see that the particle swarm has converged already after about `20` iterations.

```@example main
pr1 = plot(r1)
savefig_nothing(pr1, "getting-started-pr1.png") # hide
```

![](getting-started-pr1.png)

## Relative Efficiency and Apportionment

In order to be able to estimate ``θ`` at all,
we need a design with at least `4` points.
Let's compare how much better the optimal solution `s2` is
than a 4-point [`equidistant_design`](@ref).

```@example main
efficiency(equidistant_design(dr, 4), s2, dp)
```

Their relative D-[`efficiency`](@ref) means
that `s2` on average only needs `0.72` as many observations as the equidistant design
in order to achieve the same estimation accuracy.

Suppose we now actually want to run the experiment on ``n=42`` units.
For this we need to convert the weights of `s1` to integers
that add up to ``n``.
This is achieved with the [`apportion`](@ref) function:

```@example main
apportion(s2, 42)
```

This tells us to take `9` measurements at the first design point,
`10` at the second,
`2` at the third,
and so on.
