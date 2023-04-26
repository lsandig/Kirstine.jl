# A Simple Dose-Response Model

Suppose we want to investigate a [dose-response relationship](https://en.wikipedia.org/wiki/Dose%E2%80%93response_relationship)
between the some drug and some clinical or toxicological outcome.
A commonly used statistical model in this context is the _sigmoid Emax_ model,
(sometimes also called the four-parameter log-logistic, or 4pLL model):

```math
y_i \overset{\mathrm{iid}}{\sim} \mathrm{Normal}(\mu(x_k, \theta), \sigma^2) \quad\text{for all } i \in I_k, k = 1,\dots, K
```

There are ``K`` different dose levels ``x_1,\dots,x_k``,
and the indices of the observations ``y_1,\dots,y_n`` are grouped into sets corresponding to identical doses.
The expected response is

```math
\mu(x, \theta) = E_0 + E_{\max} \frac{x^h}{\mathrm{ED}_{50}^h + x^h}
```

with a four-element parameter vector
``\theta = (E_0, E_{\max}, \mathrm{ED}_{50}, h)``.

This is a plot of the expected response function for some arbitrary ``\theta``:
```@example
using Plots
θ = (e0 = 1, emax = 2, ed50 = 4, h = 5)
plot(x -> θ.e0 + θ.emax * x^θ.h / (θ.ed50^θ.h + x^θ.h);
     xlims = (0, 10), xguide = "dose", yguide = "response", label = "μ(dose, θ)")
savefig("getting-started-sigemax.png"); nothing # hide
```

![](getting-started-sigemax.png)

We can visually confirm that half the maximum effect above baseline is attained at a dose of ``x=4``.

The goal of optimal experimental design is to find dose levels ``x_1,\dots,x_K``
so that the observations ``y_1,\dots,y_n`` will be maximally informative about the unknown ``\theta``.
In the following sections, "maximally informative" will mean D-optimal.

## Setup

Before we can search for an optimal design,
we have to specify the model.

First we need to create new subtypes of [`NonlinearRegression`](@ref) and [`Covariate`](@ref),
and specify several helper functions.
For simple cases such as this one,
where each unit of observation ``y_i`` is a single number,
this can be conveniently done with the [`@define_scalar_unit_model`](@ref) macro.

```@example main
using Kirstine
@define_scalar_unit_model Kirstine SigEmax dose
```

This declares a `SigEmax` type and a `SigEmaxCovariate` type.
To construct a `SigEmax` object,
we have to supply the inverse of the model variance, ``1/\sigma^2``.
To construct a `SigEmaxCovariate` object,
we have to supply a value for its `dose` field.
For example,
a model with ``\sigma^2 = 4`` and a covariate wrapping a dose ``x=2.5`` would be specified like this:

```@example main
example_model = SigEmax(1 / 2^2)
example_covariate = SigEmaxCovariate(2.5)
print(example_covariate.dose)
```

Next, we need to define the
[Jacobian matrix](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant)
of the expected response function ``\mu(x, \theta)``
with respect to the parameter vector ``\theta``.
Calculating manually,
or using a computer algebra system such as [Maxima](https://maxima.sourceforge.io/)
or [WolframAlpha](https://www.wolframalpha.com/input?i=Jacobian+matrix++a+%2B+b+x%5Eh+%2F%28c%5Eh+%2B+x%5Eh%29+w.r.t.+a%2C+b%2C+c%2C+h),
we find

```math
\begin{aligned}
\partial_{E_0} \mu(x, \theta)              &= 1 \\
\partial_{E_{\max}} \mu(x, \theta)         &= \frac{x^h}{\mathrm{ED}_{50}^h + x^h}\\
\partial_{\mathrm{ED}_{50}} \mu(x, \theta) &= \frac{h E_{\max} x^h\mathrm{ED}_{50}^{h-1}}{(\mathrm{ED}_{50}^h + x^h)^2} \\
\partial_{h} \mu(x, \theta)                &= \frac{E_{\max} x^h \mathrm{ED}_{50}^h (\log(x / \mathrm{ED}_{50}))}{(\mathrm{ED}_{50}^h + x^h)^2} \\
\end{aligned}
```

for ``x \neq 0``, and the limit

```math
\lim_{x\to0} \partial_{h} \mu(x, \theta) = 0.
```

This is translated into Julia code
by defining a method for the package internal `jacobianmatrix!` function
that specializes its arguments for our `SigEmax` and `SigEmaxCovariate` types.
Its first argument is a pre-allocated `Matrix` of the correct dimension (`(1, 4)` in our case),
the elements of which we have to overwrite with the partial derivatives.

```@example main
# p must be a NamedTuple with elements `e0`, `emax` `ed50`, `h`
function Kirstine.jacobianmatrix!(jm, m::SigEmax, c::SigEmaxCovariate, p)
    dose_pow_h = c.dose^p.h
    ed50_pow_h = p.ed50^p.h
    A = dose_pow_h / (dose_pow_h + ed50_pow_h)
    B = ed50_pow_h * p.emax / (dose_pow_h + ed50_pow_h)
    jm[1, 1] = 1.0
    jm[1, 2] = A
    jm[1, 3] = - A * B * p.h / p.ed50
    jm[1, 4] = c.dose == 0 ? 0.0 : A * B * log(c.dose / p.ed50)
    return jm
end
```
Here we have tried to calculate expensive expressions only once
and we have reused intermediate values.
Also note that we follow [the conventions](https://docs.julialang.org/en/v1/manual/style-guide/#bang-convention)
and also return the modified argument `jm`.

Finally we need to specify how a design point maps to a `SigEmaxCovariate`.
In our simple case,
the design space will be the interval of possible doses,
hence the covariate value is identical to the corresponding design point.

In Julia we achieve this by first declaring a subtype of [`CovariateParameterization`](@ref),
which in this case has no fields and is only used for dispatch.

```@example main
struct CopyDose <: CovariateParameterization end
```

Then we specialize the internal method `update_model_covariate!` for our model and covariate type,
and the parameterization that we have just declared.
The design point `dp` here is a one-element `Vector`,
and we just need to copy its first element to the `dose` field of the covariate.

```@example main
function Kirstine.update_model_covariate!(c::SigEmaxCovariate, dp, m::SigEmax, cp::CopyDose)
    c.dose = dp[1]
    return c
end
```

## Locally Optimal Design

In this section we look at locally D-optimal designs,
where we specify a single "best guess" at the unknown parameter ``\theta``.

This is how we specify a design problem with
the design space ``[0, 10]``,
``\sigma^2=1``,
and our prior guess from above.
```@example main
dc = DOptimality()
ds = DesignSpace(:dose => (0, 10))
mod = SigEmax(1)
cpar = CopyDose()
guess = PriorGuess((e0 = 1, emax = 2, ed50 = 4, h = 5))
trafo = Identity()
na = MLApproximation()
nothing # hide
```
There are a couple of things to note here:
- In this case, the observation variance ``\sigma^2`` merely scales the objective function,
  hence it has no influence on the solution.
  So for simplicity, we set it to `1`.
- The name of the design space's single dimension is given by the symbol `:dose`,
  and it can be chosen independently from however we have named the field of our `SigEmaxCovariate`.
  They _do not_ have to be the same.
- The argument to [`PriorGuess`](@ref) is a [`NamedTuple`](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple),
  and its names correspond to those that we have used in `jacobianmatrix!`.
- `trafo = Identity()` simply means that we are interested in all elements of the parameter as they are.
- With choosing an [`MLApproximation`](@ref) we say
  that we only want to use the likelihood for the approximation of the posterior information matrix,
  without including any additional regularization.

We will use [particle swarm optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) to do the actual work:

```@example main
pso = Pso(iterations = 50, swarmsize = 25)
nothing # hide
```

Now we can call [`optimize_design`](@ref):
```@example main
import Random
Random.seed!(4711)
s1, r1 = optimize_design(pso, dc, ds, mod, cpar, guess, trafo, na)
nothing # hide
```
It returns two objects:
- `s1` is the best [`DesignMeasure`](@ref) that was found,
- `r1` is an [`OptimizationResult`](@ref) with additional diagnostic information.

Let's first examine the solution:

```@example main
s1
```

This shows the design measure as pairs of a design point and a corresponding weight.
Note that the representation above can be pasted into a Julia REPL.
Sometimes a matrix representation is cleaner.
Here, the weights are in the first row.

```@example main
as_matrix(s1)
```

In order to see whether `s1` actually is the solution,
we can visually verify the equivalence theorem:
the Gateaux derivatives at `s1` into each direction from (a grid on) the design space
should be non-positive.

```@example main
using Plots
plot_gateauxderivative(dc, s1, ds, mod, cpar, guess, trafo, na; legend = :outerright)
savefig(ans, "getting-started-pg1.png"); nothing # hide
```

![](getting-started-pg1.png)

Looks good.

Now let's turn to the diagnostic information.
By plotting `r1`, we can see how the solution has improved over successive iterations.

```@example main
plot(r1)
savefig(ans, "getting-started-pd1.png"); nothing # hide
```

![](getting-started-pd1.png)

We see that the particle swarm has effectively converged after 20 iterations.

But let's return to the solution for a minute.
```@example main
s1
```

We see that the weights are nearly uniform,
and that the lowest and highest doses are near the boundaries of the design space.
This is not by accident:
one can show analytically that locally D-optimal designs for the sigmoid Emax model
always have four design points with uniform weights,
and that two of them are at the minimal and maximal dose.[^LM07]
We can pass this information to [`optimize_design`](@ref):

[^LM07]: Gang Li and Dibyen Majumdar (2008). D-optimal designs for logistic models with three and four parameters. Journal of Statistical Planning and Inference, 138(7), 1950–1959. [doi:10.1016/j.jspi.2007.07.010](http://dx.doi.org/10.1016/j.jspi.2007.07.010)

```@example main
Random.seed!(4711)
s2, r2 = optimize_design(pso, dc, ds, mod, cpar, guess, trafo, na;
                         candidate = grid_design(ds, 4),
                         fixedweights = 1:4, fixedpoints = [1, 4])
nothing # hide
```

Here we first specify a `candiate`,
i.e. our initial guess at a potential solution:
4 equally spaced doses on the design space `ds`, with uniform weights.

```@example main
grid_design(ds, 4)
```

Then we tell `optimize_design` to keep fixed all of the weights (`fixedweights = 1:4`),
as well as the first and last design point (`fixedpoints = [1, 4]`).
This time the solution looks a bit cleaner:

```@example main
s2
```

However, the solutions differ more in terms of aesthetics than in terms of performance,
as their relative [`efficiency`](@ref) clearly shows:

```@example main
efficiency(s1, s2, mod, cpar, guess, trafo, na)
```

## Bayesian Optimal Design

For a Bayesian D-optimal design,
we replace the single guess for ``\theta`` by a full prior distribution.
Two kinds of priors are supported:
- A [`DiscretePrior`](@ref) represents a set of parameter values and corresponding weights.
  This is useful if only a few parameter values are thought to be possible.
- A [`PriorSample`](@ref) represents a draw from a prior distribution.
  This is useful with draws obtained by MCMC from a model fitted to data of a pilot experiment.

### Discrete Prior

Let's first look at an example with a [`DiscretePrior`](@ref).
The slope parameter ``h`` of the sigmoid Emax model can in some situations be interpreted
as the number of molecules that need to bind to a receptor in order to produce an effect.[^W97]
Let's assume that we suspect only ``h\in\{1,2,3,4\}`` are possible,
with prior probabilities ``\{0.1, 0.3, 0.4, 0.2\}``.
The remaining elements of ``\theta`` are as in the previous section.

[^W97]: James N. Weiss (1997). The hill equation revisited: uses and misuses. The FASEB Journal, 11(11), 835–841. [doi:10.1096/fasebj.11.11.9285481](http://dx.doi.org/10.1096/fasebj.11.11.9285481)

```@example main
dpr = DiscretePrior([0.1, 0.3, 0.4, 0.2],
                    [(e0 = 1, emax = 2, ed50 = 4, h = h) for h in 1:4])
nothing # hide
```

For Bayesian optimal designs, the number of design points is seldom known in advance.
This is why we start with an initial 10-point grid design,
and also increase the number of iterations and the swarm size.

```@example main
pso = Pso(iterations = 100, swarmsize = 50)
Random.seed!(31415)
s3, r3 = optimize_design(pso, dc, ds, mod, cpar, dpr, trafo, na; candidate = grid_design(ds, 10))
plot_gateauxderivative(dc, s3, ds, mod, cpar, dpr, trafo, na)
savefig(ans, "getting-started-pg3.png"); nothing # hide
```

![](getting-started-pg3.png)

But what's going on here?
Why are only 9 listed in the legend, and just 6 plotted in the figure?
And what about the point at the point at 5.5 which is not on the graph?
All of this is because the optimization procedure does not enforce unique design points,
and weights are allowed to become 0.

Before returning the solution found by the PSO algorithm,
[`optimize_design`](@ref) calls [`simplify`](@ref) and [`sort_designpoints`](@ref) on it.
In order to be conservative,
the default keyword arguments to [`simplify`](@ref) are
`mindist=0` and `minweight=0`,
meaning only identical design points are merged
and only points with zero weight are dropped.
This is why in our example, `s3` has only 9 design points.

```@example main
s3
```

We also see that the pairs of design points
at about `2.56`, at about `4.95`, and at about `10.0`
are very similar,
and have been plotted over each other in the previous figure.
The design point at `5.62` has negligible weight.
By setting `minweight=1e-4` and `mindist=1e-3`, we can simplify the result more aggressively.

```@example main
Random.seed!(31415)
s4, r4 = optimize_design(pso, dc, ds, mod, cpar, dpr, trafo, na;
                         candidate = grid_design(ds, 10), minweight = 1e-4, mindist = 1e-3);
plot_gateauxderivative(dc, s4, ds, mod, cpar, dpr, trafo, na)
savefig(ans, "getting-started-pg4.png"); nothing # hide
```

![](getting-started-pg4.png)

```@example main
s4
```

Note that we can still access the raw (unsorted and unsimplified) result at `r4.maximizer`:

```@example main
r4.maximizer
```

Now suppose we want to perform the experiment,
and our budget allows for a total of ``n=30`` measurements.
The [`apportion`](@ref) method implements the _efficient apportionment procedure_,
which returns a vector of integers.

```@example main
apportion(s4, 30)
```

### Prior Sample

Next we look at how to use a [`PriorSample`](@ref).
We generate 1000 independent draws for each element of ``\theta`` from a normal distribution.
We enforce some lower bounds on to prevent singular information matrices.
(Using [stan](https://mc-stan.org) to draw a real prior sample from (simulated) pilot data is out of scope for this introduction.)

```@example main
Random.seed!(31415)
sample_mat = max.([0 0.1 1 1], [1 2 4 5] .+ [0.5 0.5 0.5 0.5] .* randn(1000, 4))
sample = [(e0 = a, emax = b, ed50 = c, h = d) for (a, b, c, d) in eachrow(sample_mat)];
mcpr = PriorSample(sample)
nothing # hide
```

The optimization takes notably longer this time.
We see not much improvement after 15 iterations,
yet `s5` is still far from the solution.

```@example main
Random.seed!(31415)
pso = Pso(iterations = 25, swarmsize = 50)
s5, r5 = optimize_design(pso, dc, ds, mod, cpar, mcpr, trafo, na;
                         candidate = grid_design(ds, 6), fixedpoints = [1, 6],
                         minweight = 1e-4, mindist = 1e-3)
plot(plot(r5),
     plot_gateauxderivative(dc, s5, ds, mod, cpar, mcpr, trafo, na))
savefig(ans, "getting-started-pg5-pd5.png") ; nothing # hide
```

![](getting-started-pg5-pd5.png)

We can use [`refine_design`](@ref) to try to improve the design.
It repeats the following four actions for a given number of steps:
1. Simplify the design
2. Search for the direction with the largest Gateaux derivative.
3. Add the corresponding design point to the design.
4. Re-optimize the weights.

Separate PSO parameters can be given for (2) and (4).
These optimization problems have lower dimensions,
so a smaller swarm and number of iterations is often sufficient.
Now we use 5 refinement iterations:

```@example main
psod = Pso(iterations = 10, swarmsize = 50)
psow = Pso(iterations = 15, swarmsize = 25)
Random.seed!(31415)
s6, r6d, r6w = refine_design(psod, psow, 5, s5, dc, ds, mod, cpar, mcpr, trafo, na)
plot(plot(r6w),
     plot_gateauxderivative(dc, s6, ds, mod, cpar, mcpr, trafo, na))
savefig(ans, "getting-started-pg6-pd6.png") ; nothing # hide
```

![](getting-started-pg6-pd6.png)

This already looks better, but we're not quite there yet.
Note that the refinement steps have added a lot of low-weight design points between 2 and 5.
A final round of regular PSO can help nudging these design points around,
so that they can be merged more easily.

```@example main
pso = Pso(iterations = 150, swarmsize = 50)
Random.seed!(31415)
s7, r7 = optimize_design(pso, dc, ds, mod, cpar, mcpr, trafo, na; candidate = s6,
                         minweight = 1e-4, mindist = 5e-3,)
plot(plot(r7),
     plot_gateauxderivative(dc, s7, ds, mod, cpar, mcpr, trafo, na))
savefig(ans, "getting-started-pg7-pd7.png") ; nothing # hide
```

![](getting-started-pg7-pd7.png)

This time, the efficiency gains are greater,
especially when going from `s5` to `s6`.

```@example main
efficiency(s6, s5, mod, cpar, mcpr, trafo, na)
```

```@example main
efficiency(s7, s6, mod, cpar, mcpr, trafo, na)
```
