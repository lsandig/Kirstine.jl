# Optimal Design for Functions of the Parameter

Suppose we are not primarily interested in a design that maximizes the expected posterior information about the parameter ``\theta``,
but about the transformed parameter ``T(\theta)``,
where ``T(\theta)`` can be a number or a vector.

!!! note

	`Kirstine.jl` allows transformations to be freely combined with different design criteria.
	In this sense, c-optimality is just the special case of D-optimality
	combined with a transformation ``T: \mathbb{R}^r \to \mathbb{R}``.

Our example in this vignette is the three-parameter compartment model from Atkinson et al[^ACHJ93].
Used in pharmacokinetics,
it describes how the concentration of a drug in an experimental subject changes over time.
The mean function for the regression is given by
```math
\mu(x, \theta) = s (\exp(-ex) - \exp(-ax))
```
where the elements of ``\theta`` are
the absorption rate ``a``,
the elimination rate ``e``,
and a scaling factor ``s``.
The covariate ``x`` denotes the time in hours.
```@example
using Plots # hide
θ = (a = 4.298, e = 0.05884, s = 21.80)
plot(x -> θ.s * (exp(-θ.e * x) - exp(-θ.a * x));
     xlims = (0, 10), xguide = "time [h]", yguide = "response", label = "μ(time , θ)")
savefig("handling-transformations-tpc.png"); nothing # hide
```

![](handling-transformations-tpc.png)

[^ACHJ93]: Anthony C. Atkinson, Kathryn Chaloner, Agnes M. Herzberg, and June Juritz, "Optimum experimental designs for properties of a compartmental model", Biometrics, 49(2), 325–337, 1993. [doi:10.2307/2532547](http://dx.doi.org/10.2307/2532547)

## Setup

As in the [introductory example](getting-started.md),
we start by defining the (single-row) Jacobian matrix of the mean function,
and the mapping from design variables to model covariates.

```@example main
using Kirstine, Plots, Random, Statistics

@define_scalar_unit_model Kirstine TPCMod time
struct Copy <: CovariateParameterization end

function Kirstine.jacobianmatrix!(jm, m::TPCMod, c::TPCModCovariate, p)
    A = exp(-p.a * c.time)
    E = exp(-p.e * c.time)
    jm[1, 1] =  A * p.s * c.time
    jm[1, 2] = -E * p.s * c.time
    jm[1, 3] = E - A
    return m
end

function Kirstine.update_model_covariate!(c::TPCModCovariate, dp, m::TPCMod, cp::Copy)
    c.time = dp[1]
    return c
end
```

Our design space extends from `0` to `48` hours after administration of the drug.
After we instantiate a model, we have to specify our prior knowledge.

```@example main
ds = DesignSpace(:time => [0, 48])
m = TPCMod(1)
cp = Copy()
dc = DOptimality()
na = FisherMatrix()
pso = Pso(swarmsize = 50, iterations = 100)
```

In this vignette we will only look at Bayesian optimal designs
because locally optimal designs for scalar ``T(\theta)`` usually don't exist
due to their information matrices becoming singular.

!!! note

	Workarounds like generalized inverses or matrix regularization
	are currently not supported by `Kirstine.jl`.

For the prior we will use “distribution I” from[^ACHJ93],
which is constructed from two independent uniform distributions around estimates for ``a`` and ``e``,
and a point mass for ``s``.
The strength of the prior is controlled by the width of the uniform distributions.
We generate a [`DiscretePrior`](@ref) from `1000` draws.

```@example main
function draw_from_prior(n, se_factor)
    mn = [4.298, 0.05884, 21.8]
    se = [0.5, 0.005, 0]
    as = mn[1] .+ se_factor .* se[1] .* (2 .* rand(n) .- 1)
    es = mn[2] .+ se_factor .* se[2] .* (2 .* rand(n) .- 1)
    ss = mn[3] .+ se_factor .* se[3] .* (2 .* rand(n) .- 1)
    return DiscretePrior(map((a, e, s) -> (a = a, e = e, s = s), as, es, ss))
end
Random.seed!(4711)
pk = draw_from_prior(1000, 2)
nothing # hide
```

## Optimal design for estimating the parameter

In order to have a design to compare other solutions against,
we first determine the Bayesian D-optimal design for the whole parameter ``\theta``.

```@example main
t_id = Identity()
Random.seed!(1357)
s_id, r_id = optimize_design(pso, dc, ds, m, cp, pk, t_id, na)
nothing # hide
```

```@example main
s_id
```

```@example main
plot_gateauxderivative(dc, s_id, ds, m, cp, pk, t_id, na)
savefig(ans, "handling-transformations-gd-id.png"); nothing # hide
```

![](handling-transformations-gd-id.png)

## Univariate functions of the parameter
### Area under the curve

In pharmacokinetics,
one quantity of particular interest is the area under the response curve.
It can be calculated as
```math
\mathrm{AUC}(\theta) = s \bigg( \frac{1}{e} - \frac{1}{a} \bigg).
```

Suppose we want to find a design that maximizes the information about ``\mathrm{AUC}(\theta)``.
This needs an additional approximation step via the [delta method](https://en.wikipedia.org/wiki/Delta_method),
for which we need to know the Jacobian matrix of ``\mathrm{AUC}``:

```@example main
function Dauc(p)
    del_a =  p.s / p.a^2
    del_e = -p.s / p.e^2
    del_s = 1 / p.e - 1 / p.a
    return [del_a del_e del_s]
end
nothing # hide
```

Note that we return a _row_ vector,
and that the order of the partial derivatives is the same as in the Jacobian matrix of the mean function.

The only difference to the previous call of [`optimize_design`](@ref)
is that we now use a [`DeltaMethod`](@ref) object wrapped around the Jacobian matrix function
instead of the identity transformation.
```@example main
t_auc = DeltaMethod(Dauc)
Random.seed!(1357)
s_auc, r_auc = optimize_design(pso, dc, ds, m, cp, pk, t_auc, na)
s_auc
```

```@example main
plot_gateauxderivative(dc, s_auc, ds, m, cp, pk, t_auc, na)
savefig(ans, "handling-transformations-gd-auc.png"); nothing # hide
```

![](handling-transformations-gd-auc.png)

The design points are similar to those of `s_id`,
but while its weights were practically uniform,
the weight of `s_auc` is almost completely placed at its third design point.
We can also see that `s_auc` is nearly three times as efficient as `s_id` for estimating the AUC:

```@example main
efficiency(s_id, s_auc, m, cp, pk, t_auc, na)
```

Note that this relation is not necessarily symmetric:
`s_id` is five times as efficient as `s_auc` for estimating ``\theta``.

```@example main
efficiency(s_auc, s_id, m, cp, pk, t_id, na)
```

### Time to maximum concentration

Now let's find a design that is optimal for estimating the point in time where the concentration is highest.
Differentiating ``\mu`` with respect to ``x``,
equating with ``0`` and solving for ``x`` gives
```math
t_{\max}(\theta) = \frac{\log(a / e)}{a - e}.
```

Its Jacobian matrix is

```@example main
function Dttm(p)
    A = p.a - p.e
    A2 = A^2
    B = log(p.a / p.e)
    da = (A / p.a - B) / A2
    de = (B - A / p.e) / A2
    ds = 0
    return [da de ds]
end
nothing # hide
```

```@example main
t_ttm = DeltaMethod(Dttm)
Random.seed!(1357)
s_ttm, r_ttm = optimize_design(pso, dc, ds, m, cp, pk, t_ttm, na)
s_ttm
```

```@example main
plot_gateauxderivative(dc, s_ttm, ds, m, cp, pk, t_ttm, na)
savefig(ans, "handling-transformations-gd-ttm.png"); nothing # hide
```

![](handling-transformations-gd-ttm.png)

This solution differs markedly from the previous ones.

### Maximum concentration

What if we're not interested in the _time_ of maximum concentration,
but in the _value_ of the maximum concentration ``\mu(t_{\max}(\theta), \theta)`` itself?

```@example main
ttm(p) = log(p.a / p.e) / (p.a - p.e)

function Dcmax(p)
    tmax = ttm(p)
    A = exp(-p.a * tmax)
    E = exp(-p.e * tmax)
    F = p.a * A - p.e * E
    da_ttm, de_ttm, ds_ttm = Dttm(p)
    da = p.s * ( tmax * A - F * da_ttm)
    de = p.s * (-tmax * E + F * de_ttm)
    ds = E - A
    return [da de ds]
end
nothing # hide
```

```@example main
t_cmax = DeltaMethod(Dcmax)
Random.seed!(1357)
s_cmax, r_cmax = optimize_design(pso, dc, ds, m, cp, pk, t_cmax, na)
s_cmax
```

```@example main
plot_gateauxderivative(dc, s_cmax, ds, m, cp, pk, t_cmax, na)
savefig(ans, "handling-transformations-gd-cmax.png"); nothing # hide
```

![](handling-transformations-gd-cmax.png)

The solution is again mostly concentrated on one design point.
The location of this point makes intuitive sense,
considering the prior expected time to maximum concentration:

```@example main
mean(ttm, pk.p)
```

## Multivariate functions of the parameter

Finding D-optimal designs for vector functions of the parameter is just as easy.
Suppose we are interested in _both_ the time and the value of the maximum concentration.
Because we have already defined their single Jacobian matrices,
we now only need to concatenate them vertically.

```@example main
Dboth(p) = [Dttm(p); Dcmax(p)]
t_both = DeltaMethod(Dboth)
Random.seed!(1357)
s_both, r_both = optimize_design(pso, dc, ds, m, cp, pk, t_both, na)
s_both
```

```@example main
plot_gateauxderivative(dc, s_both, ds, m, cp, pk, t_both, na)
savefig(ans, "handling-transformations-gd-both.png"); nothing # hide
```

![](handling-transformations-gd-both.png)

This solution is again similar to `s_id`, but with a different weight distribution.

Finally we compare all pairwise efficiencies (solutions in rows, transformations in columns):

```@example main
solutions = [s_id, s_auc, s_ttm, s_cmax, s_both]
trafos = [t_id, t_auc, t_ttm, t_cmax, t_both]
map(Iterators.product(solutions, zip(solutions, trafos))) do (s1, (s2, t))
	efficiency(s1, s2, m, cp, pk, t, na)
end
```
