# Optimal Design for Functions of the Parameter

This vignette shows how to find a D-optimal design
for estimating some transformation ``\Transformation(\Parameter)`` of the model parameter.
The function ``\Transformation : \ParameterSet → \Reals^{t}`` must be differentiable
with a full-rank Jacobian matrix.

Our example in this vignette is the three-parameter compartment model from Atkinson et al[^ACHJ93].
In pharmacokinetics,
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

```@setup main
# we can't do the `savefig(); nothing # hide` trick when using JuliaFormatter
function savefig_nothing(plot, filename)
	savefig(plot, filename)
	return nothing
end
```

```@example main
using Plots # hide
θ = (a = 4.298, e = 0.05884, s = 21.80)
tpc = plot(
    x -> θ.s * (exp(-θ.e * x) - exp(-θ.a * x));
    xlims = (0, 10),
    xguide = "time [h]",
    yguide = "response",
    label = "μ(time , θ)",
)
savefig_nothing(tpc, "transformations-tpc.png") # hide
```

![](transformations-tpc.png)

[^ACHJ93]: Anthony C. Atkinson, Kathryn Chaloner, Agnes M. Herzberg, and June Juritz (1993). Optimum experimental designs for properties of a compartmental model. Biometrics, 49(2), 325–337. [doi:10.2307/2532547](http://dx.doi.org/10.2307/2532547)
## Setup

As in the [introductory example](tutorial.md),
we start by defining the (single-row) Jacobian matrix of the mean function,
and the mapping from design variables to model covariates.

```@example main
using Kirstine, Plots, Random, Statistics

@simple_model TPC time
@simple_parameter TPC a e s
struct Copy <: CovariateParameterization end

function Kirstine.jacobianmatrix!(jm, m::TPCModel, c::TPCCovariate, p::TPCParameter)
    A = exp(-p.a * c.time)
    E = exp(-p.e * c.time)
    jm[1, 1] = A * p.s * c.time
    jm[1, 2] = -E * p.s * c.time
    jm[1, 3] = E - A
    return m
end

function Kirstine.update_model_covariate!(c::TPCCovariate, dp, m::TPCModel, cp::Copy)
    c.time = dp[1]
    return c
end
```

In this vignette we will only look at Bayesian optimal designs
because locally optimal designs for scalar ``\Transformation(\Parameter)`` usually don't exist
due to their information matrices becoming singular.

!!! note
    
    Workarounds like generalized inverses or matrix regularization
    are currently not supported by `Kirstine.jl`.

For the prior we will use “distribution I” from[^ACHJ93],
which is constructed from two independent uniform distributions around estimates for ``a`` and ``e``,
and a point mass for ``s``.
The strength of the prior is controlled by the width of the uniform distributions.
We generate a [`PriorSample`](@ref) from `1000` draws.

```@example main
function draw_from_prior(n, se_factor)
    mn = [4.298, 0.05884, 21.8]
    se = [0.5, 0.005, 0]
    as = mn[1] .+ se_factor .* se[1] .* (2 .* rand(n) .- 1)
    es = mn[2] .+ se_factor .* se[2] .* (2 .* rand(n) .- 1)
    ss = mn[3] .+ se_factor .* se[3] .* (2 .* rand(n) .- 1)
    return PriorSample(map((a, e, s) -> TPCParameter(a = a, e = e, s = s), as, es, ss))
end
nothing # hide
```

In this vignette we will focus on the effect of different transformations.
The following helper function lets us specify the parts of the design problem that do not change just once.
We fix the seed so that different transformations do not accidentally use different prior samples.
Our design interval extends from `0` to `48` hours after administration of the drug.

```@example main
function dp_for_trafo(trafo)
    Random.seed!(4711)
    DesignProblem(
        design_region = DesignInterval(:time => [0, 48]),
        design_criterion = DOptimality(),
        covariate_parameterization = Copy(),
        model = TPCModel(sigma = 1),
        prior_knowledge = draw_from_prior(1000, 2),
        transformation = trafo,
    )
end
nothing # hide
```

In all subsequent examples we will use identical settings for the particle swarm optimizer,
and start with an equidistant design.

```@example main
pso = Pso(swarmsize = 50, iterations = 100)
dms = DirectMaximization(optimizer = pso, prototype = uniform_design([[0], [24], [48]]))
nothing # hide
```

The following is a small wrapper around [`plot_expected_function`](@ref)
that plots a time-response curve
and overlays the measurement points implied by the design.

```@example main
function plot_expected_response(d::DesignMeasure, dp::DesignProblem)
    response(t, p) = p.s * (exp(-p.e * t) - exp(-p.a * t))
    f(x, c, p) = response(x, p)
    g(c) = [c.time]
    h(c, p) = [response(c.time, p)]
    xrange = range(0, 48; length = 101)
    cp = covariate_parameterization(dp)
    plt = plot_expected_function(f, g, h, xrange, d, model(dp), cp, prior_knowledge(dp))
    plot!(plt; xguide = "time", yguide = "response", xticks = 0:6:48)
    return plt
end
nothing # hide
```

## Identity Transformation

In order to have a design to compare other solutions against,
we first determine the Bayesian D-optimal design for the whole parameter ``\theta``.

```@example main
dp_id = dp_for_trafo(Identity())
Random.seed!(1357)
s_id, r_id = solve(dp_id, dms)
nothing # hide
```

```@example main
s_id
```

```@example main
gd_id = plot_gateauxderivative(s_id, dp_id)
savefig_nothing(gd_id, "transformations-gd-id.png") # hide
```

![](transformations-gd-id.png)

```@example main
ef_id = plot_expected_response(s_id, dp_id)
savefig_nothing(ef_id, "transformations-ef-id.png") # hide
```

![](transformations-ef-id.png)

## Univariate Functions

### Area Under the Curve

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
    del_a = p.s / p.a^2
    del_e = -p.s / p.e^2
    del_s = 1 / p.e - 1 / p.a
    return [del_a del_e del_s]
end
nothing # hide
```

Note that we return a _row_ vector,
and that the order of the partial derivatives is the same as in the Jacobian matrix of the mean function.

The only difference to the previous call of `dp_for_trafo`
is that we now use a [`DeltaMethod`](@ref) object wrapped around the Jacobian matrix function
instead of the identity transformation.

```@example main
dp_auc = dp_for_trafo(DeltaMethod(Dauc))
Random.seed!(1357)
s_auc, r_auc = solve(dp_auc, dms)
s_auc
```

```@example main
gd_auc = plot_gateauxderivative(s_auc, dp_auc)
savefig_nothing(gd_auc, "transformations-gd-auc.png") # hide
```

![](transformations-gd-auc.png)

```@example main
ef_auc = plot_expected_response(s_auc, dp_auc)
savefig_nothing(ef_auc, "transformations-ef-auc.png") # hide
```

![](transformations-ef-auc.png)

The design points are similar to those of `s_id`,
but while its weights were practically uniform,
the weight of `s_auc` is almost completely placed at its third design point.
We can also see that `s_auc` is nearly three times as efficient as `s_id` for estimating the AUC:

```@example main
efficiency(s_id, s_auc, dp_auc)
```

Note that this relation is not necessarily symmetric:
`s_id` is five times as efficient as `s_auc` for estimating ``\theta``.

```@example main
efficiency(s_auc, s_id, dp_id)
```

### Time to Maximum Concentration

Now let's find a design that is optimal for estimating the point in time where the concentration is highest.
Differentiating ``\mu`` with respect to ``x``,
equating with ``0`` and solving for ``x`` gives

```math
t_{\max{}}(\theta) = \frac{\log(a / e)}{a - e}.
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
dp_ttm = dp_for_trafo(DeltaMethod(Dttm))
Random.seed!(1357)
s_ttm, r_ttm = solve(dp_ttm, dms)
s_ttm
```

```@example main
gd_ttm = plot_gateauxderivative(s_ttm, dp_ttm)
savefig_nothing(gd_ttm, "transformations-gd-ttm.png") # hide
```

![](transformations-gd-ttm.png)

```@example main
ef_ttm = plot_expected_response(s_ttm, dp_ttm)
savefig_nothing(ef_ttm, "transformations-ef-ttm.png") # hide
```

![](transformations-ef-ttm.png)

This solution differs markedly from the previous ones.

### Maximum Concentration

What if we're not interested in the _time_ of maximum concentration,
but in the _value_ of the maximum concentration ``\mu(t_{\max{}}(\theta), \theta)`` itself?

```@example main
ttm(p) = log(p.a / p.e) / (p.a - p.e)

function Dcmax(p)
    tmax = ttm(p)
    A = exp(-p.a * tmax)
    E = exp(-p.e * tmax)
    F = p.a * A - p.e * E
    da_ttm, de_ttm, ds_ttm = Dttm(p)
    da = p.s * (tmax * A - F * da_ttm)
    de = p.s * (-tmax * E + F * de_ttm)
    ds = E - A
    return [da de ds]
end
nothing # hide
```

```@example main
dp_cmax = dp_for_trafo(DeltaMethod(Dcmax))
Random.seed!(13579)
s_cmax, r_cmax = solve(dp_cmax, dms)
s_cmax
```

```@example main
gd_cmax = plot_gateauxderivative(s_cmax, dp_cmax)
savefig_nothing(gd_cmax, "transformations-gd-cmax.png") # hide
```

![](transformations-gd-cmax.png)

```@example main
ef_cmax = plot_expected_response(s_cmax, dp_cmax)
savefig_nothing(ef_cmax, "transformations-ef-cmax.png") # hide
```

![](transformations-ef-cmax.png)

The solution is again mostly concentrated on one design point.
The location of this point makes intuitive sense,
considering the prior expected time to maximum concentration:

```@example main
mean(ttm, dp_cmax.pk.p)
```

## Multivariate Functions

Finding D-optimal designs for vector functions of the parameter is just as easy.
Suppose we are interested in _both_ the time and the value of the maximum concentration.
Because we have already defined their single Jacobian matrices,
we now only need to concatenate them vertically.

```@example main
Dboth(p) = [Dttm(p); Dcmax(p)]
dp_both = dp_for_trafo(DeltaMethod(Dboth))
Random.seed!(1357)
s_both, r_both = solve(dp_both, dms)
s_both
```

```@example main
gd_both = plot_gateauxderivative(s_both, dp_both)
savefig_nothing(gd_both, "transformations-gd-both.png") # hide
```

![](transformations-gd-both.png)

```@example main
ef_both = plot_expected_response(s_both, dp_both)
savefig_nothing(ef_both, "transformations-ef-both.png") # hide
```

![](transformations-ef-both.png)

This solution is again similar to `s_id`, but with a different weight distribution.

Finally we compare all pairwise efficiencies (solutions in rows, transformations in columns):

```@example main
solutions = [s_id, s_auc, s_ttm, s_cmax, s_both]
problems = [dp_id, dp_auc, dp_ttm, dp_cmax, dp_both]
map(Iterators.product(solutions, zip(solutions, problems))) do (s1, (s2, dp))
    efficiency(s1, s2, dp)
end
```
