# Using Automatic Differentiation

Analytically finding and then implementing the Jacobian matrix of the mean function for complex models can be time-consuming.
It is possible to use your favorite [Automatic Differentiation package](https://juliadiff.org/) for this task.
This vignette shows how to use [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
for the Sigmoid Emax model from the [tutorial](tutorial.md).
Wrapping other autodiff packages should work similarly.

We start by comparing a naive autodiff version to manual gradients,
and then show a slightly hacky but faster version.

## Naive Implementation

```@example main
using Kirstine, Random, Plots, ForwardDiff
using BenchmarkTools: @benchmark

@simple_model SigEmax dose
```

Since ForwardDiff can only compute the derivative with respect to a `Vector` argument,
we define a [`Parameter`](@ref) subtype that wraps a `Vector`.

```@example main
struct SigEmaxVectorParameter <: Parameter
    θ::Vector{Float64} # θ = (e0, emax, ed50, h)
end

Kirstine.dimension(p::SigEmaxVectorParameter) = 4

# We need to prevent NaN in the automatic gradient for x == 0
mu(x, θ) = x == 0 ? θ[1] : θ[1] + θ[2] / ((x / θ[3])^(-θ[4]) + 1)

function jacobianmatrix_auto!(
    jm,
    m::SigEmaxModel,
    c::SigEmaxCovariate,
    p::SigEmaxVectorParameter,
)
    f(θ) = mu(c.dose, θ)
    return ForwardDiff.gradient!(jm, f, p.θ)
end

function jacobianmatrix_manual!(
    jm,
    m::SigEmaxModel,
    c::SigEmaxCovariate,
    p::SigEmaxVectorParameter,
)
    dose_pow_h = c.dose^p.θ[4]
    ed50_pow_h = p.θ[3]^p.θ[4]
    A = dose_pow_h / (dose_pow_h + ed50_pow_h)
    B = ed50_pow_h * p.θ[2] / (dose_pow_h + ed50_pow_h)
    jm[1, 1] = 1.0
    jm[1, 2] = A
    jm[1, 3] = -A * B * p.θ[4] / p.θ[3]
    jm[1, 4] = c.dose == 0 ? 0.0 : A * B * log(c.dose / p.θ[3])
    return jm
end
nothing # hide
```

We set up an example design problem.

```@example main
theta_mean = [1, 2, 0.4, 5]
theta_sd = [0.5, 0.5, 0.05, 0.5]
Random.seed!(31415)
sep_draws = map(1:1000) do i
    rnd = theta_mean .+ theta_sd .* randn(4)
    return SigEmaxVectorParameter(rnd)
end
prior_sample = PriorSample(sep_draws)

dp1 = DesignProblem(
    criterion = DCriterion(),
    region = DesignInterval(:dose => (0, 1)),
    model = SigEmaxModel(sigma = 1),
    covariate_parameterization = JustCopy(:dose),
    prior_knowledge = prior_sample,
)
nothing # hide
```

Let's first compare the time it takes to evaluate the Jacobian matrices once.

```@example main
jm = zeros(1, 4)
co = SigEmaxCovariate(0.5)
bm1 = @benchmark jacobianmatrix_auto!($jm, $model(dp1), $co, $sep_draws[1])
```

```@example main
bm2 = @benchmark jacobianmatrix_manual!($jm, $model(dp1), $co, $sep_draws[1])
```

We clearly see that the automatic gradient is about three times slower,
and that it does allocate additional memory.

Next, we compare the time it takes to run the Particle Swarm optimizer for a given number of iterations.
In these examples the `iterations` and `swarmsize` are smaller than needed for full convergence
in order to keep documentation compile time low.
But since runtime scales linearly in each,
this is no obstacle for timing comparisons.

```@example main
# force compilation before time measurement
dummy = DirectMaximization(
    optimizer = Pso(iterations = 2, swarmsize = 2),
    prototype = equidistant_design(region(dp1), 6),
)

strategy = DirectMaximization(
    optimizer = Pso(iterations = 20, swarmsize = 50),
    prototype = equidistant_design(region(dp1), 6),
)

Kirstine.jacobianmatrix!(jm, m, c, p) = jacobianmatrix_auto!(jm, m, c, p)
solve(dp1, dummy; maxweight = 1e-3, maxdist = 1e-2)
Random.seed!(54321)
@time s1, r1 = solve(dp1, strategy; maxweight = 1e-3, maxdist = 1e-2)
nothing # hide (force inserting stdout from @time)
```

```@example main
Kirstine.jacobianmatrix!(jm, m, c, p) = jacobianmatrix_manual!(jm, m, c, p)
solve(dp1, dummy; maxweight = 1e-3, maxdist = 1e-2)
Random.seed!(54321)
@time s2, r2 = solve(dp1, strategy; maxweight = 1e-3, maxdist = 1e-2)
nothing # hide
```

Here we see that solving the design problem with automatic gradients is about half as fast as when using the manual gradient,
and it needs about three orders of magnitude more memory and allocations.
In the next section we look at a way to speed things up by reducing the number of allocations.
But first let's check that both candidate solutions are equal.

```@example main
s1 == s2
```

## Faster Implementation

One way to reduce the number of allocations is to pass a [`ForwardDiff.GradientConfig`](https://juliadiff.org/ForwardDiff.jl/stable/user/api/#ForwardDiff.GradientConfig) object to `ForwardDiff.gradient!`.
However, this will not work if the closure `θ -> mu(c.dose, θ)` changes in every call to `jacobianmatrix!`.
The following is a slightly hacky solution to this problem
which misuses the model type.

 1. Add a `GradientConfig` field to the model type.
 2. Add a `SigEmaxCovariate` field to the model type,
 3. In the model constructor,
    set up a closure over this covariate,
    and save the closure in a third additional field.
 4. In `jacobianmatrix!`, modify the closed-over covariate
    and then call `gradient!` with the saved closure.

```@example main
struct HackySigEmaxModel{T<:Function} <: NonlinearRegression
    sigma::Float64
    cfg::ForwardDiff.GradientConfig
    c::SigEmaxCovariate
    mu_closure::T
    function HackySigEmaxModel(; sigma, pardim, mu)
        c = SigEmaxCovariate(0)
        mu_closure = θ -> mu(c, θ)
        cfg = ForwardDiff.GradientConfig(
            mu_closure,
            zeros(1, pardim),
            ForwardDiff.Chunk{pardim}(),
        )
        new{typeof(mu_closure)}(sigma, cfg, c, mu_closure)
    end
end

Kirstine.allocate_covariate(m::HackySigEmaxModel) = SigEmaxCovariate(0)
Kirstine.unit_length(m::HackySigEmaxModel) = 1
function Kirstine.update_model_vcov!(Sigma, m::HackySigEmaxModel, c::SigEmaxCovariate)
    Sigma[1, 1] = m.sigma^2
    return Sigma
end

function Kirstine.jacobianmatrix!(
    jm,
    m::HackySigEmaxModel,
    c::SigEmaxCovariate,
    p::SigEmaxVectorParameter,
)
    m.c.dose = c.dose # modify the closed-over covariate
    return ForwardDiff.gradient!(jm, m.mu_closure, p.θ, m.cfg)
end

dph = DesignProblem(
    criterion = DCriterion(),
    region = DesignInterval(:dose => (0, 1)),
    model = HackySigEmaxModel(
        sigma = 1,
        pardim = 4,
        mu = (c, θ) -> c.dose == 0 ? θ[1] : θ[1] + θ[2] / ((c.dose / θ[3])^(-θ[4]) + 1),
    ),
    covariate_parameterization = JustCopy(:dose),
    prior_knowledge = prior_sample,
)

bm3 = @benchmark Kirstine.jacobianmatrix!($jm, $model(dph), $co, $sep_draws[1])
```

This time, the automatic gradient is only slower than the manual one by a factor of two.

```@example main
solve(dph, dummy; maxweight = 1e-3, maxdist = 1e-2)
Random.seed!(54321)
@time s3, r3 = solve(dph, strategy; maxweight = 1e-3, maxdist = 1e-2)
nothing # hide
```

Solving the design problem now only takes about 30% more time.
The number of allocations is still much higher than with the manual gradient,
but cumulative memory usage compared to the naive implementation is reduced by a factor of 4.

The candidate solution is still the same.

```@example main
s3 == s1
```

## Transformations

A place where autodiff does not incur a noticeable performance hit
is the computation of a [`Transformation`](@ref)'s Jacobian matrix
at the parameter values in the prior sample,
since these are computed only once for a design problem.

To illustrate this,
lets try to find a design that is optimal for estimating
``\mathrm{ED}_{ρ}`` with some fixed ``0 < ρ < 100``,
i.e. the concentration such that

```math
\MeanFunction(\mathrm{ED}_{ρ},\Parameter) = E_0 + \frac{ρ}{100} E_{\max{}}.
```

While it is relatively easy to find that

```math
\mathrm{ED}_{ρ} =
\Transformation(\Parameter) :=
\frac{\mathrm{ED}_{50}}{(100/ρ - 1)^{1/h}},
```

figuring out the partial derivatives of ``\Transformation`` is somewhat tedious.
The automatic version, in contrast, only takes on line of code.

```@example main
# switch back to faster manual jacobian matrix of the mean function
Kirstine.jacobianmatrix!(jm, m, c, p) = jacobianmatrix_manual!(jm, m, c, p)

ED(θ, rho) = [θ[3] / (100 / rho - 1)^(1 / θ[4])]

function DED_manual(p, rho)
    del_e0 = 0
    del_emax = 0
    a = (100 / rho - 1)^(1 / p.θ[4])
    del_ed50 = 1 / a
    del_h = p.θ[3] * log(100 / rho - 1) / (p.θ[4]^2 * a)
    return [del_e0 del_emax del_ed50 del_h]
end

function DED_auto(p, rho)
    return ForwardDiff.jacobian(θ -> ED(θ, rho), p.θ)
end

# quickly check for implementation error
DED_manual(parameters(prior_sample)[1], 10) .- DED_auto(parameters(prior_sample)[1], 10)
```

```@example main
dp2 = DesignProblem(
    criterion = DCriterion(),
    region = DesignInterval(:dose => (0, 1)),
    model = SigEmaxModel(sigma = 1),
    covariate_parameterization = JustCopy(:dose),
    prior_knowledge = prior_sample,
    transformation = DeltaMethod(p -> DED_auto(p, 90)), # optimal for ED_{90}
)

solve(dp2, dummy; maxweight = 1e-3, maxdist = 1e-2)
Random.seed!(54321)
@time s3, r3 = solve(dp2, strategy; maxweight = 1e-3, maxdist = 1e-2)
nothing # hide
```

```@example main
dp3 = DesignProblem(
    criterion = DCriterion(),
    region = DesignInterval(:dose => (0, 1)),
    model = SigEmaxModel(sigma = 1),
    covariate_parameterization = JustCopy(:dose),
    prior_knowledge = prior_sample,
    transformation = DeltaMethod(p -> DED_manual(p, 90)),  # optimal for ED_{90}
)
solve(dp3, dummy; maxweight = 1e-3, maxdist = 1e-2)
Random.seed!(54321)
@time s4, r4 = solve(dp3, strategy; maxweight = 1e-3, maxdist = 1e-2)
nothing # hide
```

As we see, both versions take practically the same amount of time.

```@example main
s3 == s4
```
