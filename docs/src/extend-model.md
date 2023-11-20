# New Model Supertype

```@setup main
check_results = true
```

Kirstine.jl comes with support for nonlinear regression models.
The theory of optimal design is of course applicable also to other regression settings.
This vignette explains how to extend Kirstine.jl
so that we can also find optimal designs for logistic regression.

The statistical model is

```math
\begin{align*}
\Unit_{\IndexUnit} \mid π_{\IndexUnit} &\simiid \BernoulliDist(π_{\IndexUnit}) \\
\Logit(π_{\IndexUnit}) &= ν(\Covariate_{\IndexUnit}, \Parameter).
\end{align*}
```

Since we allow the predictor to be a nonlinear function
``ν : \CovariateSet × \ParameterSet → \Reals``,
our setup is more general the usual logistic GLM
where the predictor is linear.

Now, we need to find the [Fisher information matrix](math.md#Objective-Function) of the model.
With the log-likelihood

```math
\LogLikelihood(\Unit,\Covariate,\Parameter)
=
\Unit \log(\Expit(ν(\Covariate, \Parameter))) + (1 - \Unit) \log(1 - \Expit(ν(\Covariate, \Parameter)))
```

and some calculations we arrive at

```math
- \Expectation[\Hessian\LogLikelihood\mid\Parameter]
= \Expit(ν) (1 - \Expit(ν)) \TotalDiff ν' \TotalDiff ν.
```

## Implementation

First we introduce a new subtype of [`Model`](@ref),
and a corresponding `ModelWorkspace` to hold the pre-allocated Jacobian matrix of the predictor.
Then we implement a method for `average_fishermatrix!`
that specializes on logistic regression models
and returns the upper triangle of the averaged Fisher matrices.

If we had any expressions that depend on ``\Covariate``,
but not on ``\Parameter``,
we could store them in additional fields of the `ModelWorkspace`
and compute them in `update_model_workspace`.

```@example main
using Kirstine, Plots, Random
import LinearAlgebra: BLAS

abstract type LogisticRegression <: Kirstine.Model end

# We can define this on the supertype since it will always be the case.
Kirstine.unit_length(m::LogisticRegression) = 1

struct LRWorkspace <: Kirstine.ModelWorkspace
    jm_predictor::Matrix{Float64}
end

# We can ignore the number of design points `K` here.
function Kirstine.allocate_model_workspace(
    K::Integer,
    m::LogisticRegression,
    pk::PriorSample,
)
    return LRWorkspace(zeros(1, dimension(pk.p[1])))
end

# nothing to do
function Kirstine.update_model_workspace!(
    mw::LRWorkspace,
    m::LogisticRegression,
    c::AbstractVector{<:Covariate},
)
    return mw
end

expit(x) = 1 / (1 + exp(-x))

function Kirstine.average_fishermatrix!(
    afm::AbstractMatrix{<:Real},
    mw::LRWorkspace,
    w::AbstractVector,
    m::LogisticRegression,
    c::AbstractVector{<:Covariate},
    p::Parameter,
)
    fill!(afm, 0.0)
    for k in 1:length(w)
        # We need to implement the next two functions for concrete subtypes.
        predictor_jacobianmatrix!(mw.jm_predictor, m, c[k], p)
        prob = expit(nonlinear_predictor(m, c[k], p))
        p1mp = prob * (1 - prob)
        BLAS.syrk!('U', 'T', p1mp * w[k], mw.jm_predictor, 1.0, afm)
    end
    return afm
end
```

The call to [`BLAS.syrk!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.BLAS.syrk%21)
is a more efficient way to compute

```julia
afm .+= p1mp * w[k] * mw.jm_predictor' * mw.jm_predictor
```

in-place.
Since the result is symmetric,
only the upper triangle of `afm` is filled.

## Examples

### Linear Predictor

```math
ν(\Covariate, \Parameter) = β₀ + β₁x₁ + β₂x₂
```

where ``\Covariate = (x₁, x₂)`` and ``\Parameter = (β₀, β₁, β₂)``.

Say we are looking for a design that is optimal for estimating the odds ratios

```math
\Transformation(\Parameter) = (\exp(β₁), \exp(β₂))'.
```

```@example main
struct LinLogReg <: LogisticRegression end

mutable struct LLRCovariate <: Covariate
    x1::Float64
    x2::Float64
end

Kirstine.allocate_covariate(m::LinLogReg) = LLRCovariate(0, 0)

@simple_parameter LLR β0 β1 β2

function nonlinear_predictor(m::LinLogReg, c::LLRCovariate, p::LLRParameter)
    return p.β0 + p.β1 * c.x1 + p.β2 * c.x2
end

function predictor_jacobianmatrix!(jm, m::LinLogReg, c::LLRCovariate, p::LLRParameter)
    jm[1, 1] = 1
    jm[1, 2] = c.x1
    jm[1, 3] = c.x2
    return jm
end

function erplot1(d, dp)
    f1(x, c, p) = expit(p.β0 + p.β1 * x + p.β2 * c.x2)
    f2(x, c, p) = expit(p.β0 + p.β1 * c.x1 + p.β2 * x)
    g1(c) = [c.x1]
    g2(c) = [c.x2]
    h(c, p) = [expit(p.β0 + p.β1 * c.x1 + p.β2 * c.x2)]
    xrange = range(lowerbound(region(dp))[1], upperbound(region(dp))[1]; length = 101)
    cp = covariate_parameterization(dp)
    pk = prior_knowledge(dp)
    m = model(dp)
    p1 = plot_expected_function(f1, g1, h, xrange, d, m, cp, pk; xguide = "x1")
    p2 = plot_expected_function(f2, g2, h, xrange, d, m, cp, pk; xguide = "x2")
    return plot(p1, p2)
end

dp1 = DesignProblem(
    criterion = DOptimality(),
    region = DesignInterval(:x1 => (0, 1), :x2 => (0, 1)),
    covariate_parameterization = JustCopy(:x1, :x2),
    prior_knowledge = PriorSample([LLRParameter(β0 = -4, β1 = 6, β2 = -3)]),
    model = LinLogReg(),
    transformation = DeltaMethod(p -> [0 exp(p.β1) 0; 0 0 exp(p.β2)]),
)

Random.seed!(12345)
str1 = DirectMaximization(
    optimizer = Pso(swarmsize = 100, iterations = 50),
    prototype = random_design(region(dp1), 4),
)

Random.seed!(12345)
s1, r1 = solve(dp1, str1; mindist = 5e-2, minweight = 1e-3)
gd1 = plot_gateauxderivative(s1, dp1)
savefig(gd1, "extend-model-gd1.png") # hide
nothing
```

```@setup main
s1 == DesignMeasure(
 [0.34921961868772716, 0.0] => 0.3543739967246416,
 [0.9841082624551549, 0.0] => 0.35433329676914754,
 [1.0, 1.0] => 0.29129270650621086,
) || !check_results || error("not the expected result\n", s1)
```

![](extend-model-gd1.png)

```@example main
er1 = erplot1(s1, dp1)
savefig(er1, "extend-model-er1.png") # hide
nothing # hide
```

![](extend-model-er1.png)

### Nonlinear Predictor

A slightly silly example.

```math
ν(\Covariate, \Parameter) = a \sin(2π b (t - c))
```

where ``\Covariate = t`` and ``\Parameter = (a, b, c)``
with ``a > 0``, ``b > 0`` and ``0 ≤ c < 1``.
We are interested in estimating all of ``\Parameter``.

```@example main
struct SinLogReg <: LogisticRegression end

mutable struct Time <: Covariate
    t::Float64
end

Kirstine.allocate_covariate(m::SinLogReg) = Time(0)

@simple_parameter SLR a b c

function nonlinear_predictor(m::SinLogReg, c::Time, p::SLRParameter)
    return p.a * sin(2 * pi * p.b * (c.t - p.c))
end

function predictor_jacobianmatrix!(jm, m::SinLogReg, c::Time, p::SLRParameter)
    sin_term = sin(2 * pi * p.b * (c.t - p.c))
    cos_term = cos(2 * pi * p.b * (c.t - p.c))
    jm[1, 1] = sin_term
    jm[1, 2] = p.a * cos_term * 2 * pi * (c.t - p.c)
    jm[1, 3] = -p.a * cos_term * 2 * pi * p.b
    return jm
end

function erplot2(d, dp)
    f(x, c, p) = expit(p.a * sin(2 * pi * p.b * (x - p.c)))
    g(c) = [c.t]
    h(c, p) = [expit(p.a * sin(2 * pi * p.b * (c.t - p.c)))]
    xrange = range(lowerbound(region(dp))[1], upperbound(region(dp))[1]; length = 101)
    cp = covariate_parameterization(dp)
    pk = prior_knowledge(dp)
    m = model(dp)
    return plot_expected_function(f, g, h, xrange, d, m, cp, pk; xguide = "time")
end

dp2 = DesignProblem(
    criterion = DOptimality(),
    region = DesignInterval(:t => (0, 1)),
    covariate_parameterization = JustCopy(:t),
    prior_knowledge = PriorSample([SLRParameter(a = 1, b = 1.5, c = 0.125)]),
    model = SinLogReg(),
)

Random.seed!(12345)
str2 = DirectMaximization(
    optimizer = Pso(swarmsize = 100, iterations = 50),
    prototype = random_design(region(dp2), 4),
)

Random.seed!(12345)
s2, r2 = solve(dp2, str2; mindist = 5e-2, minweight = 1e-3)
gd2 = plot_gateauxderivative(s2, dp2)
savefig(gd2, "extend-model-gd2.png") # hide
nothing # hide
```

```@setup main
s2 == DesignMeasure(
 [0.10724768400759671] => 0.3335078264280175,
 [0.8412520842064646] => 0.33343038623171556,
 [1.0] => 0.33306178734026687,
) || !check_results || error("not the expected result\n", s2)
```

![](extend-model-gd2.png)

```@example main
er2 = erplot2(s2, dp2)
savefig(er2, "extend-model-er2.png") # hide
nothing # hide
```

![](extend-model-er2.png)
