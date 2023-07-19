# Discrete Priors With Non-Uniform Weights

```@setup main
# we can't do the `savefig(); nothing # hide` trick when using JuliaFormatter
function savefig_nothing(plot, filename)
	savefig(plot, filename)
	return nothing
end
```

Sometimes prior knowledge is not described by a continuous distribution,
from which we take a Monte-Carlo sample,
but by a genuinely discrete distribution.
We illustrate this here with the dose-response model from [“Getting Started”](getting-started.md).

## Model Setup

```@example main
using Kirstine, Random, Plots

@define_scalar_unit_model Kirstine SigEmax dose
@define_vector_parameter Kirstine SigEmaxPar e0 emax ed50 h

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

struct CopyDose <: CovariateParameterization end

function Kirstine.update_model_covariate!(c::SigEmaxCovariate, dp, m::SigEmax, cp::CopyDose)
    c.dose = dp[1]
    return c
end
ds = DesignInterval(:dose => (0, 1))
nothing # hide
```

## Prior Knowledge

The slope parameter ``h`` of the sigmoid Emax model can in some situations be interpreted
as the number of molecules that need to bind to a receptor in order to produce an effect.[^W97]
Suppose we suspect that only values of

```math
h\in\{1,2,3,4\}
```

with prior probabilities ``\{0.1, 0.3, 0.4, 0.2\}`` are possible.
For simplicity suppose further
that we know the values of the remaining elements of ``θ`` exactly.
With a [`DiscretePrior`](@ref),
we can pass the vector of prior probabilities as the optional second argument.

```@example main
prior = DiscretePrior(
    [SigEmaxPar(e0 = 1, emax = 2, ed50 = 0.4, h = h) for h in 1:4],
    [0.1, 0.3, 0.4, 0.2],
)

dp = DesignProblem(
    design_criterion = DOptimality(),
    design_space = ds,
    model = SigEmax(1),
    covariate_parameterization = CopyDose(),
    prior_knowledge = prior,
)
nothing # hide
```

[^W97]: James N. Weiss (1997). The hill equation revisited: uses and misuses. The FASEB Journal, 11(11), 835–841. [doi:10.1096/fasebj.11.11.9285481](http://dx.doi.org/10.1096/fasebj.11.11.9285481)
## Optimal Design

```@example main
strategy = DirectMaximization(
    optimizer = Pso(iterations = 50, swarmsize = 100),
    prototype = equidistant_design(ds, 8),
)

Random.seed!(31415)
s1, r1 = solve(dp, strategy, minweight = 1e-3, mindist = 1e-2)
s1
```

```@example main
gd = plot_gateauxderivative(s1, dp)
savefig_nothing(gd, "unequal-prior-weights-gd.png") # hide
```

![](unequal-prior-weights-gd.png)
