# New Design Criterion

```@setup main
check_results = true
```

In this vignette, we will implement a different variant of the D-criterion.
While [Kirstine.jl uses](math.md#D-Criterion)

```math
\DesignCriterion_D(\SomeMatrix) = \log(\det(\SomeMatrix)),
```

some authors prefer

```math
\DesignCriterion_{\tilde{D}}(\SomeMatrix) = \det(\SomeMatrix)^{1/\DimParameter}
```

for an ``a×a`` positive definite matrix ``\SomeMatrix``.

Since one criterion is a monotone bijective transformation of the other,
locally optimal designs for both will be identical.
But since this transformation is not linear,
it can not be pulled out of the integral,
and Bayesian optimal designs will be different.

For the Gateaux derivative of the objective function corresponding to ``\DesignCriterion_{\tilde{D}}``
we need to compute

```math
\MatDeriv{\DesignCriterion_{\tilde{D}}}{\SomeMatrix}{\SomeMatrix}
=
\frac{1}{a}
\det(\SomeMatrix)^{1/a}
\SomeMatrix^{-1}
```

which can, for example, be accomplished with the rules for matrix differential calculus from [^MN99].
Plugging this into the [general expressions](math.md#Gateaux-Derivative) we obtain

```math
\begin{aligned}
A
&=
\frac{1}{\DimTransformedParameter}
\det(\TNIMatrix(\DesignMeasure, \Parameter))^{1/\DimTransformedParameter}
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
 (\TotalDiff \Transformation(\Parameter))'
  \TNIMatrix(\DesignMeasure, \Parameter)
 (\TotalDiff \Transformation(\Parameter))
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
\\
\Trace[B] &= \det(\TNIMatrix(\DesignMeasure, \Parameter))^{1/\DimTransformedParameter}
\end{aligned}
```

for general transformations ``\Transformation``,
and

```math
\begin{aligned}
A
&=
\frac{1}{\DimParameter}
\det(\NIMatrix(\DesignMeasure, \Parameter))^{1/\DimParameter}
\NIMatrix^{-1}(\DesignMeasure, \Parameter)
\\
\Trace[B] &= \det(\NIMatrix(\DesignMeasure, \Parameter))^{1/\DimParameter}
\end{aligned}
```

for the special case ``\Transformation(\Parameter)=\Parameter``.

## Implementation

When a new `T <: DesignCriterion` is going to be used,
it needs a corresponding `U <: GateauxConstants`
which wraps precomputed values for all expressions
that do not depend on the direction of the Gateaux derivative.
Additionally, the following methods must be implemented.

  - `criterion_integrand(tnim, is_inv::Bool, dc::T)`:
    The function ``\DesignCriterion`` in the mathematical notation.
    This function should be fast.
    Ideally, it should not allocate new memory.
  - `gateaux_constants(dc::T, d::DesignMeasure, m::Model, cp::CovariateParameterization, pk::PriorSample, trafo::Transformation, na::NormalApproximation)`:
    Compute the value of all expensive expressions in the Gateaux derivative
    that do not depend on ``\NIMatrix(\DesignMeasureDirection,\Parameter)``
    for every parameter value in the Monte-Carlo sample.

Next, we will implement the alternative D-criterion for [`Identity`](@ref) and [`DeltaMethod`](@ref) transformations.
Since it is in some way the exponential of the predefined version,
we will call it `DexpCriterion`.

```@example main
using Kirstine
using LinearAlgebra: Symmetric, det

struct DexpCriterion <: Kirstine.DesignCriterion end

# `tnim` is the transformed normalized information matrix.
# It is implicitly symmetric, i.e. only the upper triangle contains sensible values.
# Depending on the transformation used, `tnim` can stand for M_(ζ,θ) or M_(ζ,θ)^{-1}.
# The information about this is passed in the `is_inv` flag.
# `log_det!` computes `log(det())` of its argument, treating it as implicitly symmetric
# and overwriting it in the process.
function Kirstine.criterion_integrand!(
    tnim::AbstractMatrix,
    is_inv::Bool,
    dc::DexpCriterion,
)
    sgn = is_inv ? -1 : 1
    ld = Kirstine.log_det!(tnim)
    # `tnim` could be singular. In such a case the corresponding design measure can't be a solution.
    # Hence we do not want to accidentally flip the sign then.
    cv = ld != -Inf ? sgn * ld : ld
    t = size(tnim, 1)
    return exp(cv / t)
end

function Kirstine.gateaux_constants(
    dc::DexpCriterion,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    trafo::Identity,
    na::NormalApproximation,
)
    tc = Kirstine.trafo_constants(trafo, pk)
    # For M(ζ,θ) ∈ S_+^r we need
    # A     = (1/r) det(M(ζ,θ))^{1/r} M(ζ,θ)^{-1}
    # tr(B) = det(M(ζ,θ))^{1/r}
    r = Kirstine.codomain_dimension(trafo, pk)
    # Compute M^{-1} as Symmetric matrices.
    inv_M = [inv(informationmatrix(d, m, cp, p, na)) for p in pk.p]
    # Since we already have M(ζ,θ)^{-1}, we compute tr(B) more numerically stable as
    # exp(-log(det(M(ζ,θ)^{-1}) / r)).
    # We don't use log_det!() since we don't want to modify inv_M.
    tr_B = map(iM -> exp(-log(det(iM)) / r), inv_M)
    A = map((iM, trB) -> iM * trB / r, inv_M, tr_B)
    return Kirstine.GCPriorSample(A, tr_B)
end

function Kirstine.gateaux_constants(
    dc::DexpCriterion,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    trafo::DeltaMethod,
    na::NormalApproximation,
)
    # For the
    #  * original NIM M(ζ,θ) ∈ S_+^r,
    #  * transformed NIM M_T(ζ,θ) ∈ S_+^t,
    #  * and transformation T: ℝ^r → ℝ^t with Jacobian matrix DT,
    # we need
    # A     = (1/t) det(M(ζ,θ))^{1/t} M(ζ,θ)^{-1} DT' M_T(ζ,θ) DT M(ζ,θ)^{-1}
    # tr(B) = det(M(ζ,θ))^{1/t}
    tc = Kirstine.trafo_constants(trafo, pk)
    t = Kirstine.codomain_dimension(trafo, pk)
    # This computes M^{-1} as Symmetric matrices.
    inv_M = [inv(informationmatrix(d, m, cp, p, na)) for p in pk.p]
    tr_B = map(iM -> exp(-log(det(iM)) / t), inv_M)
    # Note that these A will be dense.
    A = map(inv_M, tc.jm, tr_B) do iM, DT, trB
        # compute (M_T(ζ,θ))^{-1} from M(ζ,θ)^{-1}.
        iMT = DT * iM * DT'
        # Instead of an explicit inversion, we solve (M_T(ζ,θ))^{-1} X = DT for X.
        C = DT' * (iMT \ DT)
        return iM * C * iM * trB / t
    end
    return Kirstine.GCPriorSample(A, tr_B)
end
```

## Example

For illustration, we re-use the [discrete prior](discrete-prior.md) example.

```@example main
using Random, Plots

@simple_model SigEmax dose
@simple_parameter SigEmax e0 emax ed50 h

function Kirstine.jacobianmatrix!(
    jm,
    m::SigEmaxModel,
    c::SigEmaxCovariate,
    p::SigEmaxParameter,
)
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

prior = PriorSample(
    [SigEmaxParameter(e0 = 1, emax = 2, ed50 = 0.4, h = h) for h in 1:4],
    [0.1, 0.3, 0.4, 0.2],
)
nothing # hide
```

We compute D-optimal designs under both variants of the criterion,
once for estimating the full parameter,
and once for only estimating `ed50` and `h`.

```@example main
dp1a = DesignProblem(
    criterion = DexpCriterion(),
    region = DesignInterval(:dose => (0, 1)),
    model = SigEmaxModel(sigma = 1),
    covariate_parameterization = JustCopy(:dose),
    prior_knowledge = prior,
)

dp1b = DesignProblem(
    criterion = DCriterion(),
    region = DesignInterval(:dose => (0, 1)),
    model = SigEmaxModel(sigma = 1),
    covariate_parameterization = JustCopy(:dose),
    prior_knowledge = prior,
)

dp2a = DesignProblem(
    criterion = DexpCriterion(),
    region = DesignInterval(:dose => (0, 1)),
    model = SigEmaxModel(sigma = 1),
    covariate_parameterization = JustCopy(:dose),
    prior_knowledge = prior,
    transformation = DeltaMethod(p -> [0 0 1 0; 0 0 0 1]),
)

dp2b = DesignProblem(
    criterion = DCriterion(),
    region = DesignInterval(:dose => (0, 1)),
    model = SigEmaxModel(sigma = 1),
    covariate_parameterization = JustCopy(:dose),
    prior_knowledge = prior,
    transformation = DeltaMethod(p -> [0 0 1 0; 0 0 0 1]),
)

strategy = DirectMaximization(
    optimizer = Pso(iterations = 50, swarmsize = 100),
    prototype = equidistant_design(region(dp1a), 8),
)

Random.seed!(31415)
s1a, r1a = solve(dp1a, strategy, maxweight = 1e-3, maxdist = 1e-2)
Random.seed!(31415)
s1b, r1b = solve(dp1b, strategy, maxweight = 1e-3, maxdist = 1e-2)
Random.seed!(31415)
s2a, r2a = solve(dp2a, strategy, maxweight = 1e-3, maxdist = 1e-2)
Random.seed!(31415)
s2b, r2b = solve(dp2b, strategy, maxweight = 1e-3, maxdist = 1e-2)

gd1 = plot(
    plot_gateauxderivative(s1a, dp1a; title = "1a"),
    plot_gateauxderivative(s2a, dp2a; title = "2a"),
    plot_gateauxderivative(s1b, dp1b; title = "1b"),
    plot_gateauxderivative(s2b, dp2b; title = "2b"),
)
savefig(gd1, "extend-criterion-gd1.png") # hide
nothing # hide
```

```@setup main
s1a == DesignMeasure(
 [0.0] => 0.17605313466824343,
 [0.03498798414860223] => 0.08344304160813153,
 [0.2616383032984309] => 0.2456988438742072,
 [0.49666377406537887] => 0.2450846449152469,
 [0.9999998119304254] => 0.2497203349341709,
) || !check_results || error("not the expected result\n", s1a)
```

```@setup main
s2a == DesignMeasure(
 [0.0] => 0.14765387506515712,
 [0.029776312843391325] => 0.06478852276124815,
 [0.27329979413971395] => 0.2954574699564492,
 [0.4814631852647288] => 0.2970304648464117,
 [1.0] => 0.1950696673707337,
) || !check_results || error("not the expected result\n", s2a)
```

```@setup main
s1b == DesignMeasure(
 [0.0] => 0.17962034176078864,
 [0.04461948295716912] => 0.0934736269518622,
 [0.25627704567921966] => 0.23934177101306972,
 [0.4958739526513957] => 0.23846947150228448,
 [1.0] => 0.24909478877199498,
) || !check_results || error("not the expected result\n", s1b)
```

```@setup main
s2b == DesignMeasure(
 [0.0] => 0.1426449446174471,
 [0.046773454616330874] => 0.08797635657487009,
 [0.2635807148765203] => 0.29015625028527536,
 [0.47936857346476985] => 0.2848706752086477,
 [1.0] => 0.1943517733137599,
) || !check_results || error("not the expected result\n", s2b)
```

![](extend-criterion-gd1.png)

The solutions under both D-criterion variants are rather similar

```@example main
[reduce(vcat, points(s2a)) reduce(vcat, points(s2b))]
```

```@example main
[reduce(vcat, points(s2a)) reduce(vcat, points(s2b))]
```

but not the same

```@example main
gd2 = plot(
    plot_gateauxderivative(s1a, dp1b; title = "s1a for dp1b", legend = nothing),
    plot_gateauxderivative(s2a, dp2b; title = "s2a for dp2b", legend = nothing),
    plot_gateauxderivative(s1b, dp1a; title = "s1b for dp1a", legend = nothing),
    plot_gateauxderivative(s2b, dp2a; title = "s2b for dp2a", legend = nothing),
)
savefig(gd2, "extend-criterion-gd2.png") # hide
nothing # hide
```

![](extend-criterion-gd2.png)

[^MN99]: Jan R. Magnus and Heinz Neudecker (1999). Matrix differential calculus with applications in statistics and econometrics. Wiley. [doi:10.1002/9781119541219](https://doi.org/10.1002/9781119541219)
