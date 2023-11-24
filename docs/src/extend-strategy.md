# New Problem Solving Strategy

```@setup main
check_results = true
```

On a high level, Kirstine.jl can use different [`ProblemSolvingStrategy`](@ref)s
to find the solution to a design problem.
The package implements [`DirectMaximization`](@ref) and a version of Fedorov's [`Exchange`](@ref) algorithm.
This vignette shows how to implement a custom strategy.

Our example will be a variant of direct maximization
that runs the given [`Optimizer`](@ref) multiple times,
using the best candidate from the current iteration
as the prototype for initializing the next one.
A further extension of this example could also include a check of the Gateaux derivative
for early stopping.

## Implementation

For a `Ts <: ProblemSolvingStrategy` we need a corresponding `Tr <: ProblemSolvingResult`,
plus methods for the following functions:

  - `solve_with(dp::DesignProblem, strategy::Ts, trace_state::Bool)`
    which does the actual work and returns a `Tr`.
    The flag `trace_state` should passed to the low-level [`Optimizer`](@ref).
  - `solution(res::Tr)` to extract the best solution that was found.

Optionally,
for producing a plot of the optimization progress,
we can also implement a [type recipe](https://docs.juliaplots.org/latest/recipes/#Type-Recipes:-Easy-drop-in-replacement-of-data-types) for `Tr`.
When it comes to plotting a Vector of [`OptimizationResult`](@ref)s,
Kirstine.jl already provides a basic recipe that we can reuse.

```@example main
using Kirstine, RecipesBase, Plots

struct MultipleRuns{T<:Optimizer} <: ProblemSolvingStrategy
    optimizer::T
    prototype::DesignMeasure
    steps::Int64
    fixedweights::Vector{Int64}
    fixedpoints::Vector{Int64}
    function MultipleRuns(;
        optimizer::T,
        prototype::DesignMeasure,
        steps::Integer,
        fixedweights::AbstractVector{<:Integer} = Int64[],
        fixedpoints::AbstractVector{<:Integer} = Int64[],
    ) where T<:Optimizer
        new{T}(optimizer, prototype, steps, fixedweights, fixedpoints)
    end
end

struct MultipleRunsResult{
    S<:Kirstine.OptimizerState{DesignMeasure,Kirstine.SignedMeasure},
} <: ProblemSolvingResult
    ors::Vector{OptimizationResult{DesignMeasure,Kirstine.SignedMeasure,S}}
end

function Kirstine.solution(mrr::MultipleRunsResult)
    return mrr.ors[end].maximizer
end

function Kirstine.solve_with(dp::DesignProblem, strategy::MultipleRuns, trace_state::Bool)
    # `Kirstine.optimize()` needs to know the constraints for the optimization problem.
    constraints = Kirstine.DesignConstraints(
        strategy.prototype,
        region(dp),
        strategy.fixedweights,
        strategy.fixedpoints,
    )
    # set up pre-allocated objects for the objective function to work on
    tc = Kirstine.trafo_constants(transformation(dp), prior_knowledge(dp))
    w = Kirstine.allocate_workspaces(strategy.prototype, dp)
    optimization_helper(prototype) = Kirstine.optimize(
        strategy.optimizer,
        d -> Kirstine.objective!(w, d, dp, tc),
        [prototype],
        constraints;
        trace_state = trace_state,
    )

    # initial run
    ors = [optimization_helper(strategy.prototype)]

    # subsequent runs
    for i in 2:(strategy.steps)
        new_prototype = ors[i - 1].maximizer
        or = optimization_helper(new_prototype)
        push!(ors, or)
    end
    return MultipleRunsResult(ors)
end

@recipe function f(r::MultipleRunsResult)
    return r.ors # just need to unwrap
end
```

## Example

Again, we use the [discrete prior](discrete-prior.md) example.

```@example main
using Kirstine, Random, Plots

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

dp = DesignProblem(
    criterion = DOptimality(),
    region = DesignInterval(:dose => (0, 1)),
    model = SigEmaxModel(sigma = 1),
    covariate_parameterization = JustCopy(:dose),
    prior_knowledge = prior,
)

strategy = MultipleRuns(
    optimizer = Pso(iterations = 20, swarmsize = 25),
    prototype = equidistant_design(region(dp), 8),
    steps = 10,
)

Random.seed!(31415)
s, r = solve(dp, strategy, minweight = 1e-3, mindist = 1e-2)
gd = plot_gateauxderivative(s, dp)
savefig(gd, "extend-strategy-gd.png") # hide
nothing # hide
```

```@setup main
s == DesignMeasure(
 [0.0] => 0.17737208040258384,
 [0.043966654447468785] => 0.09441200054874936,
 [0.25627719942704874] => 0.23990318667725455,
 [0.49521184488138004] => 0.23940518207234693,
 [0.99999937754188] => 0.24890755029906544,
) || !check_results || error("not the expected result\n", s1a)
```

![](extend-strategy-gd.png)

```@example main
pr = plot(r)
savefig(pr, "extend-strategy-pr.png") # hide
nothing # hide
```

![](extend-strategy-pr.png)

We that the solution was found after four steps.
