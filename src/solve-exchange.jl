# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## solve design problem with exchange algorithm ##

"""
    Exchange <: ProblemSolvingStrategy

Find an optimal design by ascending the Gateaux derivative of the objective function.

This is a variant of the basic idea in [^YBT13].

[^YBT13]: Min Yang, Stefanie Biedermann, Elina Tang (2013). On optimal designs for nonlinear models: a general and efficient algorithm. Journal of the American Statistical Association, 108(504), 1411–1420. [doi:10.1080/01621459.2013.806268](http://dx.doi.org/10.1080/01621459.2013.806268)
"""
struct Exchange{Tod<:Optimizer,Tow<:Optimizer,Td<:AbstractDict{Symbol,<:Any}} <:
       ProblemSolvingStrategy
    optimizer_direction::Tod
    optimizer_weight::Tow
    steps::Int64
    candidate::DesignMeasure
    simplify_args::Td
    @doc """
        Exchange(;
            optimizer_direction::Optimizer,
            optimizer_weight::Optimizer,
            steps::Integer,
            candidate::DesignMeasure,
            simplify_args = Dict{Symbol,Any}(),
        )

    Improve the `candidate` design by repeating the following `steps` times,
    starting with `r = candidate`:

     1. [`simplify`](@ref) the design `r`, passing along `sargs`.
     2. Use `optimizer_direction` to find the direction (one-point design / Dirac measure) `d`
        of highest [`gateauxderivative`](@ref) at `r`.
        The vector of `prototypes` that is used for initializing `optimizer_direction`
        is constructed from one-point designs at the design points of `r`.
        See the [`Optimizer`](@ref)s for algorithm-specific details.
     3. Use `optimizer_weight` to re-calculcate optimal weights
        of a [`mixture`](@ref) of `r` and `d` for the [`DesignCriterion`](@ref).
        This is implemented as a call to [`solve`](@ref)
        with the [`DirectMaximization`](@ref) strategy
        and with all of the design points of `r` kept fixed.
     4. Set `r` to the result of step 3.

    The return value of [`solve`](@ref) for this strategy is an [`ExchangeResult`](@ref).
    """
    function Exchange(;
        optimizer_direction::Tod,
        optimizer_weight::Tow,
        steps::Integer,
        candidate::DesignMeasure,
        simplify_args::Td = Dict{Symbol,Any}(),
    ) where {Tod<:Optimizer,Tow<:Optimizer,Td<:AbstractDict{Symbol,<:Any}}
        new{Tod,Tow,Td}(
            optimizer_direction,
            optimizer_weight,
            steps,
            candidate,
            simplify_args,
        )
    end
end

"""
    ExchangeResult <: ProblemSolvingResult

Wraps the [`OptimizationResult`](@ref)s from the individual steps.

See also
[`solution(::ExchangeResult)`](@ref),
[`optimization_results_direction`](@ref),
[`optimization_results_weight`](@ref).
"""
struct ExchangeResult{
    S<:OptimizerState{DesignMeasure,SignedMeasure},
    T<:OptimizerState{DesignMeasure,SignedMeasure},
} <: ProblemSolvingResult
    ord::Vector{OptimizationResult{DesignMeasure,SignedMeasure,S}}
    orw::Vector{OptimizationResult{DesignMeasure,SignedMeasure,T}}
end

"""
    solution(er::ExchangeResult)

Return the best candidate found.
"""
function solution(er::ExchangeResult)
    return er.orw[end].maximizer
end

"""
    optimization_results_direction(er::ExchangeResult)

Get the vector of [`OptimizationResult`](@ref)s from the direction steps.
"""
function optimization_results_direction(er::ExchangeResult)
    return er.ord
end

"""
    optimization_results_weight(er::ExchangeResult)

Get the full [`OptimizationResult`](@ref)s from the re-weighting steps.
"""
function optimization_results_weight(er::ExchangeResult)
    return er.orw
end

function solve_with(dp::DesignProblem, strategy::Exchange, trace_state::Bool)
    (; candidate, optimizer_weight, optimizer_direction, steps, simplify_args) = strategy
    check_compatible(candidate, region(dp))
    w = allocate_workspaces(one_point_design(points(candidate)[1]), dp)
    constraints = DesignConstraints(region(dp), [false], [false])
    res = candidate
    or_pairs = map(1:(steps)) do i
        res = simplify(res, dp; simplify_args...)
        dir_prot = map(one_point_design, points(simplify_drop(res, 0)))
        gc = gateaux_constants(
            criterion(dp),
            res,
            model(dp),
            covariate_parameterization(dp),
            prior_knowledge(dp),
            transformation(dp),
            normal_approximation(dp),
        )
        # find direction of steepest ascent
        or_gd = optimize(
            optimizer_direction,
            d -> gateauxderivative!(w, d, dp, gc),
            dir_prot,
            constraints;
            trace_state = trace_state,
        )
        d = or_gd.maximizer
        # append the new atom
        K = numpoints(res)
        if points(d)[1] in points(simplify_drop(res, 0))
            # effectivly run the reweighting from the last round for some more iterations
            res = mixture(0, d, res) # make sure new point is at index 1
            res = simplify_merge(res, region(dp), 0)
        else
            K += 1
            res = mixture(1 / K, d, res)
        end
        # optimize weights
        wstr = DirectMaximization(;
            optimizer = optimizer_weight,
            prototype = res,
            fixedpoints = 1:K,
        )
        _, rw = solve(dp, wstr; trace_state = trace_state, simplify_args...)
        res = solution(rw)
        return or_gd, optimization_result(rw)
    end
    ors_d = map(o -> o[1], or_pairs)
    ors_w = map(o -> o[2], or_pairs)
    return ExchangeResult(ors_d, ors_w)
end
