# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## solve design problem with exchange algorithm ##

"""
    Exchange <: ProblemSolvingStrategy

Find an optimal design by ascending the Gateaux derivative of the objective function.

This is a variant of the basic idea in [^YBT13].

[^YBT13]: Min Yang, Stefanie Biedermann, Elina Tang (2013). On optimal designs for nonlinear models: a general and efficient algorithm. Journal of the American Statistical Association, 108(504), 1411–1420. [doi:10.1080/01621459.2013.806268](http://dx.doi.org/10.1080/01621459.2013.806268)
"""
struct Exchange{Tod<:ZerothOrderOptimizer,Tow<:Optimizer,Td<:AbstractDict{Symbol,<:Any}} <:
       ProblemSolvingStrategy
    optimizer_direction::Tod
    optimizer_weight::Tow
    steps::Int64
    candidate::DesignMeasure
    simplify_args::Td
    @doc """
        Exchange(;
            optimizer_direction::ZerothOrderOptimizer,
            optimizer_weight::Kirstine.Optimizer,
            steps::Integer,
            candidate::DesignMeasure,
            simplify_args = Dict{Symbol,Any}(),
        )

    Improve the initial `candidate` design by repeating the following loop `steps` times:

     1. [`simplify`](@ref) the current `candidate`,
        passing along `simplify_args`.
     2. Find the direction (one-point design) `d`
        in which the [`gateauxderivative`](@ref) at the current `candidate` is highest.
        This uses `optimizer_direction`,
        initialized with one-point prototypes at the [`points`](@ref) of the current `candidate`.
        See the low-level [`Optimizer`](@ref)s for algorithm-specific details.
     3. Append the single point of `d` to the current `candidate`.
     4. Recompute optimal weights for the current candidate.
        This is implemented as a call to [`solve`](@ref)
        with the [`DirectMaximization`](@ref) strategy
        where all of the points of the current `candidate` are kept fixed.

    Note that this strategy can produce intermediate designs with a _lower_ value of the objective function
    than the `candidate` you started with.

    The return value of [`solve`](@ref) for this strategy is an [`ExchangeResult`](@ref).
    """
    function Exchange(;
        optimizer_direction::Tod,
        optimizer_weight::Tow,
        steps::Integer,
        candidate::DesignMeasure,
        simplify_args::Td = Dict{Symbol,Any}(),
    ) where {Tod<:Optimizer,Tow<:Optimizer,Td<:AbstractDict{Symbol,<:Any}}
        if steps < 1
            throw(ArgumentError("steps must be >= 1"))
        end
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

function solve_with(adp::AbstractDesignProblem, strategy::Exchange, trace_state::Bool)
    check_compatible(strategy.candidate, region(adp))
    # The workspaces and the constraints are independent of the current candidate.
    point_for_dim = first(points(strategy.candidate))
    workspaces_direction = allocate_workspace(one_point_design(point_for_dim), adp)
    constraints_direction = DesignConstraints(region(adp), [true], [false])
    function onestep(i::Integer, candidate::DesignMeasure)
        # Simplify the result of the previous iteration.
        candidate = simplify(candidate, adp; strategy.simplify_args...)
        # Find the direction in which the Gateaux derivative is largest.
        prototypes_direction = map(one_point_design, points(candidate))
        dc_direction = try
            derivative_constants(candidate, adp)
        catch e
            if e isa SingularException
                throw(ArgumentError("step $(i): candidate design is singular"))
            else
                rethrow(e)
            end
        end
        or_gd = optimize(
            d -> gateauxderivative!(workspaces_direction, d, adp, dc_direction),
            strategy.optimizer_direction,
            prototypes_direction,
            constraints_direction;
            trace_state = trace_state,
        )
        # Append this point to the candidate.
        direction = or_gd.maximizer
        if points(direction)[1] in points(candidate)
            # The direction is alrady in the candidate support.
            # This means we will effectivly run the reweighting from the last round for some more iterations.
            # For consistency with the case where the direction is _not_ already in the support,
            # we want the point corresponding to `direction` to move to the front.
            # We achieve this by prepending it in a mixture (the weight does not matter)
            # and then relying on `simplify_merge` to delete the later occurrence.
            candidate = mixture(0, direction, candidate)
            candidate = simplify_merge(candidate, region(adp), 0)
        else
            # The direction is actually a new point, prepend it.
            candidate = mixture(1 / (1 + numpoints(candidate)), direction, candidate)
        end
        # Re-optimize the weights.
        wstr = DirectMaximization(;
            optimizer = strategy.optimizer_weight,
            prototype = candidate,
            fixedpoints = 1:numpoints(candidate),
        )
        # We discard the first return value and simplify manually at the top of the loop
        # since solve() sorts the points, and we don't want this.
        _, rw = solve(adp, wstr; trace_state = trace_state)
        return or_gd, optimization_result(rw)
    end
    ord, orw = onestep(1, strategy.candidate)
    ors_d = [ord]
    ors_w = [orw]
    for i in 2:(strategy.steps)
        last_candidate = ors_w[i - 1].maximizer
        ord, orw = onestep(i, last_candidate)
        push!(ors_d, ord)
        push!(ors_w, orw)
    end
    return ExchangeResult(ors_d, ors_w)
end
