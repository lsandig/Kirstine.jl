# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## solve design problem with exchange algorithm ##

"""
    Exchange <: Kirstine.ProblemSolvingStrategy

Find an optimal design by ascending the Gateaux derivative of the objective function.

This is a variant of the basic idea in [^YBT13].

[^YBT13]: Min Yang, Stefanie Biedermann, and Elina Tang (2013). On optimal designs for nonlinear models: a general and efficient algorithm. Journal of the American Statistical Association, 108(504), 1411–1420. [doi:10.1080/01621459.2013.806268](http://dx.doi.org/10.1080/01621459.2013.806268)
"""
@kwdef struct Exchange{
    Tod<:ZerothOrderOptimizer,
    Tow<:Optimizer,
    Td<:AbstractDict{Symbol,<:Any},
} <: ProblemSolvingStrategy
    optimizer_direction::Tod
    optimizer_weight::Tow
    steps::Int64
    gateaux_tolerance::Float64
    initial_weight::Float64
    candidate::DesignMeasure
    simplify_args::Td = Dict{Symbol,Any}()
    @doc """
        Exchange(;
            optimizer_direction::Kirstine.ZerothOrderOptimizer,
            optimizer_weight::Kirstine.Optimizer,
            steps::Integer,
            gateaux_tolerance::Real,
            initial_weight::Real,
            candidate::DesignMeasure,
            simplify_args = Dict{Symbol,Any}(),
        )

    Improve the initial `candidate` design by repeating the following in a loop at most `steps` times:

     1. Recompute optimal weights for the current `candidate` while keeping all points fixed.
     2. [`simplify`](@ref) the current `candidate`, passing along `simplify_args`.
     3. Find the one-point design `d`
        towards which the [`gateauxderivative`](@ref) at the current `candidate` is largest.
        This uses `optimizer_direction`,
        initialized with one-point prototypes at the [`points`](@ref) of the current `candidate`.
        See the low-level [`Optimizer`](@ref)s for algorithm-specific details.
     4. If the maximum Gateaux derivative is smaller than `gateaux_tolerance`, stop early.
     5. Append the single point of `d` to the support of the current `candidate`
        and give it `initial_weight`, scaling down the already existing by `1-initial_weight`.
        In the special case `initial_weight=NaN`,
        the weight of the new point is set to `1/(1 + points(candidate))`.

    Note that depending on the chosen optimizers,
    this strategy can produce intermediate designs with a _lower_ value of the objective function
    than the `candidate` you started with.

    The return value of [`solve`](@ref) for this strategy is an [`ExchangeResult`](@ref).

    !!! note

        Before version 0.8.0 of Kirstine.jl,
        the order of the re-weighting step and the derivative-maximizing step were swapped
        and `initial_weight = NaN` had been hard-coded
    """
    function Exchange(
        optimizer_direction::Tod,
        optimizer_weight::Tow,
        steps::Integer,
        gateaux_tolerance::Real,
        initial_weight::Real,
        candidate::DesignMeasure,
        simplify_args::Td,
    ) where {Tod<:Optimizer,Tow<:Optimizer,Td<:AbstractDict{Symbol,<:Any}}
        steps < 1 && throw(ArgumentError("steps must be >= 1"))
        if !(isnan(initial_weight) || 0 <= initial_weight < 1)
            throw(ArgumentError("initial_weight must be in [0, 1) or NaN"))
        end
        gateaux_tolerance < 0 && throw(ArgumentError("gateaux_tolerance must be >= 0"))
        new{Tod,Tow,Td}(
            optimizer_direction,
            optimizer_weight,
            steps,
            gateaux_tolerance,
            initial_weight,
            candidate,
            simplify_args,
        )
    end
end

"""
    Kirstine.ExchangeResult <: Kirstine.ProblemSolvingResult

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
    solution(er::Kirstine.ExchangeResult)

Return the best candidate found.
"""
solution(er::ExchangeResult) = er.orw[end].maximizer

"""
    optimization_results_direction(er::Kirstine.ExchangeResult)

Get the vector of [`OptimizationResult`](@ref)s from the direction steps.
"""
optimization_results_direction(er::ExchangeResult) = er.ord

"""
    optimization_results_weight(er::Kirstine.ExchangeResult)

Get the full [`OptimizationResult`](@ref)s from the re-weighting steps.
"""
optimization_results_weight(er::ExchangeResult) = return er.orw

"""
    total_runtime(er::Kirstine.ExchangeResult)

Get the total time spent during optimization.
"""
function total_runtime(er::ExchangeResult)
    return total_runtime_direction(er) + total_runtime_weight(er)
end

"""
    total_runtime_direction(er::Kirstine.ExchangeResult)

Get the total time spent during the direction steps.
"""
function total_runtime_direction(er::ExchangeResult)
    return mapreduce(or -> or.seconds_elapsed, +, er.ord)
end

"""
    total_runtime_weight(er::Kirstine.ExchangeResult)

Get the total time spent during the re-weighting steps.
"""
function total_runtime_weight(er::ExchangeResult)
    return mapreduce(or -> or.seconds_elapsed, +, er.orw)
end

function Base.show(io::IO, ::MIME"text/plain", er::ExchangeResult)
    println(io, "ExchangeResult")
    println(io, "* final objective: ", er.orw[end].maximum)
    println(io, "* final max Gateaux derivative: ", er.ord[end].maximum)
    println(io, "* runtime total: ", prettydifftime(total_runtime(er)))
    println(io, "      direction: ", prettydifftime(total_runtime_direction(er)))
    println(io, "         weight: ", prettydifftime(total_runtime_weight(er)))
    println(io, "* steps: ", length(er.ord))
    println(io, "* raw solution: ")
    show(io, MIME("text/plain"), solution(er))
    println(io, "\n* diagnostic information:")
    println(io, "    optimization_results_direction(<this object>)")
    print(io, "    optimization_results_weight(<this object>)")
end

function solve_with(adp::AbstractDesignProblem, strategy::Exchange, trace_state::Bool)
    check_compatible(strategy.candidate, region(adp))
    function step_weight(candidate::DesignMeasure)
        dms = DirectMaximization(;
            optimizer = strategy.optimizer_weight,
            prototype = candidate,
            fixedpoints = 1:numpoints(candidate),
        )
        # Since we simplify explicitly later and want to keep the order of the points,
        # we ignore the first return value of solve().
        _, dmr = solve(adp, dms; trace_state = trace_state)
        return optimization_result(dmr)
    end
    function step_direction(candidate::DesignMeasure)
        wk = allocate_workspace(one_point_design(first(points(strategy.candidate))), adp)
        dc = derivative_constants(candidate, adp)
        or = optimize(
            d -> gateauxderivative!(wk, d, adp, dc),
            strategy.optimizer_direction,
            map(one_point_design, points(candidate)),
            DesignConstraints(region(adp), [true], [false]);
            trace_state = trace_state,
        )
        return or
    end
    function append_point(candidate::DesignMeasure, toward::DesignMeasure)
        # The mass point of the one-point design towards which the Gateaux derivative is
        # maximal can either be already in the support of the candidate (then we don't have
        # to do anything), or it can be new (then we append it to the support, with the
        # indicated inital_weight).
        if !(only(points(toward)) in points(candidate))
            L = numpoints(candidate)
            w = isnan(strategy.initial_weight) ? 1 / (1 + L) : strategy.initial_weight
            candidate = mixture(1 - w, candidate, toward)
        end
        return candidate
    end
    function full_step(candidate::DesignMeasure, i::Integer)
        orw = try
            step_weight(candidate)
        catch e
            msg1 = "step $i: singular exception in adaptive step-size computation."
            msg2 = "It might help to increase the initial smoothness estimate."
            msg3 = "Current values of the relevant variables are:"
            msg = "$msg1\n$msg2\n$msg3"
            if e isa AdaptiveStepsizeException && e.original_exception isa SingularException
                @error msg e.initial_smooothness_estimate e.M e.gamma
            end
            rethrow(e)
        end
        candidate = simplify(orw.maximizer, adp; strategy.simplify_args...)
        ord = try
            step_direction(candidate)
        catch e
            msg = "step $i: candidate design is singular"
            e isa SingularException ? error(msg) : rethrow(e)
        end
        candidate = append_point(candidate, ord.maximizer)
        return orw, ord, ord.maximum < strategy.gateaux_tolerance, candidate
    end
    i = 1
    orw, ord, is_close_enough, candidate = full_step(strategy.candidate, i)
    orw_trace, ord_trace = [orw], [ord]
    while !is_close_enough && i < strategy.steps
        i += 1
        orw, ord, is_close_enough, candidate = full_step(candidate, i)
        push!(orw_trace, orw)
        push!(ord_trace, ord)
        is_close_enough && break
    end
    return ExchangeResult(ord_trace, orw_trace)
end
