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
            optimizer_direction::ZerothOrderOptimizer,
            optimizer_weight::Kirstine.Optimizer,
            steps::Integer,
            gateaux_tolerance::Real,
            initial_weight::Real,
            candidate::DesignMeasure,
            simplify_args = Dict{Symbol,Any}(),
        )

    Improve the initial `candidate` design by repeating the following loop at most `steps` times:

     1. [`simplify`](@ref) the current `candidate`,
        passing along `simplify_args`.
     2. Find the direction (one-point design) `d`
        in which the [`gateauxderivative`](@ref) at the current `candidate` is highest.
        This uses `optimizer_direction`,
        initialized with one-point prototypes at the [`points`](@ref) of the current `candidate`.
        See the low-level [`Optimizer`](@ref)s for algorithm-specific details.
     3. Append the single point of `d` to the support of the current `candidate`
        and give it `initial_weight`, scaling down the already existing by `1-initial_weight`.
        In the special case `initial_weight=NaN`,
        the weight of the new point is set to `1/(1+ points(candidate))`.
     4. Recompute optimal weights for the current `candidate` while keeping all point fixed.

    The loop is stopped early if the maximum of the Gateaux derivative after sub-step 2
    is smaller than `gateaux_tolerance`.
    However, sub-steps 1 through 4 are guaranteed to run at least once,
    even if the initial `candidate` is already optimal.

    Note that depending on the chosen optimizers,
    this strategy can produce intermediate designs with a _lower_ value of the objective function
    than the `candidate` you started with.

    The return value of [`solve`](@ref) for this strategy is an [`ExchangeResult`](@ref).

    !!! note

        Before version 0.8.0 of Kirstine.jl, `initial_weight = NaN` had been hard-coded.
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
        if steps < 1
            throw(ArgumentError("steps must be >= 1"))
        end
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
function solution(er::ExchangeResult)
    return er.orw[end].maximizer
end

"""
    optimization_results_direction(er::Kirstine.ExchangeResult)

Get the vector of [`OptimizationResult`](@ref)s from the direction steps.
"""
function optimization_results_direction(er::ExchangeResult)
    return er.ord
end

"""
    optimization_results_weight(er::Kirstine.ExchangeResult)

Get the full [`OptimizationResult`](@ref)s from the re-weighting steps.
"""
function optimization_results_weight(er::ExchangeResult)
    return er.orw
end

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
    println(io, "* penultimate max Gateaux derivative: ", er.ord[end].maximum)
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
    (;
        optimizer_direction,
        optimizer_weight,
        steps,
        gateaux_tolerance,
        initial_weight,
        candidate,
        simplify_args,
    ) = strategy
    check_compatible(candidate, region(adp))
    point_for_dim = first(points(candidate))
    workspace_direction = allocate_workspace(one_point_design(point_for_dim), adp)
    function step_direction(candidate::DesignMeasure)
        opds = map(one_point_design, points(candidate))
        dc_direction = derivative_constants(candidate, adp)
        fix_weights = DesignConstraints(region(adp), [true], [false])
        gd = d -> gateauxderivative!(workspace_direction, d, adp, dc_direction)
        or = optimize(gd, optimizer_direction, opds, fix_weights; trace_state = trace_state)
        return or
    end
    function step_weight(candidate::DesignMeasure, direction::DesignMeasure)
        # The mass point of the direction can either be already in the support of the
        # candidate, or it can be a new point. In both cases we want to prepend it to the
        # candidate's support. If it is already in the support, we prepend it with weight 0
        # and rely on simplify_merge() with a maxdist of 0 to merge the later occurrence
        # into the earlier one.
        if !(only(points(direction)) in points(candidate))
            if isnan(initial_weight)
                mixture_weight = 1 / (1 + numpoints(candidate))
            else
                mixture_weight = initial_weight
            end
            candidate = mixture(mixture_weight, direction, candidate)
        else
            candidate = simplify_merge(mixture(0, direction, candidate), region(adp), 0)
        end
        dms = DirectMaximization(;
            optimizer = optimizer_weight,
            prototype = candidate,
            fixedpoints = 1:numpoints(candidate),
        )
        # Since we simplify explicitly later and want to keep the order of the points,
        # we ignore the first return value of solve().
        _, dmr = solve(adp, dms; trace_state = trace_state)
        return optimization_result(dmr)
    end
    candidate = simplify(candidate, adp; simplify_args...)
    # computing the derivative constants can fail for a singular design
    ord = try
        step_direction(candidate)
    catch e
        ae = ArgumentError("step 1: candidate design is singular")
        e isa SingularException ? throw(ae) : rethrow(e)
    end
    ord_trace = [ord]
    orw = try
        step_weight(candidate, ord.maximizer)
    catch e
        if e isa AdaptiveStepsizeException && e.original_exception isa SingularException
            msg = """singular exception in adaptive step-size computation,
                     it might help to increase the initial smoothness estimate.
                     Current values of the relevant variables are:"""
            @error msg e.initial_smooothness_estimate e.M e.gamma
        end
        rethrow(e)
    end
    orw_trace = [orw]
    for i in 2:steps
        candidate = simplify(orw_trace[i - 1].maximizer, adp; simplify_args...)
        ord = try
            step_direction(candidate)
        catch e
            ae = ArgumentError("step $i: candidate design is singular")
            e isa SingularException ? throw(ae) : rethrow(e)
        end
        ord.maximum < gateaux_tolerance && break
        push!(ord_trace, ord)
        push!(orw_trace, step_weight(candidate, ord.maximizer))
    end
    return ExchangeResult(ord_trace, orw_trace)
end
