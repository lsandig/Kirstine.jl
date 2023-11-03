# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## solve design problem with exchange algorithm ##

"""
    Exchange <: ProblemSolvingStrategy

Find an optimal design by ascending the Gateaux derivative of the objective function.

This is a variant of the basic idea in [^YBT13].

[^YBT13]: Min Yang, Stefanie Biedermann, Elina Tang (2013). On optimal designs for nonlinear models: a general and efficient algorithm. Journal of the American Statistical Association, 108(504), 1411–1420. [doi:10.1080/01621459.2013.806268](http://dx.doi.org/10.1080/01621459.2013.806268)
"""
struct Exchange{Tod<:Optimizer,Tow<:Optimizer,Td<:AbstractDict{Symbol,Any}} <:
       ProblemSolvingStrategy
    od::Tod
    ow::Tow
    steps::Int64
    candidate::DesignMeasure
    simplify_args::Td
    @doc """
        Exchange(;
            od::Optimizer,
            ow::Optimizer,
            steps::Integer,
            candidate::DesignMeasure,
            simplify_args::Dict,
        )

    Improve the `candidate` design by repeating the following `steps` times,
    starting with `r = candidate`:

     1. [`simplify`](@ref) the design `r`, passing along `sargs`.
     2. Use the optimizer `od` to find the direction (one-point design / Dirac measure) `d`
        of highest [`gateauxderivative`](@ref) at `r`.
        The vector of `prototypes` that is used for initializing `od`
        is constructed from one-point designs at the design points of `r`.
        See the [`Optimizer`](@ref)s for algorithm-specific details.
     3. Use the optimizer `ow` to re-calculcate optimal weights
        of a [`mixture`](@ref) of `r` and `d` for the [`DesignCriterion`](@ref).
        This is implemented as a call to [`solve`](@ref)
        with the [`DirectMaximization`](@ref) strategy
        and with all of the design points of `r` kept fixed.
     4. Set `r` to the result of step 3.

    The return value of [`solve`](@ref) for this strategy is an [`ExchangeResult`](@ref).
    """
    function Exchange(;
        od::Tod,
        ow::Tow,
        steps::Integer,
        candidate::DesignMeasure,
        simplify_args::Td = Dict{Symbol,Any}(),
    ) where {Tod<:Optimizer,Tow<:Optimizer,Td<:AbstractDict{Symbol,Any}}
        new{Tod,Tow,Td}(od, ow, steps, candidate, simplify_args)
    end
end

"""
    ExchangeResult <: ProblemSolvingResult

Wraps the [`OptimizationResult`](@ref)s from the individual steps.

See also [`optimization_results_direction`](@ref), [`optimization_results_weight`](@ref).
"""
struct ExchangeResult{
    S<:OptimizerState{DesignMeasure,SignedMeasure},
    T<:OptimizerState{DesignMeasure,SignedMeasure},
} <: ProblemSolvingResult
    ord::Vector{OptimizationResult{DesignMeasure,SignedMeasure,S}}
    orw::Vector{OptimizationResult{DesignMeasure,SignedMeasure,T}}
end

function maximizer(er::ExchangeResult)
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
    (; candidate, ow, od, steps, simplify_args) = strategy
    check_compatible(candidate, region(dp))
    pk = prior_knowledge(dp)
    cp = covariate_parameterization(dp)
    m = model(dp)
    tc = precalculate_trafo_constants(transformation(dp), pk)
    wm = WorkMatrices(
        1, # Dirac design measure corresponds to single covariate
        unit_length(m),
        parameter_dimension(pk),
        codomain_dimension(tc),
    )
    c = allocate_initialize_covariates(one_point_design(points(candidate)[1]), m, cp)
    constraints = DesignConstraints(region(dp), [false], [false])
    res = candidate
    or_pairs = map(1:(steps)) do i
        res = simplify(res, dp; simplify_args...)
        dir_prot = map(one_point_design, points(simplify_drop(res, 0)))
        gc = gateaux_constants(criterion(dp), res, m, cp, pk, tc, normal_approximation(dp))
        # find direction of steepest ascent
        gd(d) = gateauxderivative!(wm, c, gc, d, m, cp, pk, normal_approximation(dp))
        or_gd = optimize(od, gd, dir_prot, constraints; trace_state = trace_state)
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
        wstr = DirectMaximization(; optimizer = ow, prototype = res, fixedpoints = 1:K)
        _, rw = solve(dp, wstr; trace_state = trace_state, simplify_args...)
        res = maximizer(rw)
        return or_gd, optimization_result(rw)
    end
    ors_d = map(o -> o[1], or_pairs)
    ors_w = map(o -> o[2], or_pairs)
    return ExchangeResult(ors_d, ors_w)
end
