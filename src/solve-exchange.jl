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
        and with all of the designpoints of `r` kept fixed.
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

Contains vectors `ord` and `orw` of [`OptimizationResult`](@ref)s,
one for each direction finding and reweighting step.
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

function solve_with(dp::DesignProblem, strategy::Exchange, trace_state::Bool)
    (; candidate, ow, od, steps, simplify_args) = strategy
    check_compatible(candidate, dp.dr)
    tc = precalculate_trafo_constants(dp.trafo, dp.pk)
    wm = WorkMatrices(unit_length(dp.m), parameter_dimension(dp.pk), codomain_dimension(tc))
    c = allocate_initialize_covariates(
        one_point_design(candidate.designpoint[1]),
        dp.m,
        dp.cp,
    )
    constraints = DesignConstraints(dp.dr, [false], [false])
    res = candidate
    or_pairs = map(1:(steps)) do i
        res = simplify(res, dp; simplify_args...)
        dir_prot = map(one_point_design, designpoints(simplify_drop(res, 0)))
        gc = precalculate_gateaux_constants(dp.dc, res, dp.m, dp.cp, dp.pk, tc, dp.na)
        # find direction of steepest ascent
        gd(d) = gateauxderivative!(wm, c, gc, d, dp.m, dp.cp, dp.pk, dp.na)
        or_gd = optimize(od, gd, dir_prot, constraints; trace_state = trace_state)
        d = or_gd.maximizer
        # append the new atom
        K = length(res.weight)
        if d.designpoint[1] in designpoints(simplify_drop(res, 0))
            # effectivly run the reweighting from the last round for some more iterations
            res = mixture(0, d, res) # make sure new point is at index 1
            res = simplify_merge(res, dp.dr, 0)
        else
            K += 1
            res = mixture(1 / K, d, res)
        end
        # optimize weights
        wstr = DirectMaximization(; optimizer = ow, prototype = res, fixedpoints = 1:K)
        _, rw = solve(dp, wstr; trace_state = trace_state, simplify_args...)
        res = maximizer(rw)
        return or_gd, rw.or
    end
    ors_d = map(o -> o[1], or_pairs)
    ors_w = map(o -> o[2], or_pairs)
    return ExchangeResult(ors_d, ors_w)
end
