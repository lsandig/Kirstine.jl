# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## user-visible functions for design problems and for solving them ##

"""
    DesignProblem

A `DesignProblem` has 7 components:

  - a [`DesignCriterion`](@ref),
  - a [`DesignRegion`](@ref),
  - a [`Model`](@ref),
  - a [`CovariateParameterization`](@ref),
  - some [`PriorKnowledge`](@ref),
  - a [`Transformation`](@ref),
  - and a [`NormalApproximation`](@ref).

See also [`solve`](@ref),
and the [mathematical background](math.md#Design-Problems).
"""
struct DesignProblem{
    Tdc<:DesignCriterion,
    Tdr<:DesignRegion,
    Tm<:Model,
    Tcp<:CovariateParameterization,
    Tpk<:PriorKnowledge,
    Tt<:Transformation,
    Tna<:NormalApproximation,
}
    dc::Tdc
    dr::Tdr
    m::Tm
    cp::Tcp
    pk::Tpk
    trafo::Tt
    na::Tna
    @doc """
        DesignProblem(<keyword arguments>)

    Construct a design problem with some sensible defaults.

    # Arguments
    - `design_criterion::DesignCriterion`
    - `design_region::DesignRegion`
    - `model::Model`
    - `covariate_parameterization::CovariateParameterization`
    - `prior_knowledge::PriorKnowledge`
    - `transformation::Transformation = Identity()`
    - `normal_approximation::NormalApproximation = FisherMatrix()`
    """
    function DesignProblem(;
        design_criterion::Tdc,
        design_region::Tdr,
        model::Tm,
        covariate_parameterization::Tcp,
        prior_knowledge::Tpk,
        transformation::Tt = Identity(),
        normal_approximation::Tna = FisherMatrix(),
    ) where {
        Tdc<:DesignCriterion,
        Tdr<:DesignRegion,
        Tm<:Model,
        Tcp<:CovariateParameterization,
        Tpk<:PriorKnowledge,
        Tt<:Transformation,
        Tna<:NormalApproximation,
    }
        new{Tdc,Tdr,Tm,Tcp,Tpk,Tt,Tna}(
            design_criterion,
            design_region,
            model,
            covariate_parameterization,
            prior_knowledge,
            transformation,
            normal_approximation,
        )
    end
end

"""
    solve(
        dp::DesignProblem,
        strategy::ProblemSolvingStrategy;
        trace_state = false,
        sargs...,
    )

Return a tuple `(d, r)`.

  - `d`: The best [`DesignMeasure`](@ref) found. As postprocessing, [`simplify`](@ref) is
    called with `sargs` and the design points are sorted with [`sort_designpoints`](@ref).

  - `r`: A subtype of [`ProblemSolvingResult`](@ref) that is specific to the strategy used.
    If `trace_state=true`, this object contains additional debugging information.
    The unsimplified version of `d` can be accessed as `maximizer(r)`.

See also [`DirectMaximization`](@ref), [`Exchange`](@ref).
"""
function solve(
    dp::DesignProblem,
    strategy::ProblemSolvingStrategy;
    trace_state = false,
    sargs...,
)
    or = solve_with(dp, strategy, trace_state)
    dopt = sort_designpoints(simplify(maximizer(or), dp; sargs...))
    # check that we did not accidentally simplify too much
    o_before = objective(maximizer(or), dp)
    o_after = objective(dopt, dp)
    rel_diff = (o_after - o_before) / abs(o_before)
    if rel_diff < -0.01
        @warn "simplification may have been too eager" o_before o_after rel_diff
    end
    return dopt, or
end

"""
    simplify(d::DesignMeasure, dp::DesignProblem; minweight = 0, mindist = 0, uargs...)

A wrapper that calls
[`simplify_drop`](@ref),
[`simplify_unique`](@ref),
and [`simplify_merge`](@ref).
"""
function simplify(d::DesignMeasure, dp::DesignProblem; minweight = 0, mindist = 0, uargs...)
    d = simplify_drop(d, minweight)
    d = simplify_unique(d, dp.dr, dp.m, dp.cp; uargs...)
    d = simplify_merge(d, dp.dr, mindist)
    return d
end

"""
    objective(d::DesignMeasure, dp::DesignProblem)

Evaluate the objective function.

See also the [mathematical background](math.md#Objective-Function).
"""
function objective(d::DesignMeasure, dp::DesignProblem)
    tc = precalculate_trafo_constants(dp.trafo, dp.pk)
    wm = WorkMatrices(unit_length(dp.m), parameter_dimension(dp.pk), codomain_dimension(tc))
    c = allocate_initialize_covariates(d, dp.m, dp.cp)
    return objective!(wm, c, dp.dc, d, dp.m, dp.cp, dp.pk, tc, dp.na)
end

"""
    gateauxderivative(
        at::DesignMeasure,
        directions::AbstractArray{DesignMeasure},
        dp::DesignProblem,
    )

Evaluate the Gateaux derivative for each direction.

Directions must be one-point design measures.

See also the [mathematical background](math.md#Gateaux-Derivative).
"""
function gateauxderivative(
    at::DesignMeasure,
    directions::AbstractArray{DesignMeasure},
    dp::DesignProblem,
)
    if any(d -> numpoints(d) != 1, directions)
        error("Gateaux derivatives are only implemented for one-point design directions")
    end
    tc = precalculate_trafo_constants(dp.trafo, dp.pk)
    wm = WorkMatrices(unit_length(dp.m), parameter_dimension(dp.pk), codomain_dimension(tc))
    gconst = try
        precalculate_gateaux_constants(dp.dc, at, dp.m, dp.cp, dp.pk, tc, dp.na)
    catch e
        if isa(e, SingularException)
            # undefined objective implies no well-defined derivative
            return fill(NaN, size(directions))
        else
            rethrow(e)
        end
    end
    cs = allocate_initialize_covariates(directions[1], dp.m, dp.cp)
    gd = map(directions) do d
        gateauxderivative!(wm, cs, gconst, d, dp.m, dp.cp, dp.pk, dp.na)
    end
    return gd
end

"""
    efficiency(d1::DesignMeasure, d2::DesignMeasure, dp::DesignProblem)
    efficiency(d1::DesignMeasure, d2::DesignMeasure, dp1::DesignProblem, dp2::DesignProblem)

Relative D-efficiency of `d1` to `d2`.

Note that for the method with two design problems,
the dimensions of the transformed parameters must be identical.
All other aspects are allowed to be different.

See also the [mathematical background](math.md#Efficiency).

!!! note

    This always computes D-efficiency, regardless of the criterion used in `dp`.
"""
function efficiency(d1::DesignMeasure, d2::DesignMeasure, dp::DesignProblem)
    return efficiency(d1, d2, dp, dp)
end

# same design problem as `dp`, but with criterion replaced by `DOptimality()`.
function as_doptimality_problem(dp::DesignProblem)
    dpd = DesignProblem(;
        design_criterion = DOptimality(),
        model = dp.m,
        design_region = dp.dr,
        covariate_parameterization = dp.cp,
        prior_knowledge = dp.pk,
        transformation = dp.trafo,
        normal_approximation = dp.na,
    )
    return dpd
end

function efficiency(
    d1::DesignMeasure,
    d2::DesignMeasure,
    dp1::DesignProblem,
    dp2::DesignProblem,
)
    # check that minimal requirements are met for efficiency to make sense
    tc1 = precalculate_trafo_constants(dp1.trafo, dp1.pk)
    tc2 = precalculate_trafo_constants(dp2.trafo, dp2.pk)
    if codomain_dimension(tc1) != codomain_dimension(tc2)
        throw(DimensionMismatch("dimensions of transformed parameters must match"))
    end
    t = codomain_dimension(tc1)
    # Take a shortcut via D criterion objective, which already gives the average
    # log-determinant of the transformed information matrices, and handles exceptions.
    # Note that the design region is irrelevant for relative efficiency.
    dp1d = as_doptimality_problem(dp1)
    dp2d = as_doptimality_problem(dp2)
    return exp((objective(d1, dp1d) - objective(d2, dp2d)) / t)
end

"""
    shannon_information(d::DesignMeasure, dp::DesignProblem, n::Integer)

Compute the approximate expected posterior Shannon information for an experiment
with `n` units of observation under design `d` and design problem `dp`.

See also the [mathematical background](math.md#Shannon-Information).

!!! note

     1. `d` is not [`apportion`](@ref)ed.
     2. Shannon information is motivated by D-optimal design,
        so this function ignores the criterion used in `dp`.
"""
function shannon_information(d::DesignMeasure, dp::DesignProblem, n::Integer)
    dpd = as_doptimality_problem(dp)
    # this is somewhat ugly, computing all constants is a bit unnecessary
    t = codomain_dimension(precalculate_trafo_constants(dp.trafo, dp.pk))
    return (t / 2) * (log(n) - 1 + log(2 * pi)) + 0.5 * objective(d, dpd)
end
