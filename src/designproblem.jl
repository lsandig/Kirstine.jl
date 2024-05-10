# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## implement AbstractDesignProblem interface for DesignProblem ##

"""
    DesignProblem <: AbstractDesignProblem

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
} <: AbstractDesignProblem
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
    - `criterion::DesignCriterion`
    - `region::DesignRegion`
    - `model::Model`
    - `covariate_parameterization::CovariateParameterization`
    - `prior_knowledge::PriorKnowledge`
    - `transformation::Transformation = Identity()`
    - `normal_approximation::NormalApproximation = FisherMatrix()`
    """
    function DesignProblem(;
        criterion::Tdc,
        region::Tdr,
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
            criterion,
            region,
            model,
            covariate_parameterization,
            prior_knowledge,
            transformation,
            normal_approximation,
        )
    end
end

"""
    criterion(dp::DesignProblem)

Get the [`DesignCriterion`](@ref) of `dp`.
"""
criterion(dp::DesignProblem) = dp.dc

"""
    region(dp::DesignProblem)

Get the [`DesignRegion`](@ref) of `dp`.
"""
region(dp::DesignProblem) = dp.dr

"""
    model(dp::DesignProblem)

Get the [`Model`](@ref) of `dp`.
"""
model(dp::DesignProblem) = dp.m

"""
    covariate_parameterization(dp::DesignProblem)

Get the [`CovariateParameterization`](@ref) of `dp`.
"""
covariate_parameterization(dp::DesignProblem) = dp.cp

"""
    prior_knowledge(dp::DesignProblem)

Get the [`PriorKnowledge`](@ref) of `dp`.
"""
prior_knowledge(dp::DesignProblem) = dp.pk

"""
    transformation(dp::DesignProblem)

Get the [`Transformation`](@ref) of `dp`.
"""
transformation(dp::DesignProblem) = dp.trafo

"""
    normal_approximation(dp::DesignProblem)

Get the [`NormalApproximation`](@ref) of `dp`.
"""
normal_approximation(dp::DesignProblem) = dp.na

struct DPWorkspace{Tmw<:ModelWorkspace,Tc<:Covariate} <: AbstractWorkspace
    nw::NIMWorkspace
    mw::Tmw
    c::Vector{Tc}
end

function allocate_workspace(d::DesignMeasure, dp::DesignProblem)
    nw = NIMWorkspace(
        parameter_dimension(prior_knowledge(dp)),
        codomain_dimension(transformation(dp), prior_knowledge(dp)),
    )
    mw = allocate_model_workspace(numpoints(d), model(dp), prior_knowledge(dp))
    c = implied_covariates(d, model(dp), covariate_parameterization(dp))
    return DPWorkspace(nw, mw, c)
end

struct DPObjectiveConstants{Ttc<:TrafoConstants} <: ObjectiveConstants
    tc::Ttc
end

function objective_constants(dp::DesignProblem)
    tc = trafo_constants(transformation(dp), prior_knowledge(dp))
    return DPObjectiveConstants(tc)
end

struct DPDerivativeConstants{Tgc<:GateauxConstants} <: DerivativeConstants
    gc::Tgc
end

function derivative_constants(at::DesignMeasure, dp::DesignProblem)
    gc = gateaux_constants(
        criterion(dp),
        at,
        model(dp),
        covariate_parameterization(dp),
        prior_knowledge(dp),
        transformation(dp),
        normal_approximation(dp),
    )
    return DPDerivativeConstants(gc)
end

function objective!(
    w::DPWorkspace,
    d::DesignMeasure,
    dp::DesignProblem,
    oc::DPObjectiveConstants,
)
    m = model(dp)
    update_covariates!(w.c, d, m, covariate_parameterization(dp))
    update_model_workspace!(w.mw, m, w.c)
    # When the information matrix is singular, the objective function is undefined. Lower
    # level calls may throw a PosDefException or a SingularException. This also means that
    # `d` can not be a solution to the maximization problem, hence we return negative
    # infinity in these cases.
    pk = prior_knowledge(dp)
    try
        acc = 0.0
        pk_w = weights(pk)
        pk_p = parameters(pk)
        for i in 1:length(pk_p)
            average_fishermatrix!(w.nw.r_x_r, w.mw, weights(d), m, w.c, pk_p[i])
            informationmatrix!(w.nw.r_x_r, normal_approximation(dp))
            w.nw.r_is_inv = false
            apply_transformation!(
                w.nw,
                transformation(dp),
                trafo_jacobianmatrix_for_index(oc.tc, i),
            )
            acc += pk_w[i] * criterion_functional!(w.nw.t_x_t, w.nw.t_is_inv, criterion(dp))
        end
        return acc
    catch e
        if isa(e, PosDefException) || isa(e, SingularException)
            return -Inf
        else
            rethrow(e)
        end
    end
end

function gateauxderivative!(
    w::DPWorkspace,
    direction::DesignMeasure,
    dp::DesignProblem,
    dc::DPDerivativeConstants,
)
    m = model(dp)
    update_covariates!(w.c, direction, m, covariate_parameterization(dp))
    update_model_workspace!(w.mw, m, w.c)
    pk = prior_knowledge(dp)
    acc = 0.0
    pk_w = weights(pk)
    pk_p = parameters(pk)
    for i in 1:length(pk_p)
        average_fishermatrix!(w.nw.r_x_r, w.mw, weights(direction), m, w.c, pk_p[i])
        informationmatrix!(w.nw.r_x_r, normal_approximation(dp))
        acc += pk_w[i] * gateaux_integrand(dc.gc.A[i], w.nw.r_x_r, dc.gc.tr_B[i])
    end
    return acc
end

"""
    simplify(d::DesignMeasure, dp::DesignProblem; maxweight = 0, maxdist = 0, uargs...)

A wrapper that calls
[`simplify_unique`](@ref),
[`simplify_merge`](@ref),
and [`simplify_drop`](@ref) in that order.
"""
function simplify(d::DesignMeasure, dp::DesignProblem; maxweight = 0, maxdist = 0, uargs...)
    d = simplify_unique(d, region(dp), model(dp), covariate_parameterization(dp); uargs...)
    d = simplify_merge(d, region(dp), maxdist)
    d = simplify_drop(d, maxweight)
    return d
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

# same design problem as `dp`, but with criterion replaced by `DCriterion()`.
function as_doptimality_problem(dp::DesignProblem)
    dpd = DesignProblem(;
        criterion = DCriterion(),
        model = model(dp),
        region = region(dp),
        covariate_parameterization = covariate_parameterization(dp),
        prior_knowledge = prior_knowledge(dp),
        transformation = transformation(dp),
        normal_approximation = normal_approximation(dp),
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
    t = codomain_dimension(transformation(dp1), prior_knowledge(dp1))
    if t != codomain_dimension(transformation(dp2), prior_knowledge(dp2))
        throw(DimensionMismatch("dimensions of transformed parameters must match"))
    end
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
        so this function ignores the type of `criterion(dp)`.
"""
function shannon_information(d::DesignMeasure, dp::DesignProblem, n::Integer)
    dpd = as_doptimality_problem(dp)
    t = codomain_dimension(transformation(dp), prior_knowledge(dp))
    return (t / 2) * (log(n) - 1 + log(2 * pi)) + 0.5 * objective(d, dpd)
end
