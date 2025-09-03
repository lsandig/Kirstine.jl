# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## implement AbstractDesignProblem interface for DesignProblem ##

"""
    DesignProblem <: Kirstine.AbstractDesignProblem

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
end

"""
    DesignProblem(<keyword arguments>)

Construct a design problem with some sensible defaults.

# Arguments

  - `criterion::Kirstine.DesignCriterion`
  - `region::Kirstine.DesignRegion`
  - `model::Kirstine.Model`
  - `covariate_parameterization::Kirstine.CovariateParameterization`
  - `prior_knowledge::Kirstine.PriorKnowledge`
  - `transformation::Kirstine.Transformation = Identity()`
  - `normal_approximation::Kirstine.NormalApproximation = FisherMatrix()`
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
    dp = DesignProblem{Tdc,Tdr,Tm,Tcp,Tpk,Tt,Tna}(
        criterion,
        region,
        model,
        covariate_parameterization,
        prior_knowledge,
        transformation,
        normal_approximation,
    )
    return dp
end

"""
    DesignProblem(prototype::DesignProblem; <keyword arguments>)

Construct a design problem with defaults from `prototype`.

All explicitly given keyword arguments will be part of the new design problem.
For every missing keyword argument,
the corresponding field from `prototype` is copied instead.
Note that the result does not share memory with `prototype`.

This function is useful for constructing variations of a given problem
without having to re-type a lot of code
or having to write helper functions.

# Examples

Consider some design problem

```julia
dp = DesignProblem(; criterion = DCriterion(), other_keyword_arguments...)
```

This constructs a similar problem where only the criterion differs:

```julia
DesignProblem(dp; criterion = ACriterion())
```
"""
function DesignProblem(
    dp::DesignProblem;
    criterion = deepcopy(criterion(dp)),
    region = deepcopy(region(dp)),
    model = deepcopy(model(dp)),
    covariate_parameterization = deepcopy(covariate_parameterization(dp)),
    prior_knowledge = deepcopy(prior_knowledge(dp)),
    transformation = deepcopy(transformation(dp)),
    normal_approximation = deepcopy(normal_approximation(dp)),
)
    modified_dp = DesignProblem(;
        criterion = criterion,
        region = region,
        model = model,
        covariate_parameterization = covariate_parameterization,
        prior_knowledge = prior_knowledge,
        transformation = transformation,
        normal_approximation = normal_approximation,
    )
    return modified_dp
end

function Base.:(==)(dp1::DesignProblem, dp2::DesignProblem)
    fields_are_equal =
        criterion(dp1) == criterion(dp2) &&
        region(dp1) == region(dp2) &&
        model(dp1) == model(dp2) &&
        covariate_parameterization(dp1) == covariate_parameterization(dp2) &&
        prior_knowledge(dp1) == prior_knowledge(dp2) &&
        transformation(dp1) == transformation(dp2) &&
        normal_approximation(dp1) == normal_approximation(dp2)
    return fields_are_equal
end

function Base.hash(dp::DesignProblem, h::UInt)
    h = hash(criterion(dp), h)
    h = hash(region(dp), h)
    h = hash(model(dp), h)
    h = hash(covariate_parameterization(dp), h)
    h = hash(prior_knowledge(dp), h)
    h = hash(transformation(dp), h)
    h = hash(normal_approximation(dp), h)
    return h
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

function Base.show(io::IO, ::MIME"text/plain", dp::DesignProblem)
    println(io, "DesignProblem")
    println(io, " - criterion: ", criterion(dp))
    println(io, " - region: ", region(dp))
    println(io, " - model: ", model(dp))
    println(io, " - covariate parameterization: ", covariate_parameterization(dp))
    print(io, " - prior knowledge: ")
    show(io, MIME("text/plain"), prior_knowledge(dp))
    println(io)
    println(io, " - transformation: ", transformation(dp))
    print(io, " - normal approximation: ", normal_approximation(dp))
end

struct DPWorkspace{Tmw<:ModelWorkspace,Tc<:Covariate} <: ProblemWorkspace
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

function gateaux_constants(
    d::DesignMeasure,
    dp::DesignProblem{Tc,Tr,Tm,Tcp,Tp,Tt,Tn},
) where {
    Tc<:DesignCriterion,
    Tr<:DesignRegion,
    Tm<:Model,
    Tcp<:CovariateParameterization,
    Tp<:PriorSample,
    Tt<:Transformation,
    Tn<:NormalApproximation,
}
    (; dc, dr, m, cp, pk, trafo, na) = dp
    t = codomain_dimension(trafo, pk)
    r = parameter_dimension(pk)
    numdraws = length(parameters(pk))
    gc = GCPriorSample([zeros(r, r) for _ in 1:numdraws], zeros(numdraws))
    mw = allocate_model_workspace(numpoints(d), m, pk)
    nw = NIMWorkspace(r, t)
    c = implied_covariates(d, m, cp)
    tc = trafo_constants(trafo, pk)
    gateaux_constants!(gc, mw, nw, c, dc, d, m, cp, pk, tc, na)
    return gc
end

struct DPDerivativeConstants{Tgc<:GateauxConstants} <: DerivativeConstants
    gc::Tgc
end

function derivative_constants(at::DesignMeasure, dp::DesignProblem)
    return DPDerivativeConstants(gateaux_constants(at, dp))
end

function derivative_constants!(
    dc::DPDerivativeConstants,
    w::DPWorkspace,
    d::DesignMeasure,
    dp::DesignProblem,
    oc::DPObjectiveConstants,
)
    cpar = covariate_parameterization(dp)
    pk = prior_knowledge(dp)
    cr = criterion(dp)
    na = normal_approximation(dp)
    gateaux_constants!(dc.gc, w.mw, w.nw, w.c, cr, d, model(dp), cpar, pk, oc.tc, na)
    return dc
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

function sensitivity!(
    w::DPWorkspace,
    direction::DesignMeasure,
    dp::DesignProblem,
    dc::DPDerivativeConstants,
)
    update_covariates!(w.c, direction, model(dp), covariate_parameterization(dp))
    update_model_workspace!(w.mw, model(dp), w.c)
    pk_w, pk_p = weights(prior_knowledge(dp)), parameters(prior_knowledge(dp))
    acc = 0.0
    dir_w = weights(direction)
    for i in 1:length(pk_p)
        average_fishermatrix!(w.nw.r_x_r, w.mw, dir_w, model(dp), w.c, pk_p[i])
        informationmatrix!(w.nw.r_x_r, normal_approximation(dp))
        acc += pk_w[i] * sensitivity_integrand(dc.gc.A[i], w.nw.r_x_r)
    end
    return acc
end

"""
    simplify(d::DesignMeasure, dp::DesignProblem; maxweight = 0, maxdist = 0, uargs...)

A wrapper that calls
[`simplify_unique`](@ref),
[`simplify_merge`](@ref) (with the design region of `dp`),
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

Relative efficiency of `d1` to `d2` under the corresponding problems.

The function with two design problems is only defined
if there is a sensible notion of efficiency for the
combination of the two [`DesignCriterion`](@ref) fields.
For the [`DCriterion`](@ref),
the dimensions of the transformed parameters must be identical.
Apart from that, the two design problems are allowed to be completely different.
The function with three arguments is a shorthand for when `dp1` and `dp2` are identical.

See also the [mathematical background](math.md#Efficiency).
"""
function efficiency(d1::DesignMeasure, d2::DesignMeasure, dp::DesignProblem)
    return efficiency(d1, d2, dp, dp)
end

function efficiency(
    d1::DesignMeasure,
    d2::DesignMeasure,
    dp1::DesignProblem{
        DCriterion,
        <:DesignRegion,
        <:Model,
        <:CovariateParameterization,
        <:PriorKnowledge,
        <:Transformation,
        FisherMatrix,
    },
    dp2::DesignProblem{
        DCriterion,
        <:DesignRegion,
        <:Model,
        <:CovariateParameterization,
        <:PriorKnowledge,
        <:Transformation,
        FisherMatrix,
    },
)
    # check that minimal requirements are met for efficiency to make sense
    t = codomain_dimension(transformation(dp1), prior_knowledge(dp1))
    if t != codomain_dimension(transformation(dp2), prior_knowledge(dp2))
        throw(DimensionMismatch("dimensions of transformed parameters must match"))
    end
    # Take a shortcut via D criterion objective, which already gives the average
    # log-determinant of the transformed information matrices, and handles exceptions.
    # Note that the design region is irrelevant for relative efficiency.
    return exp((objective(d1, dp1) - objective(d2, dp2)) / t)
end

function efficiency(
    d1::DesignMeasure,
    d2::DesignMeasure,
    dp1::DesignProblem{
        ACriterion,
        <:DesignRegion,
        <:Model,
        <:CovariateParameterization,
        <:PriorKnowledge,
        <:Transformation,
        FisherMatrix,
    },
    dp2::DesignProblem{
        ACriterion,
        <:DesignRegion,
        <:Model,
        <:CovariateParameterization,
        <:PriorKnowledge,
        <:Transformation,
        FisherMatrix,
    },
)
    return objective(d2, dp2) / objective(d1, dp1)
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
    dpd = DesignProblem(dp; criterion = DCriterion())
    t = codomain_dimension(transformation(dp), prior_knowledge(dp))
    return (t / 2) * (log(n) - 1 + log(2 * pi)) + 0.5 * objective(d, dpd)
end

function check_numerically_singular(d::DesignMeasure, dp::DesignProblem)
    par_singular = filter(parameters(prior_knowledge(dp))) do p
        na = normal_approximation(dp)
        im = informationmatrix(d, model(dp), covariate_parameterization(dp), p, na)
        return is_numerically_singular(im)
    end
    if !isempty(par_singular)
        msg = "some parameters have numerically singular information matrices"
        @warn msg dp d par_singular
    end
end
