# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## implement AbstractDesignProblem interface for CompositeDesignProblem ##

"""
    CompositeDesignProblem <: Kirstine.AbstractDesignProblem

A weighted construct of other abstract design problems.

See the [mathematical background](math.md#Composite-Design-Problems).
"""
struct CompositeDesignProblem{T<:AbstractDesignProblem} <: AbstractDesignProblem
    problems::Vector{T}
    weights::Vector{Float64}
    @doc """
        CompositeDesignProblem(problems, weights)

    Construct a composite design problem from a vector of [`AbstractDesignProblem`](@ref)s
    and corresponding non-negative weights.

    All `problems` must have the same [`DesignRegion`](@ref).

    `CompositeDesignProblem`s can be nested.

    See also [`problems`](@ref),
    [`weights(::CompositeDesignProblem)`](@ref),
    and [`region(::CompositeDesignProblem)`](@ref).
    """
    function CompositeDesignProblem(
        problems::AbstractVector{<:AbstractDesignProblem},
        weights::AbstractVector{<:Real},
    )
        if length(problems) != length(weights)
            throw(ArgumentError("number of problems must match number of weights"))
        elseif any(weights .< 0)
            throw(ArgumentError("problem weights must be non-negative"))
        elseif any(Ref(region(problems[1])) .!= region.(problems))
            throw(ArgumentError("all problems must have identical regions"))
        end
        return new{eltype(problems)}(problems, weights)
    end
end

"""
    problems(cdp::CompositeDesignProblem)

Get the vector of component design problems.
"""
problems(cdp::CompositeDesignProblem) = cdp.problems

"""
    weights(cdp::CompositeDesignProblem)

Get the weights associated with the component design problems.
"""
weights(cdp::CompositeDesignProblem) = cdp.weights

"""
    region(cdp::CompositeDesignProblems)

Get the common [`DesignRegion`](@ref) of the component design problems.
"""
function region(cdp::CompositeDesignProblem)
    return region(first(problems(cdp)))
end

struct CDPWorkspace{T<:AbstractWorkspace} <: AbstractWorkspace
    component_workspaces::Vector{T}
end

function allocate_workspace(d::DesignMeasure, cdp::CompositeDesignProblem)
    return CDPWorkspace(map(p -> allocate_workspace(d, p), problems(cdp)))
end

struct CDPObjectiveConstants{T<:ObjectiveConstants} <: ObjectiveConstants
    component_constants::Vector{T}
end

function objective_constants(cdp::CompositeDesignProblem)
    return CDPObjectiveConstants(map(objective_constants, problems(cdp)))
end

struct CDPDerivativeConstants{T<:DerivativeConstants} <: DerivativeConstants
    component_constants::Vector{T}
end

function derivative_constants(at::DesignMeasure, cdp::CompositeDesignProblem)
    return CDPDerivativeConstants(map(p -> derivative_constants(at, p), problems(cdp)))
end

function derivative_constants!(
    cdc::CDPDerivativeConstants,
    w::CDPWorkspace,
    d::DesignMeasure,
    cdp::CompositeDesignProblem,
    oc::CDPObjectiveConstants,
)
    map(
        cdc.component_constants,
        w.component_workspaces,
        problems(cdp),
        oc.component_constants,
    ) do dc, w, p, oc
        derivative_constants!(dc, w, d, p, oc)
    end
    return cdc
end

function objective!(
    wsp::CDPWorkspace,
    d::DesignMeasure,
    cdp::CompositeDesignProblem,
    oc::CDPObjectiveConstants,
)
    acc = 0.0
    numprobs = length(weights(cdp))
    ws = weights(cdp)
    ps = problems(cdp)
    cc = oc.component_constants
    wk = wsp.component_workspaces
    for i in 1:numprobs
        acc += ws[i] * objective!(wk[i], d, ps[i], cc[i])
    end
    return acc
end

function gateauxderivative!(
    wsp::CDPWorkspace,
    direction::DesignMeasure,
    cdp::CompositeDesignProblem,
    dc::CDPDerivativeConstants,
)
    acc = 0.0
    numprobs = length(weights(cdp))
    ws = weights(cdp)
    ps = problems(cdp)
    cc = dc.component_constants
    wk = wsp.component_workspaces
    for i in 1:numprobs
        acc += ws[i] * gateauxderivative!(wk[i], direction, ps[i], cc[i])
    end
    return acc
end

function sensitivity!(
    cwg::CDPWorkspace,
    direction::DesignMeasure,
    cdp::CompositeDesignProblem,
    cdc::CDPDerivativeConstants,
)
    acc = 0.0
    map(
        cwg.component_workspaces,
        problems(cdp),
        weights(cdp),
        cdc.component_constants,
    ) do wg, p, w, dc
        acc += w * sensitivity!(wg, direction, p, dc)
    end
    return acc
end

"""
    simplify(d::DesignMeasure, cdp::CompositeDesignProblem; maxweight = 0, maxdist = 0, uargs...)

A wrapper that calls
[`simplify_merge`](@ref),
and [`simplify_drop`](@ref) in that order.

Note that `uargs` are ignored
since there is no obvious generalization of [`simplify_unique`](@ref)
for general composite design problems.
When `uargs` is not empty, a warning is printed.
"""
function simplify(
    d::DesignMeasure,
    cdp::CompositeDesignProblem;
    maxweight = 0,
    maxdist = 0,
    uargs...,
)
    if !isempty(uargs)
        @warn "uargs $(collect(uargs)) ignored for a composite design problem"
    end
    d = simplify_merge(d, region(cdp), maxdist)
    d = simplify_drop(d, maxweight)
    return d
end
