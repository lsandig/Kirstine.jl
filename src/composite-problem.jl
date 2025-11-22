# SPDX-FileCopyrightText: 2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
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
    [`region(::CompositeDesignProblem)`](@ref).
    """
    function CompositeDesignProblem(
        problems::AbstractVector{<:AbstractDesignProblem},
        weights::AbstractVector{<:Real},
    )
        err_length = ArgumentError("number of problems must match number of weights")
        length(problems) == length(weights) || throw(err_length)
        err_empty = ArgumentError("component problems must not be empty")
        !isempty(problems) || throw(err_empty)
        err_neg = ArgumentError("problem weights must be non-negative")
        all(>=(0), weights) || throw(err_neg)
        err_region = ArgumentError("all problems must have identical regions")
        allequal(region, problems) || throw(err_region)
        return new{eltype(problems)}(problems, weights)
    end
end

function Base.:(==)(cdp1::CompositeDesignProblem, cdp2::CompositeDesignProblem)
    return problems(cdp1) == problems(cdp2) && weights(cdp1) == weights(cdp2)
end

function Base.hash(cdp::CompositeDesignProblem, h::UInt)
    return hash(problems(cdp), hash(weights(cdp), h))
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
region(cdp::CompositeDesignProblem) = region(first(problems(cdp)))

function pretty_lines_cdp(cdp::CompositeDesignProblem, depth::Integer)
    lines = ["CompositeDesignProblem"]
    indent = "  "
    depth == 0 && push!(lines, indent * "with region: " * string(region(cdp)))
    iob = IOBuffer()
    ioc = IOContext(iob, :limit => false)
    pretty = String[]
    map(problems(cdp), weights(cdp)) do dp, w
        if dp isa CompositeDesignProblem
            pretty = pretty_lines_cdp(dp, depth + 1)
        else
            show(ioc, "text/plain", dp)
            pretty = split(String(take!(iob)), '\n')
        end
        push!(lines, indent * "* " * popfirst!(pretty) * " (weight: $w)")
        append!(lines, indent .* pretty)
    end
    return lines
end

function Base.show(io::IO, ::MIME"text/plain", cdp::CompositeDesignProblem)
    print(io, join(pretty_lines_cdp(cdp, 0), '\n'))
end

struct CDPWorkspace{T<:ProblemWorkspace} <: ProblemWorkspace
    component_workspaces::Vector{T}
end

function allocate_workspace(d::DesignMeasure, cdp::CompositeDesignProblem)
    return CDPWorkspace(map(p -> allocate_workspace(d, p), problems(cdp)))
end

Base.getindex(wk::CDPWorkspace, i::Int) = wk.component_workspaces[i]

struct CDPObjectiveConstants{T<:ObjectiveConstants} <: ObjectiveConstants
    component_constants::Vector{T}
end

function objective_constants(cdp::CompositeDesignProblem)
    return CDPObjectiveConstants(map(objective_constants, problems(cdp)))
end

Base.getindex(oc::CDPObjectiveConstants, i::Int) = oc.component_constants[i]

struct CDPDerivativeConstants{T<:DerivativeConstants} <: DerivativeConstants
    component_constants::Vector{T}
end

function derivative_constants(at::DesignMeasure, cdp::CompositeDesignProblem)
    return CDPDerivativeConstants(map(p -> derivative_constants(at, p), problems(cdp)))
end

Base.getindex(dc::CDPDerivativeConstants, i::Int) = dc.component_constants[i]

function derivative_constants!(
    dc::CDPDerivativeConstants,
    wk::CDPWorkspace,
    d::DesignMeasure,
    cdp::CompositeDesignProblem,
    oc::CDPObjectiveConstants,
)
    for i in 1:length(problems(cdp))
        derivative_constants!(dc[i], wk[i], d, problems(cdp)[i], oc[i])
    end
    return dc
end

function objective!(
    wk::CDPWorkspace,
    d::DesignMeasure,
    cdp::CompositeDesignProblem,
    oc::CDPObjectiveConstants,
)
    acc = 0.0
    for i in 1:length(problems(cdp))
        acc += weights(cdp)[i] * objective!(wk[i], d, problems(cdp)[i], oc[i])
    end
    return acc
    return acc
end

function gateauxderivative!(
    wk::CDPWorkspace,
    toward::DesignMeasure,
    cdp::CompositeDesignProblem,
    dc::CDPDerivativeConstants,
)
    acc = 0.0
    for i in 1:length(problems(cdp))
        acc += weights(cdp)[i] * gateauxderivative!(wk[i], toward, problems(cdp)[i], dc[i])
    end
    return acc
end

function sensitivity!(
    wk::CDPWorkspace,
    toward::DesignMeasure,
    cdp::CompositeDesignProblem,
    dc::CDPDerivativeConstants,
)
    acc = 0.0
    for i in 1:length(problems(cdp))
        acc += weights(cdp)[i] * sensitivity!(wk[i], toward, problems(cdp)[i], dc[i])
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

function check_numerically_singular(d::DesignMeasure, cdp::CompositeDesignProblem)
    map(dp -> check_numerically_singular(d, dp), problems(cdp))
    return nothing
end
