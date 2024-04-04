# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## composite design problem made up of weighted abstract design problems  ##

"""
    CompositeDesignProblem <: AbstractDesignProblem
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
