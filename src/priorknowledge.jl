# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## concrete types for prior knowledge ##

"""
    PriorSample{T} <: PriorKnowledge{T}

A sample from a prior distribution, or a discrete prior distribution with finite support.
"""
struct PriorSample{T} <: PriorKnowledge{T}
    weight::Vector{Float64}
    p::Vector{T}
    @doc """
    PriorSample(p::AbstractVector{<:Parameter} [, weights::AbstractVector{<:Real}])

Construct a weighted prior sample on the given parameter draws.

If no `weights` are given, a uniform distribution on the elements of `p` is assumed.

See also [`parameters`](@ref), [`weights(::PriorSample)`](@ref).
"""
    function PriorSample(
        parameters::AbstractVector{T},
        weights::AbstractVector{<:Real} = fill(1 / length(parameters), length(parameters)),
    ) where T<:Parameter
        if length(weights) != length(parameters)
            throw(ArgumentError("number of weights and parameter values must be equal"))
        end
        if any(weights .< 0) || !(sum(weights) ≈ 1)
            throw(ArgumentError("weights must be non-negative and sum to one"))
        end
        return new{T}(weights, parameters)
    end
end

function parameter_dimension(pk::PriorSample)
    return dimension(parameters(pk)[1])
end

"""
    weights(pk::PriorSample)

Return a reference to the weights of the [`PriorSample`](@ref).

See also [`parameters`](@ref).
"""
function weights(pk::PriorSample)
    return pk.weight
end

"""
    parameters(pk::PriorSample)

Return a reference to the parameters of the [`PriorSample`](@ref).

See also [`weights(::DesignMeasure)`](@ref).
"""
function parameters(pk::PriorSample)
    return pk.p
end
