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
"""
    function PriorSample(
        parameters::AbstractVector{T},
        weights::AbstractVector{<:Real} = fill(1 / length(parameters), length(parameters)),
    ) where T<:Parameter
        if length(weights) != length(parameters)
            error("number of weights and parameter values must be equal")
        end
        if any(weights .< 0) || !(sum(weights) ≈ 1)
            error("weights must be non-negative and sum to one")
        end
        return new{T}(weights, parameters)
    end
end

function parameter_dimension(pk::PriorSample)
    return dimension(pk.p[1])
end
