# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## concrete types for prior knowledge ##

"""
    PriorSample{T} <: Kirstine.PriorKnowledge{T}

A sample from a prior distribution, or a discrete prior distribution with finite support.
"""
struct PriorSample{T} <: PriorKnowledge{T}
    p::Vector{T}
    weight::Vector{Float64}
    function PriorSample{T}(
        parameters::AbstractVector{T},
        weights::AbstractVector{<:Real},
    ) where T<:Parameter
        err_length = ArgumentError("number of weights and parameter values must be equal")
        length(weights) == length(parameters) || throw(err_length)
        err_weights = ArgumentError("weights must be nonnegative and sum to one")
        (all(weights .>= 0) && sum(weights) ≈ 1) || throw(err_weights)
        err_type = ArgumentError("parameters must all have the same concrete type")
        isconcretetype(T) || throw(err_type)
        err_dim = ArgumentError("parameters must all have the same dimension")
        allequal(dimension, parameters) || throw(err_dim)
        return new{T}(parameters, weights)
    end
end

"""
    PriorSample(p::AbstractVector{<:Kirstine.Parameter} [, weights::AbstractVector{<:Real}])

Construct a weighted prior sample on the given parameter draws.

If no `weights` are given, a uniform distribution on the elements of `p` is assumed.

See also [`parameters`](@ref), [`weights(::PriorSample)`](@ref).
"""
function PriorSample(
    parameters::AbstractVector{T},
    weights::AbstractVector{<:Real} = fill(1 / length(parameters), length(parameters)),
) where T<:Parameter
    return PriorSample{T}(parameters, weights)
end

parameter_dimension(pk::PriorSample) = dimension(first(parameters(pk)))

"""
    weights(pk::PriorSample)

Return a reference to the weights of the [`PriorSample`](@ref).

See also [`parameters`](@ref).
"""
weights(pk::PriorSample) = pk.weight

"""
    parameters(pk::PriorSample)

Return a reference to the parameters of the [`PriorSample`](@ref).

See also [`weights(::DesignMeasure)`](@ref).
"""
parameters(pk::PriorSample) = pk.p

"""
    ==(pk1::PriorSample, pk2::PriorSample)

Test prior samples for equality.

Two prior samples are considered equal iff

  - they have the same subtype of [`Parameter`](@ref),
  - and their parameter draws are equal,
  - and their weights are equal,
  - and the order of the draws is equal.

Note that this is stricter than when comparing prior samples as mathematical sets,
where the order of the draws does not matter.
"""
function Base.:(==)(pk1::PriorSample, pk2::PriorSample)
    return weights(pk1) == weights(pk2) && parameters(pk1) == parameters(pk2)
end

function Base.hash(pk::PriorSample, h::UInt)
    return hash(parameters(pk), hash(weights(pk), h))
end

function Base.show(io::IO, ::MIME"text/plain", pk::PriorSample)
    partype = typeof(first(parameters(pk)))
    ndraw = length(parameters(pk))
    if ndraw > 1
        print(io, "$(ndraw) draws of $(partype)")
    else
        print(io, "one-point distribution at $(only(parameters(pk)))")
    end
end

struct GCPriorSample <: GateauxConstants
    A::Vector{Matrix{Float64}}
    tr_B::Vector{Float64}
end
