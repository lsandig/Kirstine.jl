# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## generic grid search ##

abstract type GridSpec end

function construct_grid end

struct BBoxGrid{N} <: GridSpec
    n_points::NTuple{N,Int64}
    function BBoxGrid(n_points::Integer...)
        if any(n_points .< 2)
            throw(ArgumentError("need at least 2 points in each direction"))
        end
        n = length(n_points)
        new{n}(n_points)
    end
end

struct GridSearch{T<:GridSpec} <: Optimizer
    gs::T
end

struct GridSearchState{T<:AbstractPoint,U} <: OptimizerState{T,U}
    x::Vector{T}
    fx::Vector{Float64}
    function GridSearchState(x::AbstractVector{<:AbstractPoint}, fx::AbstractVector{<:Real})
        if size(x) != size(fx)
            throw(ArgumentError("length(x) and length(fx) must be identical"))
        elseif length(x) == 0
            throw(ArgumentError("vector of grid points can't be empty"))
        end
        T = typeof(x[1])
        U = typeof(ap_as_difference(x[1]))
        return new{T,U}(x, fx)
    end
end

iterations(s::GridSearch) = 1

maximizer(s::GridSearchState) = s.x[argmax(s.fx)]

maximum(s::GridSearchState) = maximum(s.fx)

n_eval(s::GridSearchState) = length(s.fx)

tick!(f, s::GridSearchState, o::GridSearch, c::AbstractConstraints, i::Integer) = s

function optimizer_state(
    f,
    o::GridSearch,
    prototypes::AbstractVector{<:AbstractPoint},
    c::AbstractConstraints,
)
    if length(prototypes) > 1
        @warn "grid search only uses first prototype"
    elseif length(prototypes) == 0
        throw(ArgumentError("grid search needs a prototype"))
    end
    # construct grid subject to constraints
    x = construct_grid(o.gs, c, prototypes[1])
    # evaluate f on grid points
    fx = map(f, x)
    return GridSearchState(x, fx)
end
