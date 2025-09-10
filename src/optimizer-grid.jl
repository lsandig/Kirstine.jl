# SPDX-FileCopyrightText: 2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## generic grid search ##

"""
    Kirstine.GridSpec

Supertype for grid specifications.

See also [`ProductGrid`](@ref), [`GridSearch`](@ref).
"""
abstract type GridSpec end

# Function stub for
#
# (gs::T, c::U, ps::AbstractVector{V}) -> AbstractVector{V}
#
# where T <: GridSpec, U <: AbstractConstraints, V <: AbstractPoint
#
# See designmeasure-abstractpoint.jl for ProductGrid and DesignMeasure implementation,
# and ../test/optimizer-grid.jl for Euclidean points.
function construct_grid end

"""
    Kirstine.ProductGrid <: Kirstine.GridSpec

A Cartesian product with the given number of grid points in each dimension.

# Examples

A 2-dimensional grid with 5×11 points:

```jldoctest
julia> Kirstine.ProductGrid(5, 11)
Kirstine.ProductGrid{2}((5, 11))
```
"""
struct ProductGrid{N} <: GridSpec
    n_points::NTuple{N,Int64}
    function ProductGrid{N}(n_points::NTuple{N,Integer}) where N
        msg_min_points = "need at least 2 points in each direction"
        any(<(2), n_points) && throw(ArgumentError(msg_min_points))
        return new{N}(n_points)
    end
end

# The inner constructor for a one-dimensional product grid needs a 1-Tuple,
# which is not very user friendly to type: `ProductGrid{1}((42,))`;
# hence we define a convenient outer splatting constructor for arbitrary dimensions.
ProductGrid(n_points::Integer...) = ProductGrid{length(n_points)}(n_points)

"""
    Kirstine.GridSearch <: ZerothOrderOptimizer

A grid search optimizer.

!!! warning "Experimental Feature"

    This optimizer is experimental.
    It is not to be considered part of the API yet,
    hence it is not exported and may be subject to breaking changes
    even in minor or patch version updates.

This optimizer is particularly useful for the direction-finding step in [`Exchange`](@ref),
but can also be used for [`DirectMaximization`](@ref)
as long as all weights of the candidate are fixed,
and there are only few non-fixed design points.
Note that for `DirectMaximization`,
the total number of grid points is exponential both in the dimension of the design region
and in the number of non-fixed design points.

(Strictly speaking this optimizer is not particle-based,
but it uses the same internal infrastructure.)

Since grid search finds the maximum over a finite set of points,
the `has_converged` flag of a corresponding [`OptimizationResult`](@ref)
is always `true` by definition.

# Initialization

Similar to optimization with [`Pso`](@ref),
the optimizer state is initialized internally with a vector of prototype points.
The first prototype determines the structure of each grid point
that is constructed from the [`Kirstine.GridSpec`](@ref).
This set of points is then merged with the vector of prototypes
and used for maximization.
"""
struct GridSearch{T<:GridSpec} <: ZerothOrderOptimizer
    gs::T
end

"""
    Kirstine.GridSearch(; gridspec::Kirstine.GridSpec)

Set up parameters for a grid search.

The [`GridSpec`](@ref) is used to construct a grid over the problem's [`DesignRegion`](@ref),
so the dimensions of both must be equal.

# Examples

```jldoctest
julia> Kirstine.GridSearch(; gridspec = Kirstine.ProductGrid(11, 11))
Kirstine.GridSearch{Kirstine.ProductGrid{2}}(Kirstine.ProductGrid{2}((11, 11)))
```
"""
GridSearch(; gridspec::GridSpec) = GridSearch(gridspec)

struct GridSearchState{T<:AbstractPoint,U} <: OptimizerState{T,U}
    x::Vector{T}
    fx::Vector{Float64}
    function GridSearchState(x::AbstractVector{<:AbstractPoint}, fx::AbstractVector{<:Real})
        size(x) != size(fx) && throw(ArgumentError("x and fx must have same length"))
        length(x) == 0 && throw(ArgumentError("vector of grid points can't be empty"))
        return new{typeof(x[1]),typeof(ap_as_difference(x[1]))}(x, fx)
    end
end

# Implementation of optimizer interface
iterations(s::GridSearch) = 1

maximizer(s::GridSearchState) = s.x[argmax(s.fx)]

maximum(s::GridSearchState) = maximum(s.fx)

n_eval_f(s::GridSearchState) = length(s.fx)

n_eval_g(s::GridSearchState) = 0

has_converged(s::GridSearchState, o::GridSearch) = true

tick!(f, s::GridSearchState, o::GridSearch, c::AbstractConstraints) = s

# Note: user-visible documentation is attached to `GridSearch`.
function optimizer_state(
    f::Function,
    o::GridSearch,
    prototypes::AbstractVector{<:AbstractPoint},
    c::AbstractConstraints,
)
    # construct grid subject to constraints
    x_grid = construct_grid(o.gs, c, prototypes)
    x = unique(vcat(x_grid, prototypes))
    # evaluate f on grid points and prototype points
    return GridSearchState(x, map(f, x))
end
