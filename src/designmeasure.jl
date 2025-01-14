# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## design measures ##

"""
    DesignMeasure

A probability measure with finite support representing a continuous experimental design.

The support points of a design measure are called _design points_.
In Julia, a design point is simply a `Vector{Float64}`.

Special kinds of design measures can be constructed with [`one_point_design`](@ref),
[`uniform_design`](@ref), [`equidistant_design`](@ref), [`random_design`](@ref).

See also [`weights(::DesignMeasure)`](@ref), [`points`](@ref), [`apportion`](@ref),
[plotting API](api.md#Plotting).
"""
struct DesignMeasure <: AbstractPoint
    points::Matrix{Float64}
    weights::Vector{Float64}
    @doc """
        DesignMeasure(
            points::AbstractMatrix{<:Real},
            weights::AbstractVector{<:Real}
        )

    Construct a design measure with design points from the columns of `points`.

    This is the only design measure constructor
    where the result *does* share memory with `points` and `weights`.
    """
    function DesignMeasure(points::AbstractMatrix{<:Real}, weights::AbstractVector{<:Real})
        if length(weights) != size(points, 2)
            throw(ArgumentError("number of weights and design points must be equal"))
        end
        if any(weights .< 0) || !(sum(weights) ≈ 1)
            throw(ArgumentError("weights must be non-negative and sum to one"))
        end
        new(points, weights)
    end
end

## constructors ##

@doc """
    DesignMeasure(
        points::AbstractVector{<:AbstractVector{<:Real}},
        weights::AbstractVector{<:Real},
    )

Construct a design measure from a vector of design points.

The result does not share memory with `points`.

# Examples
```jldoctest
julia> DesignMeasure([[1, 2], [3, 4], [5, 6]], [0.5, 0.2, 0.3])
DesignMeasure(
 [1.0, 2.0] => 0.5,
 [3.0, 4.0] => 0.2,
 [5.0, 6.0] => 0.3,
)
```
"""
function DesignMeasure(
    points::AbstractVector{<:AbstractVector{<:Real}},
    weights::AbstractVector{<:Real},
)
    if !allequal(map(length, points))
        throw(ArgumentError("design points must have identical lengths"))
    end
    return DesignMeasure(reduce(hcat, points), weights)
end

"""
    DesignMeasure(dp_w::Pair...)

Construct a design measure from `designpoint => weight` pairs.

# Examples

```jldoctest
julia> DesignMeasure([1] => 0.2, [42] => 0.3, [9] => 0.5)
DesignMeasure(
 [1.0] => 0.2,
 [42.0] => 0.3,
 [9.0] => 0.5,
)
```
"""
function DesignMeasure(dp_w::Pair...)
    ws = [w for (_, w) in dp_w]
    dps = stack(first, dp_w) # this indirectly promotes dp eltypes
    return DesignMeasure(dps, ws)
end

"""
    one_point_design(designpoint::AbstractVector{<:Real})

Construct a one-point [`DesignMeasure`](@ref).

The result does not share memory with `designpoint`.

# Examples

```jldoctest
julia> one_point_design([42])
DesignMeasure(
 [42.0] => 1.0,
)
```
"""
function one_point_design(designpoint::AbstractVector{<:Real})
    return DesignMeasure([designpoint], [1.0])
end

"""
    uniform_design(designpoints::AbstractVector{<:AbstractVector{<:Real}})

Construct a [`DesignMeasure`](@ref) with equal weights on the given `designpoints`.

The result does not share memory with `designpoints`.

# Examples

```jldoctest
julia> uniform_design([[1], [2], [3], [4]])
DesignMeasure(
 [1.0] => 0.25,
 [2.0] => 0.25,
 [3.0] => 0.25,
 [4.0] => 0.25,
)
```
"""
function uniform_design(designpoints::AbstractVector{<:AbstractVector{<:Real}})
    K = length(designpoints)
    w = fill(1 / K, K)
    return DesignMeasure(designpoints, w)
end

"""
    equidistant_design(dr::DesignInterval{1}, K::Integer)

Construct a [`DesignMeasure`](@ref) with an equally-spaced grid of `K` design points
and uniform weights on the given 1-dimensional design interval.

# Examples

```jldoctest
julia> equidistant_design(DesignInterval(:z => (0, 1)), 5)
DesignMeasure(
 [0.0] => 0.2,
 [0.25] => 0.2,
 [0.5] => 0.2,
 [0.75] => 0.2,
 [1.0] => 0.2,
)
```
"""
function equidistant_design(dr::DesignInterval{1}, K::Integer)
    if K <= 1
        throw(ArgumentError("equidistant design needs at least K = 2, got $K"))
    end
    val = range(lowerbound(dr)[1], upperbound(dr)[1]; length = K)
    designpoints = [[dp] for dp in val]
    return uniform_design(designpoints)
end

# Note: docstring for abstract type, implementation for subtypes
"""
    random_design(dr::Kirstine.DesignRegion, K::Integer)

Construct a random [`DesignMeasure`](@ref).

The design points are drawn independently from the uniform distribution on the design region,
and the weights drawn from the uniform distribution on a simplex.
"""
function random_design(dr::DesignInterval{N}, K::Integer) where N
    scl = upperbound(dr) .- lowerbound(dr)
    dp = [lowerbound(dr) .+ scl .* rand(N) for _ in 1:K]
    u = rand(K) # independent Uniform([0, 1])
    v = -log.(u) # independent Exp(1)
    w = v ./ sum(v) # Dirichlet(1, …, 1), i.e. uniform on simplex
    d = DesignMeasure(dp, w)
    return d
end

## accessors ##

"""
    points(d::DesignMeasure)

Return an iterator over the design points of `d`.

See also [`weights(::DesignMeasure)`](@ref), [`numpoints`](@ref).
"""
points(d::DesignMeasure) = eachcol(d.points)

"""
    weights(d::DesignMeasure)

Return a reference to the weights of the design measure.

See also [`points`](@ref), [`numpoints`](@ref).
"""
weights(d::DesignMeasure) = d.weights

"""
    numpoints(d::DesignMeasure)

Return the number of design points of `d`.

See also [`points`](@ref), [`weights(::DesignMeasure)`](@ref).
"""
numpoints(d::DesignMeasure) = size(d.points, 2)

## utility operations ##

"""
    ==(d1::DesignMeasure, d2::DesignMeasure)

Test design measures for equality.

Two design measures are considered equal iff

  - they have the same number of design points,
  - and their design points and weights are equal,
  - and their design points are in the same order.

Note that this is stricter than when comparing measures as mathematical functions,
where the order of the design points in the representation does not matter.
"""
function Base.:(==)(d1::DesignMeasure, d2::DesignMeasure)
    return weights(d1) == weights(d2) && points(d1) == points(d2)
end

function Base.hash(d::DesignMeasure, h::UInt)
    return hash(d.points, hash(d.weights, h))
end

function Base.show(io::IO, ::MIME"text/plain", d::DesignMeasure)
    pairs = map(weights(d), points(d)) do w, dp
        return string(dp) * " => " * string(w)
    end
    print(io, typeof(d), "(\n")
    print(io, " ", join(pairs, ",\n "))
    print(io, ",\n)")
end

"""
    sort_points(d::DesignMeasure; rev = false, dimorder = 1:length(first(points(d))))

Return a representation of `d` where the design points are sorted lexicographically.

By default,
the result is sorted by the first dimension of the design points,
and ties are resolved by the second,
further ties by the third,
and so on.
This can be changed by setting `dimorder` to a different permutation.

See also [`sort_weights`](@ref).

# Examples

```jldoctest
julia> sort_points(uniform_design([[3, 40], [2, 10], [1, 20], [2, 30]]))
DesignMeasure(
 [1.0, 20.0] => 0.25,
 [2.0, 10.0] => 0.25,
 [2.0, 30.0] => 0.25,
 [3.0, 40.0] => 0.25,
)

julia> sort_points(uniform_design([[3, 40], [2, 10], [1, 20], [2, 30]]); dimorder = [2, 1])
DesignMeasure(
 [2.0, 10.0] => 0.25,
 [1.0, 20.0] => 0.25,
 [2.0, 30.0] => 0.25,
 [3.0, 40.0] => 0.25,
)
```
"""
function sort_points(
    d::DesignMeasure;
    rev::Bool = false,
    dimorder::AbstractVector{<:Integer} = 1:length(first(points(d))),
)
    ordered_dims = 1:length(first(points(d)))
    if sort(dimorder) != ordered_dims
        throw(ArgumentError("dimorder must be a permutation of $(ordered_dims)"))
    end
    # note: no (deep)copies needed because indexing with `p` generates a copy
    dp = points(d)
    w = weights(d)
    for i in reverse(dimorder)
        p = sortperm(map(x -> x[i], dp); rev = rev)
        dp = dp[p]
        w = w[p]
    end
    return DesignMeasure(dp, w)
end

"""
    sort_weights(d::DesignMeasure; rev::Bool = false)

Return a representation of `d` where the design points are sorted by their
corresponding weights.

See also [`sort_points`](@ref).

# Examples

```jldoctest
julia> sort_weights(DesignMeasure([[1], [2], [3]], [0.5, 0.2, 0.3]))
DesignMeasure(
 [2.0] => 0.2,
 [3.0] => 0.3,
 [1.0] => 0.5,
)
```
"""
function sort_weights(d::DesignMeasure; rev::Bool = false)
    p = sortperm(weights(d); rev = rev)
    w = weights(d)[p]
    dp = points(d)[p]
    return DesignMeasure(dp, w)
end

"""
    mixture(alpha, d1::DesignMeasure, d2::DesignMeasure)

Return the convex combination ``α d_1 + (1-α) d_2``.

The result is not simplified, hence its design points might not be unique.
"""
function mixture(alpha::Real, d1::DesignMeasure, d2::DesignMeasure)
    if length(points(d1)[1]) != length(points(d2)[1])
        throw(ArgumentError("design points must have identical lengths"))
    end
    if alpha < 0 || alpha > 1
        throw(ArgumentError("mixture weight must be between 0 and 1"))
    end
    w = vcat(alpha .* weights(d1), (1 - alpha) .* weights(d2))
    dp = vcat(points(d1), points(d2))
    return DesignMeasure(dp, w)
end

## apportionment ##

"""
    apportion(weights::AbstractVector{<:Real}, n::Integer)

Find integers `a` with `sum(a) == n` such that `a ./ n` best approximates `weights`.

This is the efficient design apportionment procedure from Pukelsheim[^P06], p. 309.

[^P06]: Friedrich Pukelsheim (2006). Optimal design of experiments. Wiley. [doi:10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109)
"""
function apportion(weights::AbstractVector{<:Real}, n::Integer)
    # Phase 1: initial multipliers
    a = ceil.(Int64, (n - 0.5 * length(weights)) .* weights)
    # Phase 2: loop until discrepancy is zero
    discrepancy = sum(a) - n
    while discrepancy != 0
        if discrepancy < 0
            j = argmin(a ./ weights)
            a[j] += 1
        else # discrepancy > 0
            k = argmax((a .- 1) ./ weights)
            a[k] -= 1
        end
        discrepancy = sum(a) - n
    end
    return a
end

## simplification ##

"""
    apportion(d::DesignMeasure, n)

Apportion the weights of `d`.
"""
function apportion(d::DesignMeasure, n::Integer)
    return apportion(weights(d), n)
end

"""
    simplify_drop(d::DesignMeasure, maxweight::Real)

Construct a new `DesignMeasure` where all design points with weights smaller than or equal to
`maxweight` are removed.

The vector of remaining weights is re-normalized.
"""
function simplify_drop(d::DesignMeasure, maxweight::Real)
    maxweight < 1 || throw(ArgumentError("maxweight must be < 1"))
    if numpoints(d) == 1 # nothing to do for one-point-designs
        return deepcopy(d) # return a copy for consistency
    end
    enough_weight = weights(d) .> maxweight
    if all(enough_weight)
        return deepcopy(d)
    end
    dps = points(d)[enough_weight]
    ws = weights(d)[enough_weight]
    ws ./= sum(ws)
    return DesignMeasure(dps, ws)
end

"""
    simplify_unique(d::DesignMeasure, dr::Kirstine.DesignRegion, m::M, cp::C; uargs...)

Construct a new [`DesignMeasure`](@ref) that corresponds uniquely to its implied normalized
information matrix.

The package default is a catch-all with the abstract types `M = Model` and
`C = CovariateParameterization`, which simply returns a copy of `d`.

When called via [`simplify`](@ref), user-model specific keyword arguments will be passed in
`uargs`.

# Implementation

Users can specialize this method for their concrete subtypes `M <: Kirstine.Model` and
`C <: CovariateParameterization`. It is intended for cases where the mapping from design measure
to normalized information matrix is not one-to-one. This depends on the model and covariate
parameterization used. In such a case, `simplify_unique` should be implemented to select a
canonical version of the design.

!!! note

    User-defined versions must have type annotations on all arguments to resolve method
    ambiguity.

# Examples

For several worked examples, see the [dose-time-response vignette](dtr.md).
"""
function simplify_unique(
    d::DesignMeasure,
    dr::DesignRegion,
    m::Model,
    cp::CovariateParameterization;
    uargs...,
)
    if !isempty(uargs)
        @warn "unused keyword arguments given to generic `simplify_unique` method" uargs
    end
    # fallback no-op when no model-specific simplification is defined
    return deepcopy(d)
end

"""
    simplify_merge(d::DesignMeasure, dr::Kirstine.DesignRegion, maxdist::Real)

Merge designpoints with a normalized distance smaller or equal to `maxdist`.

Special case of `simplify_merge` with `f(z) = (z .- lb) ./ (ub .- lb)`,
where `ub` and `lb` define the bounding box of `dr`,
and `distfun(x, y) = norm(x - y)`.

In other words,
the design points of `d` are scaled into the `N`-dimensional unit cube
before measuring their pairwise distances.
Note that the largest possible distance between points is `sqrt(N)`.

# Examples

After scaling into the unit square,
the distance between the first two points less than `0.3`,
hence they are merged.

```jldoctest
julia> d = DesignMeasure([[2.25, 2], [3.5, 2.5], [10, 3]], [0.3, 0.1, 0.6]);

julia> simplify_merge(d, DesignInterval(:a => (1, 11), :b => (2, 4)), 0.3)
DesignMeasure(
 [2.5624999999999996, 2.125] => 0.4,
 [10.0, 3.0] => 0.6,
)
```
"""
function simplify_merge(d::DesignMeasure, dr::DesignRegion, maxdist::Real)
    l2dist(p1, p2) = norm(p1 .- p2)
    return simplify_merge(d, p -> coordinates_in_bounding_box(p, dr), l2dist, maxdist)
end

function coordinates_in_bounding_box(dp::AbstractVector{<:Real}, dr::DesignRegion)
    lb, ub = boundingbox(dr)
    width = collect(ub .- lb)
    return (dp .- lb) ./ width
end

function pairwise_distance_matrix(
    dps::AbstractVector{<:AbstractVector{<:Real}},
    f::Function,
    distfun::Function,
)
    return map(p -> distfun(f(p[1]), f(p[2])), Iterators.product(dps, dps))
end

# Note: design measure before functions for consistency with the design region variant
"""
    simplify_merge(d::DesignMeasure, f::Function, distfun::Function, maxdist::Real)

Merge design points with `distfun(f(dp1), f(dp2))` smaller or equal to `maxdist`.

The following two steps are repeated until all points are more than `maxdist` apart:

 1. All pairwise distances are calculated.
 2. The two points closest to each other are averaged according to their relative weights.

# Examples

The first example uses the 2-norm for measuring distances between the points directly,
and merges the first two points since they are `1/sqrt(2)` apart.
The second example uses the 1-norm instead,
where the first two points have distance `1`,
so nothing happens.

```jldoctest
julia> d = DesignMeasure([[1, 2], [1.5, 2.5], [3, 3]], [0.1, 0.4, 0.5]);

julia> simplify_merge(d, identity, (x, y) -> sqrt(sum((x .- y) .^ 2)), 0.8)
DesignMeasure(
 [1.4000000000000001, 2.4] => 0.5,
 [3.0, 3.0] => 0.5,
)

julia> simplify_merge(d, identity, (x, y) -> sum(abs.(x - y)), 0.8)
DesignMeasure(
 [1.0, 2.0] => 0.1,
 [1.5, 2.5] => 0.4,
 [3.0, 3.0] => 0.5,
)
```

The third example has two pairs of points with the same distance to each other.
When simplified with a design region, either none or both are merged.
However, when mapped onto the graph of the `exp()` function,
only the second pair will be merged
since it is spread much wider than the first.
Such a behavior might be desired
with a more complicated expected response function in place of `exp()`.

```jldoctest
julia> d = uniform_design([[0.2], [0.3], [4.9], [5.0]]);

julia> simplify_merge(d, DesignInterval(:a => (0, 10)), 0.01)
DesignMeasure(
 [0.25] => 0.5,
 [4.95] => 0.5,
)

julia> simplify_merge(d, z -> [z[1], exp(z[1])], (x, y) -> sqrt(sum((x .- y) .^ 2)), 1)
DesignMeasure(
 [0.25] => 0.5,
 [4.9] => 0.25,
 [5.0] => 0.25,
)
```
"""
function simplify_merge(d::DesignMeasure, f::Function, distfun::Function, maxdist::Real)
    # nothing to do for one-point-designs, return a copy for consistency
    numpoints(d) == 1 && return deepcopy(d)
    dps = collect.(points(d))
    ws = deepcopy(weights(d))
    cur_min_dist = 0
    while cur_min_dist <= maxdist
        # compute pairwise distances of transformed designpoints,
        # merge the two designpoints nearest to each other
        distmat = pairwise_distance_matrix(dps, f, distfun)
        distmat[diagind(distmat)] .= Inf
        cur_min_dist, idx = findmin(distmat) # i > j because rows vary fastest
        i = idx[1]
        j = idx[2]
        if cur_min_dist <= maxdist
            w_new = ws[i] + ws[j]
            dp_raw = (ws[i] .* dps[i] .+ ws[j] .* dps[j]) ./ w_new
            dp_new = min.(max.(dps[i], dps[j]), max.(min.(dps[i], dps[j]), dp_raw))
            deleteat!(dps, (j, i))
            pushfirst!(dps, dp_new)
            deleteat!(ws, (j, i))
            pushfirst!(ws, w_new)
        end
    end
    return DesignMeasure(dps, ws)
end
