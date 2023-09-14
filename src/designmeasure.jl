# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## design measures ##

"""
    DesignMeasure

A probability measure with finite support representing a continuous experimental design.

The support points of a design measure are called _design points_.
In Julia, a design point is simply a `Vector{Float64}`.

Special kinds of design measures can be constructed with [`one_point_design`](@ref),
[`uniform_design`](@ref), [`equidistant_design`](@ref), [`random_design`](@ref).

See also [`weights`](@ref), [`points`](@ref), [`as_matrix`](@ref),
[`apportion`](@ref).
"""
struct DesignMeasure <: AbstractPoint
    points::Matrix{Float64}
    weight::Vector{Float64}
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
            error("number of weights and design points must be equal")
        end
        if any(weights .< 0) || !(sum(weights) ≈ 1)
            error("weights must be non-negative and sum to one")
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
        error("design points must have identical lengths")
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
    dps = [dp for (dp, _) in dp_w]
    return DesignMeasure(dps, ws)
end

"""
    DesignMeasure(m::AbstractMatrix{<:Real})

Construct a [`DesignMeasure`](@ref) from its matrix representation.

An `(N+1, K)` matrix `m` represents a `DesignMeasure` with `K` design points from an
`N`-dimensional design region. The first row of `m` must contain the weights.

The result does not share memory with `m`.

See also [`as_matrix`](@ref).

# Examples

```jldoctest
julia> m = [0.5 0.2 0.3; 7.0 8.0 9.0; 4.0 5.0 6.0]
3×3 Matrix{Float64}:
 0.5  0.2  0.3
 7.0  8.0  9.0
 4.0  5.0  6.0

julia> DesignMeasure(m)
DesignMeasure(
 [7.0, 4.0] => 0.5,
 [8.0, 5.0] => 0.2,
 [9.0, 6.0] => 0.3,
)
```
"""
function DesignMeasure(m::AbstractMatrix{<:Real})
    if size(m, 1) < 2
        throw(ArgumentError("m must have at least two rows"))
    end
    ws = m[1, :]
    dps = [m[2:end, k] for k in 1:size(m, 2)]
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
    val = range(lowerbound(dr)[1], upperbound(dr)[1]; length = K)
    designpoints = [[dp] for dp in val]
    return uniform_design(designpoints)
end

# Note: docstring for abstract type, implementation for subtypes
"""
    random_design(dr::DesignRegion, K::Integer)

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

See also [`weights`](@ref).
"""
points(d::DesignMeasure) = eachcol(d.points)

"""
    weights(d::DesignMeasure)

Return a copy of the weights.

See also [`points`](@ref).
"""
weights(d::DesignMeasure) = copy(d.weight)

"""
    as_matrix(d::DesignMeasure)

Return a matrix representation of `d`.

A [`DesignMeasure`](@ref) with `K` design points from an `N`-dimensional
design region corresponds to a `(N+1, K)` matrix.
The first row contains the weights.

The result does not share memory with `d`.

See also [`DesignMeasure`](@ref).

# Examples

```jldoctest
julia> as_matrix(DesignMeasure([[7, 4], [8, 5], [9, 6]], [0.5, 0.2, 0.3]))
3×3 Matrix{Float64}:
 0.5  0.2  0.3
 7.0  8.0  9.0
 4.0  5.0  6.0
```
"""
function as_matrix(d::DesignMeasure)
    return vcat(transpose(weights(d)), d.points)
end

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

function Base.show(io::IO, ::MIME"text/plain", d::DesignMeasure)
    pairs = map(weights(d), points(d)) do w, dp
        return string(dp) * " => " * string(w)
    end
    print(io, typeof(d), "(\n")
    print(io, " ", join(pairs, ",\n "))
    print(io, ",\n)")
end

"""
    sort_designpoints(d::DesignMeasure; rev::Bool = false)

Return a representation of `d` where the design points are sorted lexicographically.

See also [`sort_weights`](@ref).

# Examples

```jldoctest
julia> sort_designpoints(uniform_design([[3, 4], [2, 1], [1, 1], [2, 3]]))
DesignMeasure(
 [1.0, 1.0] => 0.25,
 [2.0, 1.0] => 0.25,
 [2.0, 3.0] => 0.25,
 [3.0, 4.0] => 0.25,
)
```
"""
function sort_designpoints(d::DesignMeasure; rev::Bool = false)
    # note: no (deep)copies needed because indexing with `p` generates a copy
    dp = points(d)
    w = d.weight
    for i in length(dp[1]):-1:1
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

See also [`sort_designpoints`](@ref).

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
    p = sortperm(d.weight; rev = rev)
    w = d.weight[p]
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
        error("design points must have identical lengths")
    end
    if alpha < 0 || alpha > 1
        error("mixture weight must be between 0 and 1")
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
    simplify_drop(d::DesignMeasure, minweight::Real)

Construct a new `DesignMeasure` where only design points with weights strictly larger than
`minweight` are kept.

The vector of remaining weights is re-normalized.
"""
function simplify_drop(d::DesignMeasure, minweight::Real)
    if length(d.weight) == 1 # nothing to do for one-point-designs
        return deepcopy(d) # return a copy for consistency
    end
    enough_weight = d.weight .> minweight
    dps = points(d)[enough_weight]
    ws = d.weight[enough_weight]
    ws ./= sum(ws)
    return DesignMeasure(dps, ws)
end

"""
    simplify_unique(d::DesignMeasure, dr::DesignRegion, m::M, cp::C; uargs...)

Construct a new [`DesignMeasure`](@ref) that corresponds uniquely to its implied normalized
information matrix.

The package default is a catch-all with the abstract types `M = Model` and
`C = CovariateParameterization`, which simply returns a copy of `d`.

When called via [`simplify`](@ref), user-model specific keyword arguments will be passed in
`uargs`.

# Implementation

Users can specialize this method for their concrete subtypes `M <: Model` and
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
    simplify_merge(d::DesignMeasure, dr::DesignInterval, mindist::Real)

Merge designpoints with a normalized distance smaller or equal to `mindist`.

The design points are first transformed into the unit (hyper)cube.
The argument `mindist` is intepreted relative to this unit cube,
i.e. only `0 < mindist < sqrt(N)` make sense for a design interval of dimension `N`.

The following two steps are repeated until all points are more than `mindist` apart:

 1. All pairwise euclidean distances are calculated.
 2. The two points closest to each other are averaged with their relative weights

Finally the design points are scaled back into the original design interval.
"""
function simplify_merge(d::DesignMeasure, dr::DesignInterval, mindist::Real)
    if length(d.weight) == 1 # nothing to do for one-point-designs
        return deepcopy(d) # return a copy for consistency
    end
    # scale design interval into unit cube
    width = collect(upperbound(dr) .- lowerbound(dr))
    dps = [(dp .- lowerbound(dr)) ./ width for dp in points(d)]
    ws = weights(d)
    cur_min_dist = 0
    while cur_min_dist <= mindist
        # compute pairwise L2-distances, merge the two designpoints nearest to each other
        dist = map(p -> norm(p[1] .- p[2]), Iterators.product(dps, dps))
        dist[diagind(dist)] .= Inf
        cur_min_dist, idx = findmin(dist) # i > j because rows vary fastest
        i = idx[1]
        j = idx[2]
        if cur_min_dist <= mindist
            w_new = ws[i] + ws[j]
            dp_new = (ws[i] .* dps[i] .+ ws[j] .* dps[j]) ./ w_new
            to_keep = (1:length(dps) .!= i) .&& (1:length(dps) .!= j)
            dps = [[dp_new]; dps[to_keep]]
            ws = [w_new; ws[to_keep]]
        end
    end
    # scale back
    dps = [(dp .* width) .+ lowerbound(dr) for dp in dps]
    return DesignMeasure(dps, ws)
end
