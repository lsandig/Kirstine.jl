# Constructors and utility functions operating on design measures and design spaces

# === accessor functions === #

"""
    support(d::DesignMeasure)

Return a vector containing all design points with positive weight.

See also: [`designpoints`](@ref), [`weights`](@ref), [`simplify_drop`](@ref).
"""
support(d::DesignMeasure) = d.designpoint[d.weight .> 0.0]

"""
    designpoints(d::DesignMeasure)

Return a copy of all designpoints, including those with zero weight.

See also: [`support`](@ref), [`weights`](@ref), [`simplify_drop`](@ref).
"""
designpoints(d::DesignMeasure) = deepcopy(d.designpoint)

"""
    weights(d::DesignMeasure)

Return a copy of the design point weights.

See also: [`support`](@ref), [`designpoints`](@ref), [`simplify_drop`](@ref).
"""
weights(d::DesignMeasure) = copy(d.weight)

"""
    lowerbound(designspace)

Return the vector of lower bounds.
"""
lowerbound(ds::DesignSpace) = ds.lowerbound

"""
    upperbound(designspace)

Return the vector of upper bounds.
"""
upperbound(ds::DesignSpace) = ds.upperbound

"""
    dimension(designspace)

Return the dimension of the designspace.
"""
dimension(ds::DesignSpace{N}) where N = N

"""
    dimnames(designspace)

Return the names of the designspace's dimensions.
"""
dimnames(ds::DesignSpace) = ds.name

# === additional constructors === #

"""
    DesignMeasure(dp_w::Pair...)

Construct a design measure out of `designpoint => weight` pairs.

# Examples

```jldoctest
julia> DesignMeasure([1] => 0.2, [42] => 0.3, [9] => 0.5)
DesignMeasure(
 [1.0] => 0.2,
 [42.0] => 0.3,
 [9.0] => 0.5
)
```
"""
function DesignMeasure(dp_w::Pair...)
    ws = [w for (_, w) in dp_w]
    dps = [dp for (dp, _) in dp_w]
    return DesignMeasure(ws, dps)
end

"""
    DesignMeasure(m::AbstractMatrix{<:Real})

Construct a design measure from its matrix representation `m`.

The `(N+1, K)` matrix `m` represents a `DesignMeasure` with `K` design points from an
`N`-dimensional design space. The first row of `m` must contain the weights.

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
 [9.0, 6.0] => 0.3
)
```
"""
function DesignMeasure(m::AbstractMatrix{<:Real})
    if size(m, 1) < 2
        throw(ArgumentError("m must have at least two rows"))
    end
    ws = m[1, :]
    dps = [m[2:end, k] for k in 1:size(m, 2)]
    return DesignMeasure(ws, dps)
end

"""
    singleton_design(designpoint)

Construct a one-point DesignMeasure.
"""
function singleton_design(designpoint::AbstractVector{<:Real})
    return DesignMeasure([1.0], [designpoint])
end

"""
    uniform_design(designpoints)

Construct a DesignMeasure with equal weights on the given designpoints.
"""
function uniform_design(designpoints::AbstractVector{<:AbstractVector{<:Real}})
    K = length(designpoints)
    w = fill(1 / K, K)
    return DesignMeasure(w, designpoints)
end

"""
    grid_design(ds::DesignSpace{1}, K::Integer)

Construct a DesignMeasure with an equally-spaced grid of `K` design points
and uniform weights on the given 1-dimensional design space.
"""
function grid_design(ds::DesignSpace{1}, K::Integer)
    val = range(lowerbound(ds)[1], upperbound(ds)[1]; length = K)
    designpoints = [[dp] for dp in val]
    return uniform_design(designpoints)
end

"""
    random_design(ds::DesignSpace, K::Integer)

Construct a DesignMeasure with design points drawn independently
from a uniform distribution on the design space.

Independent weights weights are drawn from a uniform distribution on [0, 1]
and then normalized to sum to one.
"""
function random_design(ds::DesignSpace{N}, K::Integer) where N
    scl = upperbound(ds) .- lowerbound(ds)
    dp = [lowerbound(ds) .+ scl .* rand(N) for _ in 1:K]
    u = rand(K)
    w = u ./ sum(u)
    d = DesignMeasure(w, dp)
    return d
end

"""
    DesignSpace(pairs::Pair...)

Convenience constructor that takes pairs of a `Symbol` name and a `Tuple` or 2-element
`Vector` for bounds.

# Examples

```jldoctest
julia> DesignSpace(:dose => (0, 300), :time => (0, 20))
DesignSpace{2}((:dose, :time), (0.0, 0.0), (300.0, 20.0))
```
"""
function DesignSpace(pairs::Pair...)
    name = [p[1] for p in pairs]
    lb = [p[2][1] for p in pairs]
    ub = [p[2][2] for p in pairs]
    return DesignSpace(name, lb, ub)
end

# === utility functions === #

function Base.show(io::IO, ::MIME"text/plain", d::DesignMeasure)
    pairs = map(weights(d), designpoints(d)) do w, dp
        return string(dp) * " => " * string(w)
    end
    print(io, typeof(d), "(\n")
    print(io, " ", join(pairs, ",\n "))
    print(io, " \n)")
end

"""
    as_matrix(d::DesignMeasure)

Return a matrix representation of `d`.

A [`DesignMeasure`](@ref) with `K` design points from an `N`-dimensional
[`DesignSpace`](@ref) corresponds to a `(N+1, K)` matrix.
The first row contains the weights.

See also [`DesignMeasure`](@ref).

# Examples

```jldoctest
julia> as_matrix(DesignMeasure([0.5, 0.2, 0.3], [[7, 4], [8, 5], [9, 6]]))
3×3 Matrix{Float64}:
 0.5  0.2  0.3
 7.0  8.0  9.0
 4.0  5.0  6.0
```
"""
function as_matrix(d::DesignMeasure)
    return vcat(transpose(weights(d)), reduce(hcat, designpoints(d)))
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
 [3.0, 4.0] => 0.25
)
```
"""
function sort_designpoints(d::DesignMeasure; rev::Bool = false)
    # note: no (deep)copies needed because indexing with `p` generates a copy
    dp = d.designpoint
    w = d.weight
    for i in length(dp[1]):-1:1
        p = sortperm(map(x -> x[i], dp); rev = rev)
        dp = dp[p]
        w = w[p]
    end
    return DesignMeasure(w, dp)
end

"""
    sort_weights(d::DesignMeasure; rev::Bool = false)

Return a representation of `d` where the design points are sorted by their
corresponding weights.

See also [`sort_designpoints`](@ref).

# Examples

```jldoctest
julia> sort_weights(DesignMeasure([0.5, 0.2, 0.3], [[1], [2], [3]]))
DesignMeasure(
 [2.0] => 0.2,
 [3.0] => 0.3,
 [1.0] => 0.5
)
```
"""
function sort_weights(d::DesignMeasure; rev::Bool = false)
    p = sortperm(d.weight; rev = rev)
    w = d.weight[p]
    dp = d.designpoint[p]
    return DesignMeasure(w, dp)
end

"""
    mixture(alpha, d1::DesignMeasure, d2::DesignMeasure)

Return the mixture ``\alpha d_1 + (1-\alpha) d_2``,
i.e. the convex combination of `d1` and `d2`.

The result is not simplified, hence its design points may not be unique.
"""
function mixture(alpha::Real, d1::DesignMeasure, d2::DesignMeasure)
    if length(d1.designpoint[1]) != length(d2.designpoint[1])
        error("design points must have identical lengths")
    end
    if alpha < 0 || alpha > 1
        error("mixture weight must be between 0 and 1")
    end
    w = vcat(alpha .* weights(d1), (1 - alpha) .* weights(d2))
    dp = vcat(designpoints(d1), designpoints(d2))
    return DesignMeasure(w, dp)
end

"""
    apportion(weights, n::Integer)

For a vector of `weights`, find an integer vector `a` with `sum(a) == n`
such that `a ./ n` best approximates `w`.

This is the _efficient design apportionment procedure_ from p. 309 in Pukelsheim [^P06].

[^P06]: Friedrich Pukelsheim, "Optimal design of experiments", Wiley, 2006. [doi:10.1137/1.9780898719109](https://doi.org/10.1137/1.9780898719109)
"""
function apportion(weights::AbstractVector{<:Real}, n::Integer)
    l = length(weights)
    m = @. ceil((n - 0.5 * l) * weights)
    while abs(sum(m) - n) > 0.5 # don't check Floats for != 0.0
        if sum(m) - n < -0.5
            j = argmin(m ./ weights)
            m[j] += 1
        elseif sum(m) - n > 0.5
            k = argmax((m .- 1) ./ weights)
            m[k] -= 1
        end
    end
    return m
end

"""
    apportion(d::Designmeasure, n)

Find an apportionment for the weights of `d`.
"""
function apportion(d::DesignMeasure, n::Integer)
    return apportion(weights(d), n)
end

"""
    simplify(designmeasure, designspace, model, covariateparameterization;
             minweight = 0, mindist = 0, uargs...)

Convenience wrapper that calls [`simplify_drop`](@ref), [`simplify_unique`](@ref), and
[`simplify_merge`](@ref).
"""
function simplify(
    d::DesignMeasure,
    ds::DesignSpace,
    m::Model,
    cp::CovariateParameterization;
    minweight = 0,
    mindist = 0,
    uargs...,
)
    d = simplify_drop(d, minweight)
    d = simplify_unique(d, ds, m, cp; uargs...)
    d = simplify_merge(d, ds, mindist)
    return d
end

"""
    simplify_drop(designmeasure, minweight)

Construct a new `DesignMeasure` where only design points with weights strictly larger than
`minweight` are kept.

The vector of remaining weights is re-normalized.
"""
function simplify_drop(d::DesignMeasure, minweight::Real)
    if length(d.weight) == 1 # nothing to do for one-point-designs
        return deepcopy(d) # return a copy for consistency
    end
    enough_weight = d.weight .> minweight
    dps = d.designpoint[enough_weight]
    ws = d.weight[enough_weight]
    ws ./= sum(ws)
    return DesignMeasure(ws, dps)
end

"""
    simplify_unique(designmeasure, designspace, model, covariateparameterization;
                    uargs...)

Construct a new DesignMeasure that corresponds uniquely to its implied normalized
information matrix.

Users should specialize this method for their concrete `Model` and
`CovariateParameterization` subtypes. It is intended for cases where the mapping from design
measure to normalized information matrix is not one-to-one. This depends on the model and
covariate parameterization used. In such a case, `simplify_unique` should be used to
canonicalize the design.

The default function simply returns the given `designmeasure`.
"""
function simplify_unique(
    d::DesignMeasure,
    ds::DesignSpace,
    m::Model,
    cp::CovariateParameterization;
    uargs...,
)
    # fallback no-op when no model-specific simplification is defined
    return deepcopy(d)
end

"""
    simplify_merge(designmeasure, designspace, mindist)

Merge designpoints with a normalized distance smaller or equal to `mindist`.

The design points are first transformed into unit (hyper)cube.
The argument `mindist` is intepreted relative to this unit cube,
i.e. only `0 < mindist < sqrt(N)` make sense for a designspace of dimension `N`.

The following two steps are repeated until all points are more than `mindist` apart:

 1. All pairwise euclidean distances are calculated.
 2. The two points closest to each other are averaged with their relative weights

Finally the design points are scaled back into the original design space.
"""
function simplify_merge(d::DesignMeasure, ds::DesignSpace, mindist::Real)
    if length(d.weight) == 1 # nothing to do for one-point-designs
        return deepcopy(d) # return a copy for consistency
    end
    # scale design space into unit cube
    width = collect(upperbound(ds) .- lowerbound(ds))
    dps = [(dp .- lowerbound(ds)) ./ width for dp in d.designpoint]
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
    dps = [(dp .* width) .+ lowerbound(ds) for dp in dps]
    return DesignMeasure(ws, dps)
end

# == abstract point methods == #

function randomize!(
    d::DesignMeasure,
    (ds, fixw, fixp)::Tuple{DesignSpace{N},Vector{Bool},Vector{Bool}},
) where N
    K = length(d.weight)
    scl = ds.upperbound .- ds.lowerbound
    for k in 1:K
        if !fixp[k]
            rand!(d.designpoint[k])
            d.designpoint[k] .*= scl
            d.designpoint[k] .+= ds.lowerbound
        end
    end
    if !all(fixw)
        # Due to rounding errors, a sum > 1.0 can happen.
        # We need to prevent negative normalizing constants later on.
        cum_sum_fix = min(1.0, sum(d.weight[fixw]))
        if cum_sum_fix == 1.0
            @warn "fixed weights already sum to one"
        end
        cum_sum_rand = 0.0
        while cum_sum_rand < eps() # we don't want to divide by too small numbers
            for k in 1:K
                if !fixw[k]
                    d.weight[k] = rand()
                    cum_sum_rand += d.weight[k]
                end
            end
        end
        norm_const = (1 - cum_sum_fix) / cum_sum_rand
        for k in 1:K
            if !fixw[k]
                d.weight[k] *= norm_const
            end
        end
    end
    return d
end

function difference!(v::AbstractVector{<:Real}, p::DesignMeasure, q::DesignMeasure)
    # v layout: concatenate the designpoints, then the first K-1 weights
    K = length(p.weight)
    weight_shift = K * length(p.designpoint[1])
    i = 1
    flat_dp_p = Iterators.flatten(p.designpoint)
    flat_dp_q = Iterators.flatten(q.designpoint)
    for (x, y) in Iterators.zip(flat_dp_p, flat_dp_q)
        v[i] = x - y
        i += 1
    end
    for k in 1:(K - 1)
        v[k + weight_shift] = p.weight[k] - q.weight[k]
    end
    return v
end

function flat_length(p::DesignMeasure)
    return length(p.weight) * (length(p.designpoint[1]) + 1) - 1
end

function copy!(to::DesignMeasure, from::DesignMeasure)
    to.weight .= from.weight
    for k in 1:length(from.designpoint)
        to.designpoint[k] .= from.designpoint[k]
    end
    return to
end

function move!(
    p::DesignMeasure,
    v::AbstractVector{<:Real},
    (ds, fixw, fixp)::Tuple{DesignSpace{N},Vector{Bool},Vector{Bool}},
) where N
    K = length(p.designpoint) # number of design points
    D = length(p.designpoint[1]) # dimension of the design space
    weight_shift = K * D # weight offset into displacement vector
    # ignore velocity components in directions that correspond to fixed weights or points
    move_handle_fixed!(v, fixw, fixp, K, weight_shift, D)
    # handle intersections: find maximal 0<=t<=1 such that p+tv remains in the search volume
    t = move_how_far(p, v, ds, K, weight_shift, D)
    if t < 0
        @warn "t=$t means point was already outside search volume" p
    end
    # Then, set p to p + tv
    move_add_v!(p, t, v, ds, fixw, K, weight_shift, D)
    # Stop the particle if the boundary was hit.
    if t != 1.0
        v .= 0.0
    end
    return p
end

function move_handle_fixed!(v, fixw, fixp, K, weight_shift, D)
    # v layout: concatenate the designpoints, then the first K-1 weights
    sum_vw_free = 0.0
    for k in 1:(K - 1)
        if fixw[k]
            v[weight_shift + k] = 0.0
        else
            sum_vw_free += v[weight_shift + k]
        end
    end
    # When the implicit last weight is fixed we must make sure only to move
    # parallel to the simplex diagonal face, i.e. that
    #
    #   sum(v[.! fixw]) == 0.
    #
    # In oder not to prefer one direction over the others, we subtract the mean
    # from every non-fixed element of v.
    n_fixw = count(fixw)
    if n_fixw != K && fixw[K]
        mean_free = sum_vw_free / (K - n_fixw)
        for k in 1:(K - 1)
            if !fixw[k]
                v[weight_shift + k] -= mean_free
            end
        end
    end
    for k in 1:K
        if fixp[k]
            for j in 1:D
                flatindex = (k - 1) * D + j
                v[flatindex] = 0.0
            end
        end
    end
    return v
end

function move_how_far(p, v, ds, K, weight_shift, D)
    # v layout: concatenate the designpoints, then the first K-1 weights.
    t = 1.0
    # box constraints
    for k in 1:K
        for j in 1:D
            # design point k element j is flattend to index (k-1)*D+j,
            # where D is the length of a single designpoint
            flatindex = (k - 1) * D + j
            t = how_far_left(p.designpoint[k][j], t, v[flatindex], ds.lowerbound[j])
            t = how_far_right(p.designpoint[k][j], t, v[flatindex], ds.upperbound[j])
        end
    end
    # simplex constraints
    for k in 1:(K - 1)
        t = how_far_left(p.weight[k], t, v[weight_shift + k], 0.0)
    end
    sum_x = 1.0 - p.weight[K]
    sum_v = @views sum(v[(weight_shift + 1):end])
    t = how_far_simplexdiag(sum_x, t, sum_v)
    return t
end

# How far can we go from x in the direction of x + tv, without landing right of ub?
# if x + tv <= ub, return `t`; else return `s` such that x + sv == ub
function how_far_right(x, t, v, ub)
    return x + t * v > ub ? (ub - x) / v : t
end

# How far can we go from x in the direction of x + tv, without landig left of lb?
# if x + tv >= lb, return `t`; else return `s` such that x + sv == lb
function how_far_left(x, t, v, lb)
    return x + t * v < lb ? (lb - x) / v : t
end

# How far can we go from x in the direction of x + tv, without crossing the
# diagonal face of the simplex?
# Note that the simplex here is {(x_1,...,x_{K-1}) : 0 <= x_k, sum_{k=1}{K-1} x_k <= 1}.
# if sum_x + t * sum_v <= 1, return `t`; else return `s` such that sum_x + s*sum_v == 1
function how_far_simplexdiag(sum_x, t, sum_v)
    return sum_x + t * sum_v > one(sum_x) ? (one(sum_x) - sum_x) / sum_v : t
end

function move_add_v!(p, t, v, ds, fixw, K, weight_shift, D)
    # first for the design points ...
    i = 1 # index to flattened structure
    for dp in p.designpoint
        for j in 1:D
            dp[j] += t * v[i]
            # Due to rounding errors, design points can be just slightly outside the design
            # space. We fix this here.
            if dp[j] > ds.upperbound[j]
                dp[j] = ds.upperbound[j]
            end
            if dp[j] < ds.lowerbound[j]
                dp[j] = ds.lowerbound[j]
            end
            i += 1
        end
    end
    # ... then for the weights.
    weight_K = 1.0
    for k in 1:(K - 1)
        p.weight[k] += t * v[weight_shift + k]
        # Again due to rounding erros, a weight can become slightly negative. We need to fix
        # this to prevent it snowballing later on.
        if p.weight[k] < 0.0
            p.weight[k] = 0.0
        end
        if p.weight[k] > 1.0
            @warn "weight would move past 1.0 by $(p.weight[k] - 1.0)"
            p.weight[k] = 1.0
        end
        weight_K -= p.weight[k]
    end
    # In `handle_fixed()` we made sure that mathematically we have
    #
    #   weight_K == 1 - sum(weight[1:(K-1)]),
    #
    # but due to rounding errors this is not the case numerically. Hence we
    # just don't touch p.weight[K] that case.
    if !fixw[K]
        p.weight[K] = weight_K
    end
    # Fix small rounding erros as above.
    if p.weight[K] < 0.0
        p.weight[K] = 0.0
    end
    return p
end
