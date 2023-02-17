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

Return all designpoints, including those with zero weight.

!!! note

    The returned vector is a reference, assigning to its elements modifies `d`.

See also: [`support`](@ref), [`weights`](@ref), [`simplify_drop`](@ref).
"""
designpoints(d::DesignMeasure) = d.designpoint

"""
    weights(d::DesignMeasure)

Return a vector of design point weights.

!!! note

    The returned vector is a reference, assigning to its elements modifies `d`.

See also: [`support`](@ref), [`designpoints`](@ref), [`simplify_drop`](@ref).
"""
weights(d::DesignMeasure) = d.weight

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
    grid_design(K::Integer, ds::DesignSpace{1})

Construct a DesignMeasure with an equally-spaced grid of `K` design points
and uniform weights on the given 1-dimensional design space.
"""
function grid_design(K::Integer, ds::DesignSpace{1})
    val = range(lowerbound(ds)[1], upperbound(ds)[1]; length = K)
    designpoints = [[dp] for dp in val]
    return uniform_design(designpoints)
end

"""
    random_design(K::Integer, ds::DesignSpace)

Construct a DesignMeasure with design points drawn independently
from a uniform distribution on the design space.

Independent weights weights are drawn from a uniform distribution on [0, 1]
and then normalized to sum to one.
"""
function random_design(K::Integer, ds::DesignSpace{N}) where N
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

## Examples

```@example
DesignSpace(:dose => (0, 300), :time => (0, 20))
```
"""
function DesignSpace(pairs::Pair...)
    name = [p[1] for p in pairs]
    lb = [p[2][1] for p in pairs]
    ub = [p[2][2] for p in pairs]
    return DesignSpace(name, lb, ub)
end

# === utility functions === #

"""
    mixture(alpha, d1::DesignMeasure, d2::DesignMeasure)

Return the mixture ``α d_1 + (1-α) d_2``,
i.e. the convex combination of `d1` and `d2`.

The result is not simplified, hence its design points may not be unique.
"""
function mixture(alpha::Real, d1::DesignMeasure, d2::DesignMeasure)
    if length(designpoints(d1)[1]) != length(designpoints(d2)[1])
        error("design points must have identical lengths")
    end
    if alpha < 0 || alpha > 1
        error("mixture weight must be between 0 and 1")
    end
    w = vcat(alpha .* weights(d1), (1 - alpha) .* weights(d2))
    dp = vcat(deepcopy(designpoints(d1)), deepcopy(designpoints(d2)))
    return DesignMeasure(w, dp)
end

"""
    apportion(w, n::Integer)

For a vector of weights `w`, find an integer vector `a` with `sum(a) == n`
such that `a ./ n` best approximates `w`.

This is the _efficient design apportionment procedure_ from p. 309 in Pukelsheim, F. (2006).
[Optimal design of experiments](https://doi.org/10.1137/1.9780898719109)
"""
function apportion(w::AbstractVector{<:Real}, n::Integer)
    l = length(w)
    m = @. ceil((n - 0.5 * l) * w)
    while abs(sum(m) - n) > 0.5 # don't check Floats for != 0.0
        if sum(m) - n < -0.5
            j = argmin(m ./ w)
            m[j] += 1
        elseif sum(m) - n > 0.5
            k = argmax((m .- 1) ./ w)
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
             minweight = 1e-4, mindist = 1e-3, moreargs...)

Convenience wrapper that calls [`simplify_drop`](@ref), [`simplify_unique`](@ref), and
[`simplify_merge`](@ref).
"""
function simplify(
    d::DesignMeasure,
    ds::DesignSpace,
    m::Model,
    cp::CovariateParameterization;
    minweight = 1e-4,
    mindist = 1e-3,
    moreargs...,
)
    d = simplify_drop(d, minweight)
    d = simplify_unique(d, ds, m, cp; moreargs...)
    d = simplify_merge(d, ds, mindist)
    return d
end

"""
    simplify_drop(designmeasure, minweight)

Construct a new `DesignMeasure` where all design points with weights smaller than
`minweight` are removed.

The vector of remaining weights is re-normalized.
"""
function simplify_drop(d::DesignMeasure, minweight::Real)
    if length(weights(d)) == 1 # nothing to do for one-point-designs
        return deepcopy(d) # return a copy for consistency
    end
    enough_weight = weights(d) .> minweight
    dps = designpoints(d)[enough_weight]
    ws = weights(d)[enough_weight]
    ws ./= sum(ws)
    return DesignMeasure(ws, dps)
end

"""
    simplify_unique(designmeasure, designspace, model, covariateparameterization;
                    moreargs...)

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
    moreargs...,
)
    # fallback no-op when no model-specific simplification is defined
    return deepcopy(d)
end

"""
    simplify_merge(designmeasure, designspace, mindist)

Merge designpoints that are less than `mindist` apart (relatively).

The design points are first transformed into unit (hyper)cube.
The argument `mindist` is intepreted relative to this unit cube,
i.e. only `0 < mindist < sqrt(N)` make sense for a designspace of dimension `N`.

The following two steps are repeated until all points are at least `mindist` apart:

 1. All pairwise euclidean distances are calculated.
 2. The two points closest to each other are averaged with their relative weights

Finally the design points are scaled back into the original design space.
"""
function simplify_merge(d::DesignMeasure, ds::DesignSpace, mindist::Real)
    if length(weights(d)) == 1 # nothing to do for one-point-designs
        return deepcopy(d) # return a copy for consistency
    end
    # scale design space into unit cube
    width = collect(upperbound(ds) .- lowerbound(ds))
    dps = [(dp .- lowerbound(ds)) ./ width for dp in designpoints(d)]
    ws = deepcopy(weights(d))
    cur_min_dist = 0
    while cur_min_dist < mindist
        # compute pairwise L2-distances, merge the two designpoints nearest to each other
        dist = map(p -> norm(p[1] .- p[2]), Iterators.product(dps, dps))
        dist[diagind(dist)] .= Inf
        cur_min_dist, idx = findmin(dist) # i > j because rows vary fastest
        i = idx[1]
        j = idx[2]
        if cur_min_dist < mindist
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
