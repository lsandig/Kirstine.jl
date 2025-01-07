# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## abstract point interface to design measures for optimizer ##

# Only used internally to represent difference vectors between DesignMeasures in
# particle-based optimizers. Structurally the same as a DesignMeasure, but weights can be
# arbitrary real numbers.
struct SignedMeasure <: AbstractPointDifference
    atoms::Matrix{Float64}
    weights::Vector{Float64}
    function SignedMeasure(atoms, weights)
        if length(weights) != size(atoms, 2)
            throw(ArgumentError("number of weights and atoms must be equal"))
        end
        new(atoms, weights)
    end
end

# analogous accessor to points(d::DesignMeasure)
atoms(s::SignedMeasure) = eachcol(s.atoms)

function Base.:(==)(s1::SignedMeasure, s2::SignedMeasure)
    return s1.weights == s2.weights && s1.atoms == s2.atoms
end

function Base.hash(s::SignedMeasure, h::UInt)
    return hash(s.atoms, hash(s.weights, h))
end

struct DesignConstraints{N,T<:DesignRegion{N}} <: AbstractConstraints
    dr::T
    fixw::Vector{Bool}
    fixp::Vector{Bool}
    function DesignConstraints(dr::T, fixw, fixp) where {N,T<:DesignRegion{N}}
        if length(fixw) != length(fixp)
            throw(DimensionMismatch("fix vectors must have identical lengths"))
        end
        # Fixing all weights but one is mathematically equivalent to fixing them all.
        # For numerical stability and identical program behavior in both cases
        # it is better to explicitly fix them all.
        if count(fixw) == length(fixw) - 1
            @info "explicitly fixing implicitly fixed weight"
            fixw .= true
        end
        return new{N,T}(dr, fixw, fixp)
    end
end

function DesignConstraints(
    d::DesignMeasure,
    dr::DesignRegion,
    fixedweights::AbstractVector{<:Integer},
    fixedpoints::AbstractVector{<:Integer},
)
    check_compatible(d, dr)
    K = numpoints(d)
    if any(fixedweights .< 1) || any(fixedweights .> K)
        throw(ArgumentError("indices for fixed weights must be between 1 and $K"))
    end
    if any(fixedpoints .< 1) || any(fixedpoints .> K)
        throw(ArgumentError("indices for fixed points must be between 1 and $K"))
    end
    fixw = [k in fixedweights for k in 1:K]
    fixp = [k in fixedpoints for k in 1:K]
    return DesignConstraints(dr, fixw, fixp)
end

function check_compatible(d::DesignMeasure, dr::DesignRegion{N}) where N
    for dp in points(d)
        if length(dp) != N
            throw(ArgumentError("design point length must match design region dimension"))
        end
        if !isinside(dp, dr)
            @error "" dp dr
            throw(ArgumentError("design point is outside design region"))
        end
    end
    return true
end

function assert_constraints(ds::AbstractVector{DesignMeasure}, c::DesignConstraints)
    for d in ds
        check_compatible(d, c.dr)
    end
    K = length(c.fixw)
    np = numpoints.(ds)
    if !all(np .== K)
        msg = "all prototypes must have $K point(s), found $(np)"
        throw(ArgumentError(msg))
    end
    val_fixw = [weights(d)[c.fixw] for d in ds]
    val_fixp = [points(d)[c.fixp] for d in ds]
    if !allequal(val_fixw)
        msg = "fixed weights must be identical across prototypes, found $(val_fixw)"
        throw(ArgumentError(msg))
    end
    if !allequal(val_fixp)
        msg = "fixed points must be identical across prototypes, found $(val_fixp)"
        throw(ArgumentError(msg))
    end
    return nothing
end

function construct_grid(
    gs::ProductGrid{N},
    c::DesignConstraints{N,<:DesignRegion{N}},
    prototypes::AbstractVector{DesignMeasure},
) where N
    if isempty(prototypes)
        throw(ArgumentError("grid search needs a prototype"))
    elseif N != length(first(points(first(prototypes))))
        throw(ArgumentError("grid spec and design point dimension do not match"))
    elseif !(all(c.fixw))
        throw(ArgumentError("grid on weights is not implemented"))
    end
    # first build a grid on the design region
    region_ranges = marginal_grids(c.dr, gs.n_points)
    bbg = collect.(Iterators.product(region_ranges...))
    region_grid = filter(p -> isinside(p, c.dr), bbg)
    if isempty(region_grid)
        throw(ArgumentError("no grid points satisfy design constraints"))
    end
    # then build a product of grids for each point of the design measure;
    # each "factor" is either the region_grid (for free points)
    # or a vector with a single design point (for fixed points)
    prototype_weights = weights(first(prototypes))
    prototype_points = collect.(points(first(prototypes)))
    points_ranges = map(1:length(prototype_points)) do k
        c.fixp[k] ? [prototype_points[k]] : region_grid
    end
    # Note: The length of prototype_points is not known at compile time.
    # Hence, the following product iterator does not have a concrete type.
    # Without the manual type annotation, g would be of type Any.
    # This problem does _not_ occur above for bbg
    # since dimension(c.dr) == N _is_ known at compile time.
    pr_iter = Iterators.product(points_ranges...)
    points_grid = vec(collect.(pr_iter))::Vector{Vector{Vector{Float64}}}
    g = [DesignMeasure(pts, prototype_weights) for pts in points_grid]
    return g
end

function fw_gradient_length(d::DesignMeasure, c::DesignConstraints)
    K = numpoints(d)
    dim = length(first(points(d)))
    return (K - sum(c.fixp)) * dim + K - sum(c.fixw)
end

# Since we're only optimizing weights, our convex region is a probability simplex.
# For a K-point design, its vertices v_1, ..., v_K are the unit basis vectors of ℝ^K.
# In Frank-Wolfe notation, to maximize v_k' * g,
# we just need to find the index of the largest element of g
# and return the corresponding v_k.
function fw_linear_maximization_oracle!(
    v::DesignMeasure,
    gx::AbstractVector{<:Real},
    c::DesignConstraints,
)
    !all(c.fixp) && error("only implemented for fixed points")
    v.weights .= 0
    v.weights[argmax(gx)] = 1
    return v
end

function ap_rand!(d::DesignMeasure, c::DesignConstraints)
    K = numpoints(d)
    pts = points(d)
    for k in 1:K
        if !c.fixp[k]
            random_designpoint!(pts[k], c.dr)
        end
    end
    if !all(c.fixw)
        # Due to rounding errors, a sum > 1.0 can happen.
        # We need to prevent negative normalizing constants later on.
        cum_sum_fix = min(1.0, sum(weights(d)[c.fixw]))
        if cum_sum_fix == 1.0
            @warn "fixed weights already sum to one"
        end
        # rationale for the weights: when no indices are fixed, re-normalized exponential
        # weights imply a uniform distribution on the simplex, i.e. Dirichlet(1, …, 1).
        cum_sum_rand = 0.0
        while cum_sum_rand < eps() # we don't want to divide by too small numbers
            for k in 1:K
                if !c.fixw[k]
                    d.weights[k] = -log(rand())
                    cum_sum_rand += d.weights[k]
                end
            end
        end
        norm_const = (1 - cum_sum_fix) / cum_sum_rand
        for k in 1:K
            if !c.fixw[k]
                d.weights[k] *= norm_const
            end
        end
    end
    return d
end

function ap_diff!(v::SignedMeasure, p::DesignMeasure, q::DesignMeasure)
    v.weights .= p.weights .- q.weights
    v.atoms .= p.points .- q.points
    return v
end

function ap_copy!(to::DesignMeasure, from::DesignMeasure)
    to.weights .= from.weights
    to.points .= from.points
    return to
end

function ap_as_difference(p::DesignMeasure)
    return SignedMeasure(deepcopy(p.points), deepcopy(p.weights))
end

function ap_dist(p::DesignMeasure, q::DesignMeasure)
    acc = 0.0
    for i in 1:length(p.points)
        acc += (p.points[i] - q.points[i])^2
    end
    for k in 1:(length(p.weights) - 1)
        acc += (p.weights[k] - q.weights[k])^2
    end
    return sqrt(acc)
end

function ap_rand!(v::SignedMeasure, lb::Real, ub::Real)
    rand!(v.weights)
    v.weights .*= (ub - lb)
    v.weights .+= lb
    rand!(v.atoms)
    v.atoms .*= (ub - lb)
    v.atoms .+= lb
    return v
end

function ap_mul!(v1::SignedMeasure, v2::SignedMeasure)
    v1.weights .*= v2.weights
    v1.atoms .*= v2.atoms
    return v1
end

function ap_mul!(v::SignedMeasure, a::Real)
    v.weights .*= a
    v.atoms .*= a
    return v
end

function ap_add!(v1::SignedMeasure, v2::SignedMeasure)
    v1.weights .+= v2.weights
    v1.atoms .+= v2.atoms
    return v1
end

function ap_add!(p::DesignMeasure, v::SignedMeasure, c::DesignConstraints)
    # ignore velocity components in directions that correspond to fixed weights or points
    move_handle_fixed!(v, c.fixw, c.fixp)
    # handle intersections: find maximal 0<=t<=1 such that p+tv remains in the search volume
    t = move_how_far(p, v, c.dr)
    # treat a very small t as zero to reduce numerical inaccuracies in the next step
    if t < 1e-8
        t = 0.0
    end
    # Then, set p to p + tv
    move_add_v!(p, t, v, c.dr, c.fixw)
    # Stop the particle if the boundary was hit.
    if t != 1.0
        ap_mul!(v, 0)
    end
    # check that we have not accidentally moved outside
    check_compatible(p, c.dr)
    return p
end

function move_handle_fixed!(v::SignedMeasure, fixw, fixp)
    K = length(v.weights)
    sum_vw_free = 0.0
    for k in 1:(K - 1)
        if fixw[k]
            v.weights[k] = 0.0
        else
            sum_vw_free += v.weights[k]
        end
    end
    # We treat the final weight as implicitly determined by the first (K-1) ones. When it is
    # not fixed this is unproblematic, as we can simply ignore the coresponding velocity
    # compound in the move operation, and as a final step set it to 1 - sum(weight[1:K-1]).
    #
    # When all weights are fixed, we also don't have to do anything.
    #
    # Only when the final weight is fixed, but some others are not, we have to be more
    # careful. We must make sure to only move parallel to the simplex diagonal face, i.e.
    # that
    #
    #   sum(v.weights[.! fixw]) == 0.
    #
    # In oder not to prefer one direction over the others, we subtract the mean
    # from every non-fixed element of v.weights.
    n_fixw = count(fixw)
    if n_fixw != K && fixw[K]
        mean_free = sum_vw_free / (K - n_fixw)
        for k in 1:(K - 1)
            if !fixw[k]
                v.weights[k] -= mean_free
            end
        end
    end
    for k in 1:K
        if fixp[k]
            v.atoms[:, k] .= 0.0
        end
    end
    return v
end

function move_how_far(p::DesignMeasure, v::SignedMeasure, dr::DesignRegion{N}) where N
    t = 1.0
    # design region constraints
    K = numpoints(p)
    pts = points(p)
    ats = atoms(v)
    # For each designpoint figure out how far we can go along v.
    # The minimum of these lengths is how far we can go overall.
    for k in 1:K
        t = min(t, move_designpoint_how_far(pts[k], ats[k], dr))
    end
    # simplex constraints
    #  * lower bounds are zero
    for k in 1:(K - 1) # ingore implicit last weight
        s = p.weights[k] + t * v.weights[k] < 0 ? -p.weights[k] / v.weights[k] : t
        t = min(s, t)
    end
    #  * upper bound for sum of first K-1 elements is one
    sum_x = 1.0 - p.weights[K]
    sum_v = @views sum(v.weights[1:(K - 1)])
    s = sum_x + t * sum_v > 1 ? (1 - sum_x) / sum_v : t
    t = min(s, t)
    return t
end

function move_add_v!(
    p::DesignMeasure,
    t,
    v::SignedMeasure,
    dr::DesignRegion{N},
    fixw,
) where N
    K = numpoints(p)
    # first for the design points ...
    pts = points(p)
    ats = atoms(v)
    for k in 1:K
        move_designpoint!(pts[k], t, ats[k], dr)
    end
    # ... then for the weights.
    p.weights .+= t .* v.weights
    weight_K = 1.0
    for k in 1:(K - 1)
        # Again due to rounding erros, a weight can become slightly negative. We need to fix
        # this to prevent it snowballing later on.
        p.weights[k] = min(max(p.weights[k], 0.0), 1.0)
        weight_K -= p.weights[k]
    end
    # In `handle_fixed()` we made sure that mathematically we have
    #
    #   weight_K == 1 - sum(weight[1:(K-1)]),
    #
    # but due to rounding errors this is not the case numerically. Hence we
    # just don't touch p.weights[K] that case.
    if !fixw[K]
        p.weights[K] = weight_K
    end
    # Fix small rounding erros as above.
    p.weights[K] = max(p.weights[K], 0.0)
    return p
end
