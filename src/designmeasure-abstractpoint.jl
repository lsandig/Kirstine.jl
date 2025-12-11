# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## abstract point interface to design measures for optimizer ##

# Only used internally to represent difference vectors between DesignMeasures in
# particle-based optimizers. Structurally the same as a DesignMeasure, but weights can be
# arbitrary real numbers.
struct SignedMeasure <: AbstractPointDifference
    atoms::Matrix{Float64}
    weights::Vector{Float64}
    function SignedMeasure(atoms::AbstractMatrix{<:Real}, weights::AbstractVector{<:Real})
        err_length = ArgumentError("number of weights and atoms must be equal")
        length(weights) == size(atoms, 2) || throw(err_length)
        return new(atoms, weights)
    end
end

# analogous accessor to points(d::DesignMeasure)
atoms(s::SignedMeasure) = eachcol(s.atoms)

function Base.:(==)(s1::SignedMeasure, s2::SignedMeasure)
    return s1.weights == s2.weights && s1.atoms == s2.atoms
end

Base.hash(s::SignedMeasure, h::UInt) = hash(s.atoms, hash(s.weights, h))

struct DesignConstraints{N,T<:DesignRegion{N}} <: AbstractConstraints
    dr::T
    fixw::Vector{Bool}
    fixp::Vector{Bool}
    function DesignConstraints(
        dr::T,
        fixw::AbstractVector{Bool},
        fixp::AbstractVector{Bool},
    ) where {N,T<:DesignRegion{N}}
        err_dim = DimensionMismatch("fix vectors must have identical lengths")
        length(fixw) == length(fixp) || throw(err_dim)
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
    msg_weight = "indices for fixed weights must be between 1 and $(numpoints(d))"
    msg_point = "indices for fixed points must be between 1 and $(numpoints(d))"
    !all(in(1:numpoints(d)), fixedweights) && throw(ArgumentError(msg_weight))
    !all(in(1:numpoints(d)), fixedpoints) && throw(ArgumentError(msg_weight))
    fixw = [k in fixedweights for k in 1:numpoints(d)]
    fixp = [k in fixedpoints for k in 1:numpoints(d)]
    return DesignConstraints(dr, fixw, fixp)
end

function assert_constraints(ds::AbstractVector{DesignMeasure}, c::DesignConstraints)
    map(d -> check_compatible(d, c.dr), ds)
    msg = "all prototypes must have $(length(c.fixw)) point(s), found $(numpoints.(ds))"
    !all(d -> numpoints(d) == length(c.fixw), ds) && throw(ArgumentError(msg))
    val_fixw = [weights(d)[c.fixw] for d in ds]
    msgw = "fixed weights must be identical across prototypes, found $(val_fixw)"
    !allequal(val_fixw) && throw(ArgumentError(msgw))
    val_fixp = [points(d)[c.fixp] for d in ds]
    msgp = "fixed points must be identical across prototypes, found $(val_fixp)"
    !allequal(val_fixp) && throw(ArgumentError(msgp))
    return nothing
end

function construct_grid(
    gs::ProductGrid{N},
    c::DesignConstraints{N,<:DesignRegion{N}},
    prototypes::AbstractVector{DesignMeasure},
) where N
    isempty(prototypes) && throw(ArgumentError("grid search needs a prototype"))
    msg_dim = "grid spec and design point dimension do not match"
    N != length(first(points(first(prototypes)))) && throw(ArgumentError(msg_dim))
    !(all(c.fixw)) && throw(ArgumentError("grid on weights is not implemented"))
    # first build a grid on the design region
    region_ranges = marginal_grids(c.dr, gs.n_points)
    bounding_box_grid = collect.(Iterators.product(region_ranges...))
    grid_points_in_region = filter(p -> isinside(p, c.dr), bounding_box_grid)
    msg_constraint = "no grid points satisfy design constraints"
    isempty(grid_points_in_region) && throw(ArgumentError(msg_constraint))
    # then build a product of grids for each point of the design measure;
    # each "factor" is either the grid_points_in_region (for free points)
    # or a vector with a single design point (for fixed points)
    prototype_weights = weights(first(prototypes))
    prototype_points = collect.(points(first(prototypes)))
    points_ranges = map(1:length(prototype_points)) do k
        return c.fixp[k] ? [prototype_points[k]] : grid_points_in_region
    end
    # Note: The length of prototype_points is not known at compile time.
    # Hence, the following product iterator does not have a concrete type.
    # Without the manual type annotation, g would be of type Any.
    # This problem does _not_ occur above for bounding_box_grid
    # since dimension(c.dr) == N _is_ known at compile time.
    product_iterator = Iterators.product(points_ranges...)
    points_grid = vec(collect.(product_iterator))::Vector{Vector{Vector{Float64}}}
    g = [DesignMeasure(pts, prototype_weights) for pts in points_grid]
    return g
end

# Note: in contrast to ap_diff(p, q), this does _not_ consider the last weight to be specified implicitly.
# For Frank-Wolfe it is more convenient to use the full K-dimensional weights vector.
function fw_norm(s::SignedMeasure)
    # Note that we map over the single entries of the s.atoms matrix here!
    ssq = mapreduce(x -> x^2, +, s.atoms) + mapreduce(x -> x^2, +, s.weights)
    return sqrt(ssq)
end

function fw_gradient_length(d::DesignMeasure, c::DesignConstraints)
    dim = length(first(points(d)))
    return (numpoints(d) - sum(c.fixp)) * dim + numpoints(d) - sum(c.fixw)
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
    any(c.fixw) && error("only implemented for all weights nonfixed")
    v.weights .= 0
    v.weights[argmax(gx)] = 1
    return v
end

function fw_evaluate_linearization(
    gx::AbstractVector{<:Real},
    s::SignedMeasure,
    c::DesignConstraints,
)
    !all(c.fixp) && error("only implemented for fixed points")
    return dot(gx, s.weights)
end

function fw_evaluate_linearization(
    gx::AbstractVector{<:Real},
    d::DesignMeasure,
    c::DesignConstraints,
)
    !all(c.fixp) && error("only implemented for fixed points")
    return dot(gx, d.weights)
end

# Note: assumes that any fixed entries are identical for all design measures in qs.
# This does _not_ handle nonconvex design regions,
# and does _not_ correct for numerical errors that might move the point outside of the region.
function ap_convex_combination!(
    p::DesignMeasure,
    alpha::AbstractVector{<:Real},
    qs::AbstractVector{DesignMeasure},
    c::DesignConstraints,
)
    err_length = ArgumentError("not the same number of points and weights")
    length(alpha) == length(qs) || throw(err_length)
    sum(alpha) ≈ 1.0 || throw(ArgumentError("sum(alpha) must be ≈ 1"))
    for k in 1:length(p.weights)
        if c.fixw[k]
            p.weights[k] = qs[1].weights[k]
        else
            p.weights[k] = 0
            for l in 1:length(alpha)
                p.weights[k] += alpha[l] * qs[l].weights[k]
            end
            p.weights[k] = min(p.weights[k], 1.0)
        end
    end
    p.weights ./= sum(p.weights)
    for k in 1:numpoints(p)
        if c.fixp[k]
            points(p)[k] .= points(qs[1])[k]
        else
            points(p)[k] .= 0
            for l in 1:length(alpha)
                points(p)[k] .+= alpha[l] .* points(qs[l])[k]
            end
        end
    end
    return p
end

function ap_rand!(d::DesignMeasure, c::DesignConstraints)
    for k in 1:numpoints(d)
        if !c.fixp[k]
            random_designpoint!(points(d)[k], c.dr)
        end
    end
    if !all(c.fixw)
        # Due to rounding errors, a sum > 1.0 can happen.
        # We need to prevent negative normalizing constants later on.
        sum_fix = min(1.0, sum(weights(d)[c.fixw]))
        sum_fix == 1.0 && @warn("fixed weights already sum to one")
        # rationale for the weights: when no indices are fixed, renormalized exponential
        # weights imply a uniform distribution on the simplex, i.e. Dirichlet(1, …, 1).
        sum_rand = 0.0
        while sum_rand < eps() # we don't want to divide by too small numbers
            for k in 1:numpoints(d)
                if !c.fixw[k]
                    d.weights[k] = -log(rand())
                    sum_rand += d.weights[k]
                end
            end
        end
        norm_const = (1 - sum_fix) / sum_rand
        for k in 1:numpoints(d)
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

function ap_copy!(to::SignedMeasure, from::SignedMeasure)
    to.weights .= from.weights
    to.atoms .= from.atoms
    return to
end

function ap_as_difference(p::DesignMeasure)
    return SignedMeasure(deepcopy(p.points), deepcopy(p.weights))
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
    t = t < 1e-8 ? 0.0 : t
    # Then, set p to p + tv
    move_add_tv!(p, t, v, c.dr, c.fixw)
    # Stop the particle if the boundary was hit.
    if t != 1.0
        ap_mul!(v, 0)
    end
    # check that we have not accidentally moved outside
    check_compatible(p, c.dr)
    return p
end

function move_handle_fixed!(
    v::SignedMeasure,
    fixw::AbstractVector{Bool},
    fixp::AbstractVector{Bool},
)
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
    # from every nonfixed element of v.weights.
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
    # For each designpoint figure out how far we can go along v.
    # The minimum of these lengths is how far we can go overall.
    for k in 1:K
        t = min(t, move_designpoint_how_far(points(p)[k], atoms(v)[k], dr))
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

function move_add_tv!(
    p::DesignMeasure,
    t::Real,
    v::SignedMeasure,
    dr::DesignRegion{N},
    fixw::AbstractVector{Bool},
) where N
    K = numpoints(p)
    # first for the design points ...
    map((x, y) -> move_designpoint!(x, t, y, dr), points(p), atoms(v))
    # ... then for the weights.
    p.weights .+= t .* v.weights
    weight_K = 1.0
    for k in 1:(K - 1)
        # Again due to rounding erros, a weight can become slightly negative. We need to fix
        # this to prevent it snowballing later on.
        p.weights[k] = min(max(p.weights[k], 0.0), 1.0)
        weight_K -= p.weights[k]
    end
    # In `move_handle_fixed!()` we made sure that mathematically we have
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
