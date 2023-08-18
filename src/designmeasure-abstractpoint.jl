# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## abstract point interface to design measures for optimizer ##

# Only used internally to represent difference vectors between DesignMeasures in
# particle-based optimizers. Structurally the same as a DesignMeasure, but weights can be
# arbitrary real numbers.
struct SignedMeasure <: AbstractPointDifference
    atom::Vector{Vector{Float64}}
    weight::Vector{Float64}
    function SignedMeasure(atoms, weights)
        if length(weights) != length(atoms)
            error("number of weights and atoms must be equal")
        end
        if !allequal(map(length, atoms))
            error("atoms must have identical lengths")
        end
        new(atoms, weights)
    end
end

struct DesignConstraints{N,T<:DesignRegion{N}} <: AbstractConstraints
    dr::T
    fixw::Vector{Bool}
    fixp::Vector{Bool}
    function DesignConstraints(dr::T, fixw, fixp) where {N,T<:DesignRegion{N}}
        if length(fixw) != length(fixp)
            throw(DimensionMismatch("fix vectors must have identical lengths"))
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
    K = length(weights(d))
    if any(fixedweights .< 1) || any(fixedweights .> K)
        error("indices for fixed weights must be between 1 and $K")
    end
    if any(fixedpoints .< 1) || any(fixedpoints .> K)
        error("indices for fixed points must be between 1 and $K")
    end
    fixw = [k in fixedweights for k in 1:K]
    fixp = [k in fixedpoints for k in 1:K]
    # Fixing all weights but one is equivalent to fixing them all. For
    # numerical stability it is better to explicitly fix them all.
    if count(fixw) == K - 1
        @info "explicitly fixing implicitly fixed weight"
        fixw .= true
    end
    return DesignConstraints(dr, fixw, fixp)
end

function check_compatible(d::DesignMeasure, dr::DesignInterval)
    lb = dr.lowerbound
    ub = dr.upperbound
    for dp in d.designpoint
        if length(dp) != length(lb)
            error("designpoint length must match design region dimension")
        end
        if any(dp .< lb) || any(dp .> ub)
            sandwich = hcat([:lb, :dp, :ub], permutedims([[lb...] dp [ub...]]))
            # error() does not pretty print matrices, so we manually format it
            b = IOBuffer()
            show(b, "text/plain", sandwich)
            sstr = String(take!(b))
            error("designpoint is outside design region\n $sstr")
        end
    end
    return true
end

function ap_random_point!(
    d::DesignMeasure,
    c::DesignConstraints{N,DesignInterval{N}},
) where N
    K = length(d.weight)
    scl = c.dr.upperbound .- c.dr.lowerbound
    for k in 1:K
        if !c.fixp[k]
            rand!(d.designpoint[k])
            d.designpoint[k] .*= scl
            d.designpoint[k] .+= c.dr.lowerbound
        end
    end
    if !all(c.fixw)
        # Due to rounding errors, a sum > 1.0 can happen.
        # We need to prevent negative normalizing constants later on.
        cum_sum_fix = min(1.0, sum(d.weight[c.fixw]))
        if cum_sum_fix == 1.0
            @warn "fixed weights already sum to one"
        end
        cum_sum_rand = 0.0
        while cum_sum_rand < eps() # we don't want to divide by too small numbers
            for k in 1:K
                if !c.fixw[k]
                    d.weight[k] = rand()
                    cum_sum_rand += d.weight[k]
                end
            end
        end
        norm_const = (1 - cum_sum_fix) / cum_sum_rand
        for k in 1:K
            if !c.fixw[k]
                d.weight[k] *= norm_const
            end
        end
    end
    return d
end

function ap_difference!(v::SignedMeasure, p::DesignMeasure, q::DesignMeasure)
    K = length(p.weight)
    v.weight .= p.weight .- q.weight
    for k in 1:K
        v.atom[k] .= p.designpoint[k] .- q.designpoint[k]
    end
    return v
end

function ap_copy!(to::DesignMeasure, from::DesignMeasure)
    to.weight .= from.weight
    for k in 1:length(from.designpoint)
        to.designpoint[k] .= from.designpoint[k]
    end
    return to
end

function ap_as_difference(p::DesignMeasure)
    return SignedMeasure(deepcopy(p.designpoint), deepcopy(p.weight))
end

function ap_random_difference!(v::SignedMeasure)
    rand!(v.weight)
    for k in 1:length(v.weight)
        rand!(v.atom[k])
    end
    return v
end

function ap_mul_hadamard!(v1::SignedMeasure, v2::SignedMeasure)
    v1.weight .*= v2.weight
    for k in 1:length(v1.weight)
        v1.atom[k] .*= v2.atom[k]
    end
    return v1
end

function ap_mul_scalar!(v::SignedMeasure, a::Real)
    v.weight .*= a
    for k in 1:length(v.weight)
        v.atom[k] .*= a
    end
    return v
end

function ap_add!(v1::SignedMeasure, v2::SignedMeasure)
    v1.weight .+= v2.weight
    for k in 1:length(v1.weight)
        v1.atom[k] .+= v2.atom[k]
    end
    return v1
end

function ap_move!(p::DesignMeasure, v::SignedMeasure, c::DesignConstraints)
    K = length(p.designpoint) # number of design points
    # ignore velocity components in directions that correspond to fixed weights or points
    move_handle_fixed!(v, c.fixw, c.fixp)
    # handle intersections: find maximal 0<=t<=1 such that p+tv remains in the search volume
    t = move_how_far(p, v, c.dr)
    # Then, set p to p + tv
    move_add_v!(p, t, v, c.dr, c.fixw)
    # Stop the particle if the boundary was hit.
    if t != 1.0
        ap_mul_scalar!(v, 0)
    end
    # check that we have not accidentally moved outside
    check_compatible(p, c.dr)
    return p
end

function move_handle_fixed!(v::SignedMeasure, fixw, fixp)
    K = length(v.weight)
    sum_vw_free = 0.0
    for k in 1:(K - 1)
        if fixw[k]
            v.weight[k] = 0.0
        else
            sum_vw_free += v.weight[k]
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
    #   sum(v.weight[.! fixw]) == 0.
    #
    # In oder not to prefer one direction over the others, we subtract the mean
    # from every non-fixed element of v.weight.
    n_fixw = count(fixw)
    if n_fixw != K && fixw[K]
        mean_free = sum_vw_free / (K - n_fixw)
        for k in 1:(K - 1)
            if !fixw[k]
                v.weight[k] -= mean_free
            end
        end
    end
    for k in 1:K
        if fixp[k]
            v.atom[k] .= 0.0
        end
    end
    return v
end

function move_how_far(p::DesignMeasure, v::SignedMeasure, dr::DesignInterval{N}) where N
    t = 1.0
    K = length(p.designpoint)
    # box constraints
    for k in 1:K
        for j in 1:N
            t = how_far_left(p.designpoint[k][j], t, v.atom[k][j], dr.lowerbound[j])
            t = how_far_right(p.designpoint[k][j], t, v.atom[k][j], dr.upperbound[j])
        end
    end
    # simplex constraints
    for k in 1:(K - 1) # ingore implicit last weight
        t = how_far_left(p.weight[k], t, v.weight[k], 0.0)
    end
    sum_x = 1.0 - p.weight[K]
    sum_v = @views sum(v.weight[1:(K - 1)])
    t = how_far_simplexdiag(sum_x, t, sum_v)
    return t
end

# How far can we go from x in the direction of x + tv, without landing right of ub?
# if x + tv <= ub, return `t`; else return `s` such that x + sv == ub
function how_far_right(x, t, v, ub)
    return x + t * v > ub ? (ub - x) / v : t
end

# How far can we go from x in the direction of x + tv, without landing left of lb?
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

function move_add_v!(
    p::DesignMeasure,
    t,
    v::SignedMeasure,
    dr::DesignInterval{N},
    fixw,
) where N
    K = length(p.designpoint)
    # first for the design points ...
    for k in 1:K
        p.designpoint[k] .+= t .* v.atom[k]
        for j in 1:N
            # Due to rounding errors, design points can be just slightly outside the design
            # interval. We fix this here.
            p.designpoint[k][j] =
                min(max(p.designpoint[k][j], dr.lowerbound[j]), dr.upperbound[j])
        end
    end
    # ... then for the weights.
    p.weight .+= t .* v.weight
    weight_K = 1.0
    for k in 1:(K - 1)
        # Again due to rounding erros, a weight can become slightly negative. We need to fix
        # this to prevent it snowballing later on.
        p.weight[k] = min(max(p.weight[k], 0.0), 1.0)
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
    p.weight[K] = max(p.weight[K], 0.0)
    return p
end
