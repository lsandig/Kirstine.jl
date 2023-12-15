# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## functions on the supertype

"""
    dimension(dr::DesignRegion{N})

Return the dimension `N` of the design region.
"""
dimension(dr::DesignRegion{N}) where N = N

## concrete design regions and constraints for optimization ##

"""
    DesignInterval{N} <: DesignRegion{N}

A (hyper)rectangular subset of ``\\Reals^N``.

See also [`lowerbound`](@ref), [`upperbound`](@ref), [`dimnames`](@ref).
"""
struct DesignInterval{N} <: DesignRegion{N}
    name::NTuple{N,Symbol}
    lowerbound::NTuple{N,Float64}
    upperbound::NTuple{N,Float64}
    @doc """
        DesignInterval(names, lowerbounds, upperbounds)

    Construct a design interval.

    # Examples
    ```jldoctest
    julia> DesignInterval([:dose, :time], [0, 0], [300, 20])
    DesignInterval{2}((:dose, :time), (0.0, 0.0), (300.0, 20.0))
    ```
    """
    function DesignInterval(name, lowerbound, upperbound)
        n = length(name)
        if !(n == length(lowerbound) == length(upperbound))
            error("lengths of name, upper and lower bounds must be identical")
        end
        if any(upperbound .<= lowerbound)
            error("upper bounds must be strictly larger than lower bounds")
        end
        new{n}(
            tuple(name...),
            tuple(convert.(Float64, lowerbound)...),
            tuple(convert.(Float64, upperbound)...),
        )
    end
end

"""
    lowerbound(dr::DesignInterval)

Return the vector of lower bounds.
"""
lowerbound(dr::DesignInterval) = dr.lowerbound

"""
    upperbound(dr::DesignInterval)

Return the vector of upper bounds.
"""
upperbound(dr::DesignInterval) = dr.upperbound

# Note: docstring for the supertype, implementation for the subtypes
"""
    dimnames(dr::DesignRegion)

Return the names of the design region's dimensions.
"""
dimnames(dr::DesignInterval) = dr.name

"""
    DesignInterval(name_bounds::Pair...)

Construct a design interval from `name => (lb, ub)` pairs for individual dimensions.

Note that the order of the arguments matters
and that it should match the implementation of [`map_to_covariate!`](@ref)
for your `Model` and `Covariate` subtypes.

# Examples

```jldoctest
julia> DesignInterval(:dose => (0, 300), :time => (0, 20))
DesignInterval{2}((:dose, :time), (0.0, 0.0), (300.0, 20.0))
```
"""
function DesignInterval(name_bounds::Pair...)
    name = [p[1] for p in name_bounds]
    lb = [p[2][1] for p in name_bounds]
    ub = [p[2][2] for p in name_bounds]
    return DesignInterval(name, lb, ub)
end

# internal method for testing whether a design point is inside the interval
function isinside(dp::AbstractVector{<:Real}, dr::DesignInterval)
    return all(dp .>= lowerbound(dr)) && all(dp .<= upperbound(dr))
end

# generate a single design point inside the interval
function random_designpoint!(dp::AbstractVector{<:Real}, dr::DesignInterval)
    scl = upperbound(dr) .- lowerbound(dr)
    rand!(dp)
    dp .*= scl
    dp .+= lowerbound(dr)
    return dp
end

# If dp + v is inside dr, return 1.
# Else return smallest t ∈ [0, 1) such that dp + t * v is on the boundary of dr
function move_designpoint_how_far(dp, v, dr::DesignInterval{N}) where N
    t = 1.0
    lb = lowerbound(dr)
    ub = upperbound(dr)
    for j in 1:N
        # Intersect only if we would step outside when moving a distance of `t` along `v`.
        # Do this for both bounds and set the new to to the minimum.
        sl = dp[j] + t * v[j] > ub[j] ? (ub[j] - dp[j]) / v[j] : t
        sr = dp[j] + t * v[j] < lb[j] ? (lb[j] - dp[j]) / v[j] : t
        t = min(sl, sr)
    end
    return t
end

# Move dp to dp + t * v, and make sure that the result numerically stays inside or on the boundary of dr
function move_designpoint!(dp, t, v, dr::DesignInterval{N}) where N
    dp .+= t .* v
    # Due to rounding errors, design points can be just slightly outside the design
    # interval. We fix this here.
    dp .= min.(max.(dp, dr.lowerbound), dr.upperbound)
    return dp
end

function boundingbox(dr::DesignInterval)
    return lowerbound(dr), upperbound(dr)
end
