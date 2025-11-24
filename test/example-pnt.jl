# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

struct Pnt <: Kirstine.AbstractPoint
    x::Vector{Float64}
end

function Base.:(==)(p1::Pnt, p2::Pnt)
    return p1.x == p2.x
end

function Base.hash(p::Pnt, h::UInt)
    return hash(p.x, h)
end

struct PntDiff <: Kirstine.AbstractPointDifference
    v::Vector{Float64}
end

struct BoxConstraints <: Kirstine.AbstractConstraints
    lb::Pnt
    ub::Pnt
end

function Kirstine.assert_constraints(ps::AbstractVector{Pnt}, c::BoxConstraints)
    all_above_lb = all(p -> all(p.x .>= c.lb.x), ps)
    all_below_ub = all(p -> all(p.x .<= c.ub.x), ps)
    if !(all_above_lb && all_below_ub)
        throw(ArgumentError("prototypes do not satisfy constraints"))
    end
    return nothing
end

function Kirstine.ap_rand!(p::Pnt, c::BoxConstraints)
    rand!(p.x)
    p.x .*= c.ub.x .- c.lb.x
    p.x .+= c.lb.x
    return p
end

function Kirstine.ap_diff!(d::PntDiff, p::Pnt, q::Pnt)
    d.v .= p.x .- q.x
    return d
end

function Kirstine.ap_copy!(to::Pnt, from::Pnt)
    to.x .= from.x
    return to
end

function Kirstine.ap_add!(p::Pnt, d::PntDiff, c::BoxConstraints)
    p.x .+= d.v
    p.x .= max.(c.lb.x, p.x)
    p.x .= min.(c.ub.x, p.x)
    return p
end

function Kirstine.ap_as_difference(p::Pnt)
    return PntDiff(deepcopy(p.x))
end

function Kirstine.ap_rand!(d::PntDiff, lb::Real, ub::Real)
    rand!(d.v)
    d.v .*= (ub - lb)
    d.v .+= lb
    return d
end

function Kirstine.ap_mul!(d1::PntDiff, d2::PntDiff)
    d1.v .*= d2.v
    return d1
end

function Kirstine.ap_mul!(d::PntDiff, a::Real)
    d.v .*= a
    return d
end

function Kirstine.ap_add!(d1::PntDiff, d2::PntDiff)
    d1.v .+= d2.v
    return d1
end

function Kirstine.ap_copy!(d1::PntDiff, d2::PntDiff)
    d1.v .= d2.v
    return d1
end
