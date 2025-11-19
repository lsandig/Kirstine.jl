# SPDX-FileCopyrightText: 2023,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

# concrete parameter subtypes for tests where we do not need a full design problem

struct TestPar2 <: Kirstine.Parameter
    a::Float64
    b::Float64
end

struct TestPar3 <: Kirstine.Parameter
    a::Float64
    b::Float64
    c::Float64
end

struct TestPar3opt <: Kirstine.Parameter
    a::Float64
    b::Float64
    c::Float64
    c_is_known::Bool
end

Kirstine.dimension(p::TestPar2) = 2

Kirstine.dimension(p::TestPar3) = 3

Kirstine.dimension(p::TestPar3opt) = p.c_is_known ? 2 : 3
