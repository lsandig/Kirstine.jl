# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

# model and covariate subtypes for tests where we do not need a full design problem

struct TestMod <: NonlinearRegression end

mutable struct TestCovar2 <: Covariate
    a::Float64
    b::Float64
end

mutable struct TestCovarNonFloat <: Covariate
    a::Int64
    b::Vector{Float64}
end
