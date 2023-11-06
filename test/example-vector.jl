# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

# Vector-unit model with non-constant measurement error variance
#
# This is a modification of `example-compartment.jl`, and not necessarily meaningful
# biologically.

struct VUMod <: NonlinearRegression
    sigma_squared::Float64
    m::Int64
end

mutable struct VUCovariate <: Covariate
    time::Vector{Float64}
end

Kirstine.unit_length(m::VUMod) = m.m

function Kirstine.update_model_vcov!(s, m::VUMod, c::VUCovariate)
    fill!(s, 0.0)
    for j in 1:(m.m)
        s[j, j] = m.sigma_squared * (1 + c.time[j])
    end
    return s
end

Kirstine.allocate_covariate(m::VUMod) = VUCovariate(zeros(m.m))

@simple_parameter VU a e s

function Kirstine.jacobianmatrix!(jm, m::VUMod, c::VUCovariate, p::VUParameter)
    for j in 1:length(c.time)
        A = exp(-p.a * c.time[j])
        E = exp(-p.e * c.time[j])
        jm[j, 1] = A * p.s * c.time[j]
        jm[j, 2] = -E * p.s * c.time[j]
        jm[j, 3] = E - A
    end
    return jm
end

struct EquiTime <: CovariateParameterization end

function Kirstine.map_to_covariate!(c::VUCovariate, dp, m::VUMod, cp::EquiTime)
    for j in 1:(m.m)
        c.time[j] = (j - 1) * dp[1]
    end
    return c
end
