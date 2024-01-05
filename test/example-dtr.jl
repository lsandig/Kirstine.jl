# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

@kwdef struct DTRMod <: NonlinearRegression
    sigma::Float64
    m::Int64
end

mutable struct DoseTimeCovariate <: Covariate
    dose::Float64
    time::Vector{Float64}
end

Kirstine.unit_length(m::DTRMod) = m.m

function Kirstine.update_model_vcov!(s, m::DTRMod, c::DoseTimeCovariate)
    fill!(s, 0.0)
    for j in 1:(m.m)
        s[j, j] = m.sigma^2
    end
end

Kirstine.allocate_covariate(m::DTRMod) = DoseTimeCovariate(0, zeros(m.m))

@simple_parameter DTR a e e0 emax ec50

function Kirstine.jacobianmatrix!(jm, m::DTRMod, c::DoseTimeCovariate, p::DTRParameter)
    for j in 1:length(c.time)
        A = exp(-p.a * c.time[j]) # calculate exponentials only once
        B = exp(-p.e * c.time[j])
        C = B - A
        rd = p.e - p.a # rate difference
        den = c.dose * C * p.a - rd * p.ec50
        den2 = den^2
        # ∂a
        jm[j, 1] = -c.dose * p.ec50 * p.emax * (A * c.time[j] * p.a * rd + p.e * C) / den2
        # ∂e
        jm[j, 2] = c.dose * p.a * p.ec50 * p.emax * (B * c.time[j] * rd + C) / den2
        # ∂e0
        jm[j, 3] = 1.0
        # ∂emax
        jm[j, 4] = c.dose * p.a * C / den
        # ∂ec50
        jm[j, 5] = c.dose * p.a * p.emax * C * rd / den2
    end
    return jm
end

struct CopyBoth <: CovariateParameterization end

function Kirstine.map_to_covariate!(c::DoseTimeCovariate, dp, m::DTRMod, cp::CopyBoth)
    c.dose = dp[1]
    c.time[1] = dp[2]
    return c
end
