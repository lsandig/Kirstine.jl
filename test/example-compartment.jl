# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

# Three-parameter compartmental model from
# Atkinson, A. C., Chaloner, K., Herzberg, A. M., & Juritz, J. (1993).
# Optimum experimental designs for properties of a compartmental model.
# Biometrics, 49(2), 325–337. http://dx.doi.org/10.2307/2532547
@define_scalar_unit_model Kirstine TPCMod time

@define_vector_parameter Kirstine TPCPar a e s

struct CopyTime <: CovariateParameterization end

function Kirstine.jacobianmatrix!(jm, m::TPCMod, c::TPCModCovariate, p::TPCPar)
    A = exp(-p.a * c.time)
    E = exp(-p.e * c.time)
    jm[1, 1] = A * p.s * c.time
    jm[1, 2] = -E * p.s * c.time
    jm[1, 3] = E - A
    return m
end

function Kirstine.update_model_covariate!(
    c::TPCModCovariate,
    dp::AbstractVector{<:Real},
    m::TPCMod,
    cp::CopyTime,
)
    c.time = dp[1]
    return c
end

# response, area under the curve, time-to-maximum, maximum concentration
mu(c, p) = p.s * (exp(-p.e * c.time) - exp(-p.a * c.time))
auc(p) = p.s * (1 / p.e - 1 / p.a)
ttm(p) = log(p.a / p.e) / (p.a - p.e)
cmax(p) = mu(TPCModCovariate(ttm(p)), p)

# jacobian matrices (transposed gradients) from Appendix 1
function Dauc(p)
    da = p.s / p.a^2
    de = -p.s / p.e^2
    ds = 1 / p.e - 1 / p.a
    return [da de ds]
end

function Dttm(p)
    A = p.a - p.e
    A2 = A^2
    B = log(p.a / p.e)
    da = (A / p.a - B) / A2
    de = (B - A / p.e) / A2
    ds = 0
    return [da de ds]
end

function Dcmax(p)
    tmax = ttm(p)
    A = exp(-p.a * tmax)
    E = exp(-p.e * tmax)
    F = p.a * A - p.e * E
    da_ttm, de_ttm, ds_ttm = Dttm(p)
    da = p.s * (tmax * A - F * da_ttm)
    de = p.s * (-tmax * E + F * de_ttm)
    ds = E - A
    return [da de ds]
end

function DOmnibus(p)
    # divide by asymptotic standard deviations (cf. Table 1 and p. 332/333)
    return vcat(Dauc(p) ./ sqrt(2194), Dttm(p) ./ sqrt(0.02815), Dcmax(p) ./ sqrt(1.0))
end

function draw_from_prior(n, se_factor)
    # a, e, s
    mn = [4.298, 0.05884, 21.8]
    se = [0.5, 0.005, 0]
    as = mn[1] .+ se_factor .* se[1] .* (2 .* rand(n) .- 1)
    es = mn[2] .+ se_factor .* se[2] .* (2 .* rand(n) .- 1)
    ss = mn[3] .+ se_factor .* se[3] .* (2 .* rand(n) .- 1)
    return PriorSample(map((a, e, s) -> TPCPar(; a = a, e = e, s = s), as, es, ss))
end
