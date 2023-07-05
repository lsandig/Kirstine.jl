struct EmaxModel <: NonlinearRegression
    inv_sigma_sq::Float64
end

mutable struct Dose <: Covariate
    dose::Float64
end

@kwdef struct EmaxPar <: Parameter
    e0::Float64
    emax::Float64
    ec50::Float64
end

Kirstine.dimension(p::EmaxPar) = 3

struct CopyDose <: CovariateParameterization end

Kirstine.unit_length(m::EmaxModel) = 1

Kirstine.invcov(m::EmaxModel) = m.inv_sigma_sq

Kirstine.allocate_covariate(m::EmaxModel) = Dose(0)

function Kirstine.update_model_covariate!(
    c::Dose,
    dp::AbstractVector{<:Real},
    m::EmaxModel,
    cp::CopyDose,
)
    c.dose = dp[1]
    return c
end

function Kirstine.jacobianmatrix!(jm, m::EmaxModel, c::Dose, p::EmaxPar)
    x = c.dose
    jm[1, 1] = 1.0
    jm[1, 2] = x / (x + p.ec50)
    jm[1, 3] = -p.emax * x / (x + p.ec50)^2
    return jm
end

function emax_solution(p, ds)
    # Locally D-optimal design for the Emax model has an analytic solution,
    # see Theorem 2 in
    #
    #   Dette, H., Kiss, C., Bevanda, M., & Bretz, F. (2010).
    #   Optimal designs for the emax, log-linear and exponential models.
    #   Biometrika, 97(2), 513–518. http://dx.doi.org/10.1093/biomet/asq020
    #
    a = ds.lowerbound[1]
    b = ds.upperbound[1]
    x_star = (a * (b + p.ec50) + b * (a + p.ec50)) / (a + b + 2 * p.ec50)
    return uniform_design([[a], [x_star], [b]])
end
