# D-optimal design

function criterion_integrand!(nim::AbstractMatrix, dc::DOptimality, trafo::Identity)
    return log_det!(nim)
end

function gateaux_integrand(
    inv_nim_at::AbstractMatrix,
    nim_direction::AbstractMatrix,
    dc::DOptimality,
    trafo::Identity,
)
    return tr_prod(nim_direction, inv_nim_at) - size(inv_nim_at, 1)
end

# == relative D-efficiency == #

"""
    efficiency(d1::DesignMeasure,
               d2::DesignMeasure,
               m::NonlinearRegression,
               cp::CovariateParameterization,
               pk::PriorKnowledge,
               trafo::Transformation)

Relative D-efficiency of `d1` to `d2` under prior knowledge `pk`.
"""
function efficiency(
    d1::DesignMeasure,
    d2::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
)
    return efficiency(d1, d2, m, m, cp, cp, pk, trafo)
end

"""
    efficiency(d1::DesignMeasure,
               d2::DesignMeasure,
               m1::NonlinearRegression,
               m2::NonlinearRegression,
               cp1::CovariateParameterization,
               cp2::CovariateParameterization,
               pk::PriorKnowledge,
               trafo::Transformation)

Relative D-efficiency of `d1` to `d2` under prior knowledge `pk`.

Note that the models and/or covariate parameterizations need not be identical.
"""
function efficiency(
    d1::DesignMeasure,
    d2::DesignMeasure,
    m1::NonlinearRegression,
    m2::NonlinearRegression,
    cp1::CovariateParameterization,
    cp2::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
)
    pardim = parameter_dimension(pk)
    nim1 = zeros(pardim, pardim)
    jm1 = zeros(unit_length(m1), pardim)
    c1 = allocate_initialize_covariates(d1, m1, cp1)
    nim2 = zeros(pardim, pardim)
    jm2 = zeros(unit_length(m2), pardim)
    c2 = allocate_initialize_covariates(d2, m2, cp2)

    ei = eff_integral(nim1, nim2, jm1, jm2, c1, c2, d1, d2, m1, m2, cp1, cp2, pk, trafo)
    return exp(ei / pardim)
end

#! format: off
function eff_integral(nim1, nim2, jm1, jm2, c1, c2, d1, d2, m1, m2, cp1, cp2,
                      pk::PriorGuess, trafo::Identity)
#! format: on
    informationmatrix!(nim1, jm1, d1.weight, m1, invcov(m1), c1, pk.p)
    informationmatrix!(nim2, jm2, d2.weight, m2, invcov(m2), c2, pk.p)
    return log_det!(nim1) - log_det!(nim2)
end

#! format: off
function eff_integral(nim1, nim2, jm1, jm2, c1, c2, d1, d2, m1, m2, cp1, cp2,
                      pk::PriorSample, trafo::Identity)
#! format: on
    log_det_diff = 0.0
    for i in 1:length(pk.p)
        informationmatrix!(nim1, jm1, d1.weight, m1, invcov(m1), c1, pk.p[i])
        informationmatrix!(nim2, jm2, d2.weight, m2, invcov(m2), c2, pk.p[i])
        ld1 = log_det!(nim1)
        ld2 = log_det!(nim2)
        log_det_diff += ld1 - ld2
    end
    return log_det_diff
end

#! format: off
function eff_integral(nim1, nim2, jm1, jm2, c1, c2, d1, d2, m1, m2, cp1, cp2,
                      pk::DiscretePrior, trafo::Identity)
#! format: on
    log_det_diff = 0.0
    for i in 1:length(pk.p)
        informationmatrix!(nim1, jm1, d1.weight, m1, invcov(m1), c1, pk.p[i])
        informationmatrix!(nim2, jm2, d2.weight, m2, invcov(m2), c2, pk.p[i])
        ld1 = log_det!(nim1)
        ld2 = log_det!(nim2)
        log_det_diff += pk.weight[i] * (ld1 - ld2)
    end
    return log_det_diff
end
