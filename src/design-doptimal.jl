# D-optimal design

function criterion_integrand!(tnim::AbstractMatrix, is_inv::Bool, dc::DOptimality)
    # TODO: try to catch the SingularException further up and print the offending design!
    sgn = is_inv ? -1 : 1
    return sgn * log_det!(tnim)
end

# Note: If we don't specialize on trafo here, we can't exploit the special structure for
# Identity(), resulting in an 8-fold slowdown.
function gateaux_integrand(dc::DOptimality, inv_nim_at, _, nim_direction, trafo::Identity)
    # Note: the dummy argument (inv_nim_at_mul_B) will always be the identity matrix for trafo::Identity
    return tr_prod(inv_nim_at, nim_direction, :U) - size(inv_nim_at, 1)
end

function calc_inv_nim_at_mul_B(dc::DOptimality, pk::PriorGuess, trafo::Identity, inv_nim_at)
    r = parameter_dimension(pk)
    return [diagm(ones(r))]
end

function calc_inv_nim_at_mul_B(
    dc::DOptimality,
    pk::PriorSample,
    trafo::Identity,
    inv_nim_at,
)
    r = parameter_dimension(pk)
    return [diagm(ones(r)) for _ in 1:length(pk.p)]
end

function calc_inv_nim_at_mul_B(
    dc::DOptimality,
    pk::DiscretePrior,
    trafo::Identity,
    inv_nim_at,
)
    r = parameter_dimension(pk)
    return [diagm(ones(r)) for _ in 1:length(pk.p)]
end

# == relative D-efficiency == #

"""
    efficiency(d1::DesignMeasure,
               d2::DesignMeasure,
               m::NonlinearRegression,
               cp::CovariateParameterization,
               pk::PriorKnowledge,
               trafo::Transformation,
               na::NormalApproximation)

Relative D-efficiency of `d1` to `d2` under prior knowledge `pk`.
"""
function efficiency(
    d1::DesignMeasure,
    d2::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
    na::NormalApproximation,
)
    return efficiency(d1, d2, m, m, cp, cp, pk, trafo, na)
end

"""
    efficiency(d1::DesignMeasure,
               d2::DesignMeasure,
               m1::NonlinearRegression,
               m2::NonlinearRegression,
               cp1::CovariateParameterization,
               cp2::CovariateParameterization,
               pk::PriorKnowledge,
               trafo::Transformation,
               na::NormalApproximation)

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
    na::NormalApproximation,
)
    pardim = parameter_dimension(pk)
    tpardim = codomain_dimension(trafo, pk)
    tnim1 = zeros(tpardim, tpardim)
    nim1 = zeros(pardim, pardim)
    jm1 = zeros(unit_length(m1), pardim)
    c1 = allocate_initialize_covariates(d1, m1, cp1)
    tnim2 = zeros(tpardim, tpardim)
    nim2 = zeros(pardim, pardim)
    jm2 = zeros(unit_length(m2), pardim)
    c2 = allocate_initialize_covariates(d2, m2, cp2)

    #! format: off
    ei = eff_integral!(tnim1, tnim2, nim1, nim2, jm1, jm2, d1.weight, d2.weight,
                       m1, m2, c1, c2, pk, trafo, na)
    #! format: on
    return exp(ei / tpardim)
end

#! format: off
function eff_integral!(tnim1, tnim2, nim1, nim2, jm1, jm2, w1, w2,
                       m1, m2, c1, c2, pk::DiscretePrior, trafo, na)
#! format: on
    n = length(pk.p)
    acc = 0.0
    for i in 1:n
        informationmatrix!(nim1, jm1, w1, m1, invcov(m1), c1, pk.p[i], na)
        informationmatrix!(nim2, jm2, w2, m2, invcov(m2), c2, pk.p[i], na)
        _, is_inv1 = apply_transformation!(tnim1, nim1, false, trafo, i)
        _, is_inv2 = apply_transformation!(tnim2, nim2, false, trafo, i)
        acc += pk.weight[i] * eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
    end
    return acc
end

#! format: off
function eff_integral!(tnim1, tnim2, nim1, nim2, jm1, jm2, w1, w2,
                       m1, m2, c1, c2, pk::PriorSample, trafo, na)
#! format: on
    n = length(pk.p)
    acc = 0.0
    for i in 1:n
        informationmatrix!(nim1, jm1, w1, m1, invcov(m1), c1, pk.p[i], na)
        informationmatrix!(nim2, jm2, w2, m2, invcov(m2), c2, pk.p[i], na)
        _, is_inv1 = apply_transformation!(tnim1, nim1, false, trafo, i)
        _, is_inv2 = apply_transformation!(tnim2, nim2, false, trafo, i)
        acc += eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
    end
    return acc / n
end

#! format: off
function eff_integral!(tnim1, tnim2, nim1, nim2, jm1, jm2, w1, w2,
                       m1, m2, c1, c2, pk::PriorGuess, trafo, na)
#! format: on
    informationmatrix!(nim1, jm1, w1, m1, invcov(m1), c1, pk.p, na)
    informationmatrix!(nim2, jm2, w2, m2, invcov(m2), c2, pk.p, na)
    _, is_inv1 = apply_transformation!(tnim1, nim1, false, trafo, 1)
    _, is_inv2 = apply_transformation!(tnim2, nim2, false, trafo, 1)
    return eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
end

function eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
    sgn1 = is_inv1 ? -1 : 1
    sgn2 = is_inv2 ? -1 : 1
    log_num = sgn1 * log_det!(tnim1)
    log_den = sgn2 * log_det!(tnim2)
    return log_num - log_den
end
