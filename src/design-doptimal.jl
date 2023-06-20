# D-optimal design

function criterion_integrand!(tnim::AbstractMatrix, is_inv::Bool, dc::DOptimality)
    # TODO: try to catch the SingularException further up and print the offending design!
    sgn = is_inv ? -1 : 1
    return sgn * log_det!(tnim)
end

# == Gateaux derivative for identity transformation == #
function gateaux_integrand(c::GCDIdentity, nim_direction, index)
    return tr_prod(c.invM[index], nim_direction, :U) - c.parameter_length
end

function precalculate_gateaux_constants(dc::DOptimality, d, m, cp, pk, trafo::Identity, na)
    invM = inverse_information_matrices(d, m, cp, pk, na)
    parameter_length = parameter_dimension(pk)
    return GCDIdentity(invM, parameter_length)
end

# == Gateaux derivative for delta method transformation == #
function gateaux_integrand(c::GCDDeltaMethod, nim_direction, index)
    # Note: invM_B_invM[index] is dense and symmetric, we can use the upper triangle
    return tr_prod(c.invM_B_invM[index], nim_direction, :U) - c.transformed_parameter_length
end

#! format: off
function precalculate_gateaux_constants(dc::DOptimality, d, m, cp, pk::DiscretePrior, trafo::DeltaMethod, na)
#! format: on
    invM = inverse_information_matrices(d, m, cp, pk, na)
    r = parameter_dimension(pk)
    t = codomain_dimension(trafo, pk)
    invM_copy = deepcopy(invM[1])
    work = zeros(r, t)
    invM_B_invM = map(1:length(pk.p)) do i
        invM_copy .= invM[i] # Note: will be modified
        inv_tnim, _ =
            apply_transformation!(zeros(t, t), work, invM_copy, true, trafo, i)
        J = trafo.tjm[i]
        B = J' * (Symmetric(inv_tnim) \ J)
        sym_invM = Symmetric(invM[i])
        return sym_invM * B * sym_invM
    end
    return GCDDeltaMethod(invM_B_invM, t)
end

# Note: precalculating constants is not that time critical, so we can get away with this wrapping strategy.
#! format: off
function precalculate_gateaux_constants(dc::DOptimality, d, m, cp, pk::PriorSample, trafo::DeltaMethod, na)
#! format: on
    pk_wrap = DiscretePrior(fill(1 / length(pk.p), length(pk.p)), pk.p)
    return precalculate_gateaux_constants(dc, d, m, cp, pk_wrap, trafo, na)
end

#! format: off
function precalculate_gateaux_constants(dc::DOptimality, d, m, cp, pk::PriorGuess, trafo::DeltaMethod, na)
#! format: on
    pk_wrap = DiscretePrior([1.0], [pk.p])
    return precalculate_gateaux_constants(dc, d, m, cp, pk_wrap, trafo, na)
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
    work = zeros(pardim, tpardim)
    tnim1 = zeros(tpardim, tpardim)
    nim1 = zeros(pardim, pardim)
    jm1 = zeros(unit_length(m1), pardim)
    c1 = allocate_initialize_covariates(d1, m1, cp1)
    tnim2 = zeros(tpardim, tpardim)
    nim2 = zeros(pardim, pardim)
    jm2 = zeros(unit_length(m2), pardim)
    c2 = allocate_initialize_covariates(d2, m2, cp2)

    #! format: off
    ei = eff_integral!(tnim1, tnim2, work, nim1, nim2, jm1, jm2, d1.weight, d2.weight,
                       m1, m2, c1, c2, pk, trafo, na)
    #! format: on
    return exp(ei / tpardim)
end

#! format: off
function eff_integral!(tnim1, tnim2, work, nim1, nim2, jm1, jm2, w1, w2,
                       m1, m2, c1, c2, pk::DiscretePrior, trafo, na)
#! format: on
    n = length(pk.p)
    acc = 0.0
    for i in 1:n
        informationmatrix!(nim1, jm1, w1, m1, invcov(m1), c1, pk.p[i], na)
        informationmatrix!(nim2, jm2, w2, m2, invcov(m2), c2, pk.p[i], na)
        _, is_inv1 = apply_transformation!(tnim1, work, nim1, false, trafo, i)
        _, is_inv2 = apply_transformation!(tnim2, work, nim2, false, trafo, i)
        acc += pk.weight[i] * eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
    end
    return acc
end

#! format: off
function eff_integral!(tnim1, tnim2, work, nim1, nim2, jm1, jm2, w1, w2,
                       m1, m2, c1, c2, pk::PriorSample, trafo, na)
#! format: on
    n = length(pk.p)
    acc = 0.0
    for i in 1:n
        informationmatrix!(nim1, jm1, w1, m1, invcov(m1), c1, pk.p[i], na)
        informationmatrix!(nim2, jm2, w2, m2, invcov(m2), c2, pk.p[i], na)
        _, is_inv1 = apply_transformation!(tnim1, work, nim1, false, trafo, i)
        _, is_inv2 = apply_transformation!(tnim2, work, nim2, false, trafo, i)
        acc += eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
    end
    return acc / n
end

#! format: off
function eff_integral!(tnim1, tnim2, work, nim1, nim2, jm1, jm2, w1, w2,
                       m1, m2, c1, c2, pk::PriorGuess, trafo, na)
#! format: on
    informationmatrix!(nim1, jm1, w1, m1, invcov(m1), c1, pk.p, na)
    informationmatrix!(nim2, jm2, w2, m2, invcov(m2), c2, pk.p, na)
    _, is_inv1 = apply_transformation!(tnim1, work, nim1, false, trafo, 1)
    _, is_inv2 = apply_transformation!(tnim2, work, nim2, false, trafo, 1)
    return eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
end

function eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
    sgn1 = is_inv1 ? -1 : 1
    sgn2 = is_inv2 ? -1 : 1
    log_num = sgn1 * log_det!(tnim1)
    log_den = sgn2 * log_det!(tnim2)
    return log_num - log_den
end
