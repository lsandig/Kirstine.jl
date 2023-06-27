# D-optimal design

function criterion_integrand!(tnim::AbstractMatrix, is_inv::Bool, dc::DOptimality)
    sgn = is_inv ? -1 : 1
    ld = log_det!(tnim)
    # With the DeltaMethod, `tnim` can be singular without having raised an exception up to
    # now. Similarly to how we handle the PosDefException in `objective!`, we
    # unconditionally return -Inf.
    cv = ld != -Inf ? sgn * ld : ld
    return cv
end

# == Gateaux derivative for identity transformation == #
function gateaux_integrand(c::GCDIdentity, nim_direction, index)
    return tr_prod(c.invM[index], nim_direction, :U) - c.parameter_length
end

function precalculate_gateaux_constants(dc::DOptimality, d, m, cp, pk, tc::TCIdentity, na)
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
function precalculate_gateaux_constants(dc::DOptimality, d, m, cp, pk::DiscretePrior, tc::TCDeltaMethod, na)
#! format: on
    invM = inverse_information_matrices(d, m, cp, pk, na)
    r = parameter_dimension(pk)
    t = codomain_dimension(tc)
    invM_copy = deepcopy(invM[1])
    work = zeros(r, t)
    invM_B_invM = map(1:length(pk.p)) do i
        invM_copy .= invM[i] # Note: will be modified
        inv_tnim, _ = apply_transformation!(zeros(t, t), work, invM_copy, true, tc, i)
        J = tc.jm[i]
        B = J' * (Symmetric(inv_tnim) \ J)
        sym_invM = Symmetric(invM[i])
        return sym_invM * B * sym_invM
    end
    return GCDDeltaMethod(invM_B_invM, t)
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
    return efficiency(d1, d2, m, m, cp, cp, pk, trafo, na, na)
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
               na1::NormalApproximation,
               na2::NormalApproximation)

Relative D-efficiency of `d1` to `d2` under prior knowledge `pk`.

Note that the models, covariate parameterizations or normal approximations need not be
identical.
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
    na1::NormalApproximation,
    na2::NormalApproximation,
)
    tc = precalculate_trafo_constants(trafo, pk)
    pardim = parameter_dimension(pk)
    tpardim = codomain_dimension(tc)
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
                       m1, m2, c1, c2, pk, tc, na1, na2)
    #! format: on
    return exp(ei / tpardim)
end

#! format: off
function eff_integral!(tnim1, tnim2, work, nim1, nim2, jm1, jm2, w1, w2,
                       m1, m2, c1, c2, pk::DiscretePrior, tc, na1, na2)
#! format: on
    n = length(pk.p)
    acc = 0.0
    for i in 1:n
        informationmatrix!(nim1, jm1, w1, m1, invcov(m1), c1, pk.p[i], na1)
        informationmatrix!(nim2, jm2, w2, m2, invcov(m2), c2, pk.p[i], na2)
        _, is_inv1 = apply_transformation!(tnim1, work, nim1, false, tc, i)
        _, is_inv2 = apply_transformation!(tnim2, work, nim2, false, tc, i)
        acc += pk.weight[i] * eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
    end
    return acc
end

function eff_integrand!(tnim1, tnim2, is_inv1, is_inv2)
    sgn1 = is_inv1 ? -1 : 1
    sgn2 = is_inv2 ? -1 : 1
    log_num = sgn1 * log_det!(tnim1)
    log_den = sgn2 * log_det!(tnim2)
    return log_num - log_den
end
