# D-optimal design

function criterion_integrand!(tnim::AbstractMatrix, is_inv::Bool, dc::DOptimality)
    # TODO: try to catch the SingularException further up and print the offending design!
    sgn = is_inv ? -1 : 1
    return sgn * log_det!(tnim)
end

# Note: We must specialize on trafo here to exploit the special structure for Identity()
function gateaux_integrand(dc::DOptimality, inv_nim_at, _, nim_direction, trafo::Identity)
    # Note: the dummy argument (inv_nim_at_mul_B) will always be the identity matrix for trafo::Identity
    return tr_prod(inv_nim_at, nim_direction, :U) - size(inv_nim_at, 1)
end

function gateaux_integrand(
    dc::DOptimality,
    inv_nim_at,
    inv_nim_at_mul_B,
    nim_direction,
    trafo::DeltaMethod,
)
    X = inv_nim_at_mul_B # dense and _not_ symmetric
    # Note: A benchmark suggests that the Symmetric() views do not incur dynamic memory
    # allocations. Writing out the index swapping by hand would make the for loops much less
    # readable.
    Y = Symmetric(inv_nim_at)
    Z = Symmetric(nim_direction)
    # explcitly calculate tr[XYZ] = sum_{i,k,j} X_{ij} Y_{jk} Z_{ki}
    tr_XYZ = 0.0
    r = size(inv_nim_at_mul_B, 1)
    for i in 1:r
        for j in 1:r
            for k in 1:r
                tr_XYZ += X[i, j] * Y[j, k] * Z[k, i]
            end
        end
    end
    return tr_XYZ - tr(inv_nim_at_mul_B)
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

function calc_inv_nim_at_mul_B(
    dc::DOptimality,
    pk::PriorGuess,
    trafo::DeltaMethod,
    inv_nim_at,
)
    t = codomain_dimension(trafo, pk)
    ina_copy = deepcopy(inv_nim_at[1])
    r = parameter_dimension(pk)
    work = zeros(r, t)
    inv_tnim, _ = apply_transformation!(zeros(t, t), work, ina_copy, true, trafo, 1)
    J = trafo.tjm[1]
    B = J' * (Symmetric(inv_tnim) \ J)
    return [Symmetric(inv_nim_at[1]) * B]
end

function calc_inv_nim_at_mul_B(
    dc::DOptimality,
    pk::PriorSample,
    trafo::DeltaMethod,
    inv_nim_at,
)
    t = codomain_dimension(trafo, pk)
    ina_copy = deepcopy(inv_nim_at[1])
    r = parameter_dimension(pk)
    work = zeros(r, t)
    res = map(1:length(pk.p)) do i
        ina_copy .= inv_nim_at[i] # Note: this matrix gets overwritten
        inv_tnim, _ = apply_transformation!(zeros(t, t), work, ina_copy, true, trafo, i)
        J = trafo.tjm[i]
        B = J' * (Symmetric(inv_tnim) \ J)
        return Symmetric(inv_nim_at[i]) * B
    end
    return res
end

function calc_inv_nim_at_mul_B(
    dc::DOptimality,
    pk::DiscretePrior,
    trafo::DeltaMethod,
    inv_nim_at,
)
    t = codomain_dimension(trafo, pk)
    ina_copy = deepcopy(inv_nim_at[1])
    r = parameter_dimension(pk)
    work = zeros(r, t)
    res = map(1:length(pk.p)) do i
        ina_copy .= inv_nim_at[i]
        inv_tnim, _ = apply_transformation!(zeros(t, t), work, ina_copy, true, trafo, i)
        J = trafo.tjm[i]
        B = J' * (Symmetric(inv_tnim) \ J)
        return Symmetric(inv_nim_at[i]) * B
    end
    return res
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
