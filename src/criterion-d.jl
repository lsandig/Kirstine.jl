# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## D-optimal design and relative D-efficiency ##

struct GCDIdentity <: GateauxConstants
    invM::Vector{Matrix{Float64}}
    parameter_length::Int64
end

struct GCDDeltaMethod <: GateauxConstants
    invM_B_invM::Vector{Matrix{Float64}}
    transformed_parameter_length::Int64
end

@doc raw"""
    DOptimality

Criterion for D-optimal experimental design.

Log-determinant of the normalized information matrix.

See also the [mathematical background](math.md#D-Criterion).
"""
struct DOptimality <: DesignCriterion end

function criterion_integrand!(tnim::AbstractMatrix, is_inv::Bool, dc::DOptimality)
    sgn = is_inv ? -1 : 1
    ld = log_det!(tnim)
    # With the DeltaMethod, `tnim` can be singular without having raised an exception up to
    # now. Similarly to how we handle the PosDefException in `objective!`, we
    # unconditionally return -Inf.
    cv = ld != -Inf ? sgn * ld : ld
    return cv
end

## Gateaux derivative: Identity ##

function gateaux_integrand(c::GCDIdentity, nim_direction, index)
    return tr_prod(c.invM[index], nim_direction, :U) - c.parameter_length
end

function precalculate_gateaux_constants(dc::DOptimality, d, m, cp, pk, tc::TCIdentity, na)
    invM = inverse_information_matrices(d, m, cp, pk, na)
    parameter_length = parameter_dimension(pk)
    return GCDIdentity(invM, parameter_length)
end

## Gateaux derivative: DeltaMethod ##

function gateaux_integrand(c::GCDDeltaMethod, nim_direction, index)
    # Note: invM_B_invM[index] is dense and symmetric, we can use the upper triangle
    return tr_prod(c.invM_B_invM[index], nim_direction, :U) - c.transformed_parameter_length
end

#! format: off
function precalculate_gateaux_constants(dc::DOptimality, d, m, cp, pk::PriorSample, tc::TCDeltaMethod, na)
#! format: on
    invM = inverse_information_matrices(d, m, cp, pk, na)
    t = codomain_dimension(tc)
    wm = WorkMatrices(unit_length(m), parameter_dimension(pk), t)
    invM_B_invM = map(1:length(pk.p)) do i
        wm.r_x_r .= invM[i]
        inv_tnim, _ = apply_transformation!(wm, true, tc, i)
        J = tc.jm[i]
        B = J' * (Symmetric(inv_tnim) \ J)
        sym_invM = Symmetric(invM[i])
        return sym_invM * B * sym_invM
    end
    return GCDDeltaMethod(invM_B_invM, t)
end

## relative D-efficiency ##

"""
    efficiency(d1::DesignMeasure,
               d2::DesignMeasure,
               m::NonlinearRegression,
               cp::CovariateParameterization,
               pk::PriorKnowledge,
               trafo::Transformation,
               na::NormalApproximation)

Relative D-efficiency of `d1` to `d2`.
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
               pk::PriorSample,
               trafo::Transformation,
               na1::NormalApproximation,
               na2::NormalApproximation)

Relative D-efficiency of `d1` to `d2`.

Note that the models, covariate parameterizations or normal approximations need not be
identical.

See also the [mathematical background](math.md#Efficiency).
"""
function efficiency(
    d1::DesignMeasure,
    d2::DesignMeasure,
    m1::NonlinearRegression,
    m2::NonlinearRegression,
    cp1::CovariateParameterization,
    cp2::CovariateParameterization,
    pk::PriorSample,
    trafo::Transformation,
    na1::NormalApproximation,
    na2::NormalApproximation,
)
    tc = precalculate_trafo_constants(trafo, pk)
    r = parameter_dimension(pk)
    t = codomain_dimension(tc)
    wm1 = WorkMatrices(unit_length(m1), r, t)
    wm2 = WorkMatrices(unit_length(m2), r, t)
    c1 = allocate_initialize_covariates(d1, m1, cp1)
    c2 = allocate_initialize_covariates(d2, m2, cp2)
    ic1 = invcov(m1)
    ic2 = invcov(m2)
    ei = 0.0 # efficiency integral
    for i in 1:length(pk.p)
        informationmatrix!(wm1.r_x_r, wm1.m_x_r, d1.weight, m1, ic1, c1, pk.p[i], na1)
        informationmatrix!(wm2.r_x_r, wm2.m_x_r, d2.weight, m2, ic2, c2, pk.p[i], na2)
        log_num = try
            _, is_inv1 = apply_transformation!(wm1, false, tc, i)
            (is_inv1 ? -1 : 1) * log_det!(wm1.t_x_t)
        catch e
            # A PosDefException means we found out the hard way that wm1.t_x_t is not
            # invertible. So we can assume is_inv1 would have been set to true.
            if isa(e, PosDefException)
                -Inf
            else
                rethrow(e)
            end
        end
        log_den = try
            _, is_inv2 = apply_transformation!(wm2, false, tc, i)
            (is_inv2 ? -1 : 1) * log_det!(wm2.t_x_t)
        catch e
            if isa(e, PosDefException)
                -Inf
            else
                rethrow(e)
            end
        end
        ei += pk.weight[i] * (log_num - log_den)
    end
    return exp(ei / t)
end
