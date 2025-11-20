# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## D-optimal design ##

# Constants for the Gateaux derivative.
#
# The general form is ∫(tr[A(ζ,θ)M(δ,θ)] - tr[B(ζ,θ)])p(θ)dθ.
#
# For the Identity trafo we precompute
#
#       A(ζ,θ) = M(ζ,θ)^{-1}
#   tr[B(ζ,θ)] = r
#
# For the DeltaMethod trafo we precompute
#
#       A(ζ,θ) = M(ζ,θ)^{-1} DT'(θ) Q(ζ,θ) DT(θ) M(ζ,θ)^{-1}
#   tr[B(ζ,θ)] = t

@doc raw"""
    DCriterion <: Kirstine.DesignCriterion

Criterion for D-optimal experimental design.

Log-determinant of the normalized information matrix.

See also the [mathematical background](math.md#D-Criterion).
"""
struct DCriterion <: DesignCriterion end

function criterion_functional!(
    Q_or_inv_Q::AbstractMatrix{<:Real},
    is_inv::Bool,
    dc::DCriterion,
)
    sgn = is_inv ? -1 : 1
    ld = log_det!(Q_or_inv_Q)
    # With the DeltaMethod, `Q_or_inv_Q` can be singular without having raised an exception up to
    # now. Similarly to how we handle the PosDefException in `objective!`, we
    # unconditionally return -Inf.
    cv = ld != -Inf ? sgn * ld : ld
    return cv
end

## Gateaux derivative: Identity ##

function gateaux_constants!(
    gc::GCPriorSample,
    mw::ModelWorkspace,
    nw::NIMWorkspace,
    c::AbstractVector{<:Covariate},
    dc::DCriterion,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    tc::TCIdentity,
    na::NormalApproximation,
)
    ps = parameters(pk)
    update_covariates!(c, d, m, cp)
    update_model_workspace!(mw, m, c)
    for i in 1:length(ps)
        average_fishermatrix!(nw.r_x_r, mw, weights(d), m, c, ps[i])
        informationmatrix!(nw.r_x_r, na)
        mnorm = opnorm1_sym_u(nw.r_x_r)
        potrf!('U', nw.r_x_r)
        potri!('U', nw.r_x_r)
        invmnorm = opnorm1_sym_u(nw.r_x_r)
        # Test via reciprocal 1-norm condition number whether we consider M to be
        # numerically singular (same criterion that solve() uses in R).
        # Computing it as product of operator norms is from LinerAlgebra.cond(),
        # the -1 is a dummy value, we don't actually know the singular value's index.
        inv(mnorm * invmnorm) < eps(Float64) && throw(SingularException(-1))
        gc.A[i] .= nw.r_x_r # only upper triangle
        gc.tr_B[i] = parameter_dimension(pk)
    end
    return gc
end

## Gateaux derivative: DeltaMethod ##

function gateaux_constants!(
    gc::GCPriorSample,
    mw::ModelWorkspace,
    nw::NIMWorkspace,
    c::AbstractVector{<:Covariate},
    dc::DCriterion,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    tc::TCDeltaMethod,
    na::NormalApproximation,
)
    ps = parameters(pk)
    update_covariates!(c, d, m, cp)
    update_model_workspace!(mw, m, c)
    for i in 1:length(ps)
        average_fishermatrix!(nw.r_x_r, mw, weights(d), m, c, ps[i])
        informationmatrix!(nw.r_x_r, na)
        mnorm = opnorm1_sym_u(nw.r_x_r)
        gc.A[i] .= nw.r_x_r
        potrf!('U', gc.A[i])
        potri!('U', gc.A[i])
        invmnorm = opnorm1_sym_u(gc.A[i])
        # check for numerical singularity, see Identity method
        inv(mnorm * invmnorm) < eps(Float64) && throw(SingularException(-1))
        gc.A[i] .= nw.r_x_r # store it temporarily here, apply_transformation!() will overwrite it
        # nw.t_x_t := inv(Q)
        nw.r_is_inv = false
        _apply_transformation_delta!(nw, tc.jm[i])
        # nw.r_x_t := inv(M) * DT'
        nw.r_x_t .= tc.jm[i]'
        posv!('U', gc.A[i], nw.r_x_t)
        # nw.t_x_r := DT * inv(M)
        nw.t_x_r .= nw.r_x_t'
        # nw.t_x_r := Q * DT * inv(M)
        posv!('U', nw.t_x_t, nw.t_x_r)
        # nw.r_x_r := inv(M) * DT' * Q * DT * inv(M)
        gemm!('N', 'N', 1.0, nw.r_x_t, nw.t_x_r, 0.0, gc.A[i]) # dense
        gc.tr_B[i] = size(nw.t_x_t, 1)
    end
    return gc
end

# For a problem with the D-criterion, we can estimate a lower bound for the efficiency with
# respect to the (unknown) optimal design from the value of the maximal Gateaux derivative
# at the current candidate design.
function efficiency_lower_bound(
    max_gateaux_derivative::Real,
    dim_transformed_parameter::Integer,
)
    err_unbounded = ArgumentError("gateaux derivative must be bounded from below")
    max_gateaux_derivative > -Inf || throw(err_unbounded)
    err_dimpar = ArgumentError("dimension of transformed parameter must be >= 1")
    dim_transformed_parameter >= 1 || throw(err_dimpar)
    return exp(-max_gateaux_derivative / dim_transformed_parameter)
end
