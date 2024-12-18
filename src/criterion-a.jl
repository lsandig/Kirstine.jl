# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## A-optimal design ##

# Constants for the Gateaux derivative.
#
# The general form is ∫(tr[A(ζ,θ)M(δ,θ)] - tr[B(ζ,θ)])p(θ)dθ.
#
# For the Identity trafo we precompute
#
#       A(ζ,θ) = M(ζ,θ)^{-2}
#   tr[B(ζ,θ)] = tr[M(ζ,θ)^{-1}].
#
# For the DeltaMethod trafo we precompute
#
#       A(ζ,θ) = M(ζ,θ)^{-1} DT'(θ) DT(θ) M(ζ,θ)^{-1}
#   tr[B(ζ,θ)] = tr[DT'(θ) DT(θ) M(ζ,θ)^{-1}].

"""
    ACriterion <: DesignCriterion

Criterion for A-optimal experimental design.

Trace of the inverted normalized information matrix.

See also the [mathematical background](math.md#A-Criterion).
"""
struct ACriterion <: DesignCriterion end

function criterion_functional!(tnim::AbstractMatrix, is_inv::Bool, dc::ACriterion)
    if is_inv
        # Note: In this branch there won't be an exception if `tnim` is singular. But as this
        # branch is called with the DeltaMethod transformation, an exception will already
        # have been thrown in `apply_transformation!`.
        return -tr(tnim)
    else
        potrf!('U', tnim) # Cholesky factor
        potri!('U', tnim) # inverse thereof
        # Note that an exception thrown by `potri!` is caught in `objective!`, which then
        # returns -Inf. This makes sense since `tr(tnim)` is the sum of the reciprocals of
        # `tnim`'s eigenvalues, which for singular `tnim` contain at least one zero.
        return -tr(tnim)
    end
end

function gateaux_constants!(
    gc::GCPriorSample,
    mw::ModelWorkspace,
    nw::NIMWorkspace,
    c::AbstractVector{<:Covariate},
    dc::ACriterion,
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
        # check for numerical singularity, see DCriterion/Identity method
        inv(mnorm * invmnorm) < eps(Float64) && throw(SingularException(-1))
        gc.tr_B[i] = tr(nw.r_x_r)
        # mul!() needs dense matrices, symm!() would need a dense B
        mirror_to_lower_triangle!(nw.r_x_r)
        mul!(gc.A[i], nw.r_x_r, nw.r_x_r)
    end
    return gc
end

function gateaux_constants!(
    gc::GCPriorSample,
    mw::ModelWorkspace,
    nw::NIMWorkspace,
    c::AbstractVector{<:Covariate},
    dc::ACriterion,
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
        potrf!('U', nw.r_x_r)
        potri!('U', nw.r_x_r)
        invmnorm = opnorm1_sym_u(nw.r_x_r)
        # check for numerical singularity, see DCriterion/Identity method
        inv(mnorm * invmnorm) < eps(Float64) && throw(SingularException(-1))
        # nw.t_x_r := DT * inv(M)
        symm!('R', 'U', 1.0, nw.r_x_r, tc.jm[i], 0.0, nw.t_x_r)
        # gc.A[i] := (DT * inv(M))' * DT * inv(M)
        gemm!('T', 'N', 1.0, nw.t_x_r, nw.t_x_r, 0.0, gc.A[i])
        # nw.t_x_t := DT * inv(M) * DT'
        gemm!('N', 'T', 1.0, nw.t_x_r, tc.jm[i], 0.0, nw.t_x_t)
        gc.tr_B[i] = tr(nw.t_x_t)
    end
    return gc
end
