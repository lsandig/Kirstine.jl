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

function criterion_integrand!(tnim::AbstractMatrix, is_inv::Bool, dc::ACriterion)
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

function gateaux_constants(
    dc::ACriterion,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    trafo::Identity,
    na::NormalApproximation,
)
    invM = [inv(informationmatrix(d, m, cp, p, na)) for p in pk.p]
    tr_B = map(tr, invM)
    A = map(m -> Symmetric(m)^2, invM)
    return GCPriorSample(A, tr_B)
end

function gateaux_constants(
    dc::ACriterion,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    trafo::DeltaMethod,
    na::NormalApproximation,
)
    tc = trafo_constants(trafo, pk)
    invM = [inv(informationmatrix(d, m, cp, p, na)) for p in pk.p]
    JpJ = map(J -> J' * J, tc.jm)
    tr_B = map((J, iM) -> tr(J * Symmetric(iM)), JpJ, invM)
    A = map((J, iM) -> Symmetric(iM) * J * Symmetric(iM), JpJ, invM)
    return GCPriorSample(A, tr_B)
end
