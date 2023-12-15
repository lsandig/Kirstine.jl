# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
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
#       A(ζ,θ) = M(ζ,θ)^{-1} DT'(θ) M_T(ζ,θ) DT(θ) M(ζ,θ)^{-1}
#   tr[B(ζ,θ)] = t

@doc raw"""
    DCriterion

Criterion for D-optimal experimental design.

Log-determinant of the normalized information matrix.

See also the [mathematical background](math.md#D-Criterion).
"""
struct DCriterion <: DesignCriterion end

function criterion_integrand!(tnim::AbstractMatrix, is_inv::Bool, dc::DCriterion)
    sgn = is_inv ? -1 : 1
    ld = log_det!(tnim)
    # With the DeltaMethod, `tnim` can be singular without having raised an exception up to
    # now. Similarly to how we handle the PosDefException in `objective!`, we
    # unconditionally return -Inf.
    cv = ld != -Inf ? sgn * ld : ld
    return cv
end

## Gateaux derivative: Identity ##

function gateaux_constants(
    dc::DCriterion,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    trafo::Identity,
    na::NormalApproximation,
)
    A = [inv(informationmatrix(d, m, cp, p, na)) for p in pk.p]
    tr_B = fill(parameter_dimension(pk), length(pk.p))
    return GCPriorSample(A, tr_B)
end

## Gateaux derivative: DeltaMethod ##

function gateaux_constants(
    dc::DCriterion,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    trafo::DeltaMethod,
    na::NormalApproximation,
)
    # This computes Symmetric versions of of M(ζ,θ)^{-1}.
    iim = [inv(informationmatrix(d, m, cp, p, na)) for p in pk.p]
    tc = trafo_constants(trafo, pk)
    nw = NIMWorkspace(parameter_dimension(pk), codomain_dimension(trafo, pk))

    # Note that A will be dense.
    A = map(iim, tc.jm) do iM, DT
        # compute (M_T(ζ,θ))^{-1} from M(ζ,θ)^{-1}.
        # since typeof(trafo) == DeltaMethod, we know that the result will be inverted
        nw.r_x_r .= iM
        nw.r_is_inv = true
        apply_transformation!(nw, trafo, DT)
        # Now, nw.t_x_t contains the upper triangle of (M_T(ζ,θ))^{-1}.
        # Instead of an explicit inversion, we solve (M_T(ζ,θ))^{-1} X = DT for X.
        C = DT' * (Symmetric(nw.t_x_t) \ DT)
        return iM * C * iM
    end
    tr_B = fill(codomain_dimension(trafo, pk), length(pk.p))
    return GCPriorSample(A, tr_B)
end
