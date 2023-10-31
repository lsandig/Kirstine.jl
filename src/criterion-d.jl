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

# Although tr_B is constant in both cases we store it redundantly in a vector
# in order to keep the structure consistent with the GCA* types.
struct GCDIdentity <: GateauxConstants
    A::Vector{Matrix{Float64}}
    tr_B::Vector{Float64}
end

struct GCDDeltaMethod <: GateauxConstants
    A::Vector{Matrix{Float64}}
    tr_B::Vector{Float64}
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

# Note: only the upper triangle of the symmetric matrix A needs to be filled out,
# since `gateaux_integrand` uses `tr_prod` for the multiplication.
# But producing a dense matrix does not hurt either.
function gateaux_integrand(c::GCDIdentity, nim_direction, index)
    return tr_prod(c.A[index], nim_direction, :U) - c.tr_B[index]
end

function gateaux_constants(
    dc::DOptimality,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    tc::TCIdentity,
    na::NormalApproximation,
)
    A = inverse_information_matrices(d, m, cp, pk, na) # only upper triangles
    tr_B = fill(parameter_dimension(pk), length(pk.p))
    return GCDIdentity(A, tr_B)
end

## Gateaux derivative: DeltaMethod ##

function gateaux_integrand(c::GCDDeltaMethod, nim_direction, index)
    return tr_prod(c.A[index], nim_direction, :U) - c.tr_B[index]
end

function gateaux_constants(
    dc::DOptimality,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    tc::TCDeltaMethod,
    na::NormalApproximation,
)
    t = codomain_dimension(tc)
    # This computes the upper triangle of M(ζ,θ)^{-1}.
    inv_M = inverse_information_matrices(d, m, cp, pk, na)
    # We already know that the result will be inverted.
    inv_MT, _ = transformed_information_matrices(inv_M, true, pk, tc)
    # Note that A will be dense.
    A = map(inv_M, inv_MT, tc.jm) do iM, iMT, DT
        # Now, iMT contains the upper triangle of (M_T(ζ,θ))^{-1}.
        # Instead of an explicit inversion, we solve (M_T(ζ,θ))^{-1} X = DT for X.
        C = DT' * (Symmetric(iMT) \ DT)
        return Symmetric(iM) * C * Symmetric(iM)
    end
    tr_B = fill(t, length(pk.p))
    return GCDDeltaMethod(A, tr_B)
end
