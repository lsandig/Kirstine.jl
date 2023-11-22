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
    trafo::Identity,
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
    trafo::DeltaMethod,
    na::NormalApproximation,
)
    # This computes Symmetric versions of of M(ζ,θ)^{-1}.
    iim = [inv(informationmatrix(d, m, cp, p, na)) for p in pk.p]
    tc = trafo_constants(trafo, pk)
    nw = NIMWorkspace(parameter_dimension(pk), codomain_dimension(tc))

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
    tr_B = fill(codomain_dimension(tc), length(pk.p))
    return GCDDeltaMethod(A, tr_B)
end
