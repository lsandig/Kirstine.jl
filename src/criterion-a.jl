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

struct GCAIdentity <: GateauxConstants
    A::Vector{Matrix{Float64}}
    tr_B::Vector{Float64}
end

struct GCADeltaMethod <: GateauxConstants
    A::Vector{Matrix{Float64}}
    tr_B::Vector{Float64}
end

"""
    AOptimality <: DesignCriterion

Criterion for A-optimal experimental design.

Trace of the inverted normalized information matrix.

See also the [mathematical background](math.md#A-Criterion).
"""
struct AOptimality <: DesignCriterion end

function criterion_integrand!(tnim::AbstractMatrix, is_inv::Bool, dc::AOptimality)
    if is_inv
        # Note: In this branch there won't be an exception if tnum is singular. But as this
        # branch is called with the DeltaMethod transformation, an exception will already
        # have been thrown in `apply_transformation!`.
        return -tr(tnim)
    else
        potrf!('U', tnim) # cholesky factor
        potri!('U', tnim) # inverse thereof
        # Note that an exception thrown by `potri!` is caught in `objective!`, which then
        # returns -Inf. This makes sense since tr(tnim) is the sum of the reciprocals of
        # tnim's eigenvalues, which for singular tnim contain at least one zero.
        return -tr(tnim)
    end
end

# Note: only the upper triangle of the symmetric matrix A needs to be filled out,
# since `gateaux_integrand` uses `tr_prod` for the multiplication.
# But producing a dense matrix does not hurt either.
function gateaux_integrand(c::GCAIdentity, nim_direction, index)
    return tr_prod(c.A[index], nim_direction, :U) - c.tr_B[index]
end

function precalculate_gateaux_constants(dc::AOptimality, d, m, cp, pk, tc::TCIdentity, na)
    invM = inverse_information_matrices(d, m, cp, pk, na)
    tr_B = map(tr, invM)
    A = map(m -> Symmetric(m)^2, invM)
    return GCAIdentity(A, tr_B)
end

function gateaux_integrand(c::GCADeltaMethod, nim_direction, index)
    return tr_prod(c.A[index], nim_direction, :U) - c.tr_B[index]
end

#! format: off
function precalculate_gateaux_constants(dc::AOptimality, d, m, cp, pk::PriorSample, tc::TCDeltaMethod, na)
    #! format: on
    invM = inverse_information_matrices(d, m, cp, pk, na)
    JpJ = map(J -> J' * J, tc.jm)
    tr_B = map((J, iM) -> tr(J * Symmetric(iM)), JpJ, invM)
    A = map((J, iM) -> Symmetric(iM) * J * Symmetric(iM), JpJ, invM)
    return GCADeltaMethod(A, tr_B)
end
