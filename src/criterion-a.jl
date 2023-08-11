# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## A-optimal design ##

struct GCAIdentity <: GateauxConstants
    B::Vector{Matrix{Float64}} # inv(M(at))^2
    tr_C::Vector{Float64}      # tr(inv(M(at)))
end

struct GCADeltaMethod <: GateauxConstants
    B::Vector{Matrix{Float64}} # inv(M(at)) * J' * J * inv(M(at))
    tr_C::Vector{Float64}      # tr(J' * J * inv(M(at)))
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

function gateaux_integrand(c::GCAIdentity, nim_direction, index)
    return tr_prod(c.B[index], nim_direction, :U) - c.tr_C[index]
end

function precalculate_gateaux_constants(dc::AOptimality, d, m, cp, pk, tc::TCIdentity, na)
    invM = inverse_information_matrices(d, m, cp, pk, na)
    tr_C = map(tr, invM)
    B = map(m -> Symmetric(m)^2, invM)
    return GCAIdentity(B, tr_C)
end

function gateaux_integrand(c::GCADeltaMethod, nim_direction, index)
    return tr_prod(c.B[index], nim_direction, :U) - c.tr_C[index]
end

#! format: off
function precalculate_gateaux_constants(dc::AOptimality, d, m, cp, pk::PriorSample, tc::TCDeltaMethod, na)
#! format: on
    invM = inverse_information_matrices(d, m, cp, pk, na)
    JpJ = map(J -> J' * J, tc.jm)
    tr_C = map((J, iM) -> tr(J * Symmetric(iM)), JpJ, invM)
    B = map((J, iM) -> Symmetric(iM) * J * Symmetric(iM), JpJ, invM)
    return GCADeltaMethod(B, tr_C)
end
