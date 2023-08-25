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
    wm = WorkMatrices(length(weights(d)), unit_length(m), parameter_dimension(pk), t)
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
