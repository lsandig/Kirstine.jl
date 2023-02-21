# D-optimal design

function criterion_integrand!(nim::AbstractMatrix, dc::DOptimality, trafo::Identity)
    return log_det!(nim)
end

function gateaux_integrand(
    nim_x::AbstractMatrix,
    nim_candidate::AbstractMatrix,
    dc::DOptimality,
    trafo::Identity,
)
    return tr_prod(nim_x, nim_candidate) - size(nim_candidate, 1)
end
