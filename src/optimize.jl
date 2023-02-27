# interface for abstract optimizer implementation

function optimize(
    optimizer::Optimizer,
    f, # x<:AbstractPoint -> Float64
    candidates::AbstractVector{<:AbstractPoint},
    constraints;
    trace_state = false,
)
    state = optimizer_state(f, optimizer, candidates, constraints)
    t_x = [deepcopy(maximizer(state))]
    t_fx = [maximum(state)]
    t_state = [deepcopy(state)]
    for i in 1:iterations(optimizer)
        tick!(f, state, optimizer, constraints)
        push!(t_x, deepcopy(maximizer(state)))
        push!(t_fx, maximum(state))
        if trace_state
            push!(t_state, deepcopy(state))
        end
    end
    return OptimizationResult(maximizer(state), maximum(state), t_x, t_fx, t_state)
end
