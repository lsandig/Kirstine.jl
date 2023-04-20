# interface for abstract optimizer implementation

function optimize(
    optimizer::Optimizer,
    f, # x<:AbstractPoint -> Float64
    candidates::AbstractVector{<:AbstractPoint},
    constraints;
    trace_state = false,
)
    time_start = time()
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
    time_end = time()
    twall = time_end - time_start
    x = maximizer(state)
    fx = maximum(state)
    return OptimizationResult(x, fx, t_x, t_fx, t_state, n_eval(state), twall)
end
