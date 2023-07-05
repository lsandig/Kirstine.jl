# interface for abstract optimizer implementation

function optimize(
    optimizer::Optimizer,
    f, # x<:AbstractPoint -> Float64
    prototypes::AbstractVector{<:AbstractPoint},
    constraints::AbstractConstraints;
    trace_state = false,
)
    time_start = time()
    state = optimizer_state(f, optimizer, prototypes, constraints)
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

function prettyruntime(t)
    h, r = divrem(t, 3600)
    m, r = divrem(r, 60)
    s, r = divrem(r, 1)
    ms, r = divrem(r, 1e-3)
    string_h = string(Int64(h))
    pad_m = lpad(Int64(m), 2, '0')
    pad_s = lpad(Int64(s), 2, '0')
    pad_ms = lpad(Int64(ms), 3, '0')
    return join([string_h, pad_m, pad_s, pad_ms], ":", ".")
end

function Base.show(
    io::IO,
    ::MIME"text/plain",
    r::OptimizationResult{T,S},
) where {T<:AbstractPoint,S<:OptimizerState{T}}
    println(io, "OptimizationResult")
    println(io, "* iterations: ", length(r.trace_fx))
    println(io, "* evaluations: ", r.n_eval)
    println(io, "* elapsed time: ", prettyruntime(r.seconds_elapsed))
    println(io, "* traced states: ", length(r.trace_state))
    println(io, "* state type: ", S)
    println(io, "* maximum: ", r.maximum)
    print(io, "* maximizer: ")
    show(io, MIME("text/plain"), r.maximizer)
end

function Base.show(
    io::IO,
    r::OptimizationResult{T,S},
) where {T<:AbstractPoint,S<:OptimizerState{T}}
    print(io, "maximum: ", r.maximum)
end
