# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## abstract particle based optimization ##

"""
    Kirstine.OptimizationResult

Wrapper for results of particle-based optimization.

# `Kirstine.OptimizationResult` fields

| Field           | Description                                          |
|:--------------- |:---------------------------------------------------- |
| maximizer       | final maximizer                                      |
| maximum         | final objective value                                |
| trace_x         | vector of current maximizer in each iteration        |
| trace_fx        | vector of current objective value in each iteration  |
| trace_gx        | vector of current gradient in each iteration         |
| trace_state     | vector of internal optimizer state in each iteration |
| n\\_eval\\_f    | total number of objective evaluations                |
| n\\_eval\\_g    | total number of gradient evaluations                 |
| seconds_elapsed | total runtime                                        |

Note that for a [`ZerothOrderOptimizer`](@ref) `trace_gx` will be empty
and `n_eval_g == 0`.

See also [`solve`](@ref).
"""
struct OptimizationResult{
    T<:AbstractPoint,
    U<:AbstractPointDifference,
    S<:OptimizerState{T,U},
}
    maximizer::T
    maximum::Float64
    trace_x::Vector{T}
    trace_fx::Vector{Float64}
    trace_gx::Vector{Vector{Float64}}
    trace_state::Vector{S}
    n_eval_f::Int64
    n_eval_g::Int64
    seconds_elapsed::Float64
end

function optimize(
    f, # x<:AbstractPoint -> Float64
    optimizer::ZerothOrderOptimizer,
    prototypes::AbstractVector{<:AbstractPoint},
    constraints::AbstractConstraints;
    trace_state = false,
)
    time_start = time()
    assert_constraints(prototypes, constraints)
    state = optimizer_state(f, optimizer, prototypes, constraints)
    t_x = [deepcopy(maximizer(state))]
    t_fx = [maximum(state)]
    t_state = typeof(state)[]
    if trace_state
        push!(t_state, deepcopy(state))
    end
    for i in 2:iterations(optimizer)
        tick!(f, state, optimizer, constraints, i)
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
    if !isfinite(fx)
        @warn "maximum is not finite" x fx
    end
    t_gx = Vector{Float64}[]
    nf = n_eval_f(state)
    ng = 0
    return OptimizationResult(x, fx, t_x, t_fx, t_gx, t_state, nf, 0, twall)
end

function optimize(
    f, # x<:AbstractPoint -> Float64
    g!, # (g::AbstractVector{<:Real}, x<:AbstractPoint) -> g
    optimizer::FirstOrderOptimizer,
    prototypes::AbstractVector{<:AbstractPoint},
    constraints::AbstractConstraints;
    trace_state = false,
)
    time_start = time()
    assert_constraints(prototypes, constraints)
    state = optimizer_state(f, g!, optimizer, prototypes, constraints)
    t_x = [deepcopy(maximizer(state))]
    t_fx = [maximum(state)]
    t_gx = [deepcopy(gradient(state))]
    t_state = typeof(state)[]
    if trace_state
        push!(t_state, deepcopy(state))
    end
    for i in 2:iterations(optimizer)
        tick!(f, g!, state, optimizer, constraints, i)
        push!(t_x, deepcopy(maximizer(state)))
        push!(t_fx, maximum(state))
        push!(t_gx, gradient(state))
        if trace_state
            push!(t_state, deepcopy(state))
        end
    end
    time_end = time()
    twall = time_end - time_start
    x = maximizer(state)
    fx = maximum(state)
    if !isfinite(fx)
        @warn "maximum is not finite" x fx
    end
    nf = n_eval_f(state)
    ng = n_eval_g(state)
    return OptimizationResult(x, fx, t_x, t_fx, t_gx, t_state, nf, ng, twall)
end

function prettydifftime(t)
    maybe_sign = t >= 0 ? "" : "-"
    t = abs(t)
    h, r = divrem(t, 3600)
    m, r = divrem(r, 60)
    s, r = divrem(r, 1)
    ms, r = divrem(r, 1e-3)
    string_h = string(Int64(h))
    pad_m = lpad(Int64(m), 2, '0')
    pad_s = lpad(Int64(s), 2, '0')
    pad_ms = lpad(Int64(ms), 3, '0')
    return maybe_sign * join([string_h, pad_m, pad_s, pad_ms], ":", ".")
end

function Base.show(
    io::IO,
    ::MIME"text/plain",
    r::OptimizationResult{T,U,S},
) where {T<:AbstractPoint,U<:AbstractPointDifference,S<:OptimizerState{T,U}}
    println(io, "OptimizationResult")
    println(io, "* iterations: ", length(r.trace_fx))
    println(io, "* objective evaluations: ", r.n_eval_f)
    println(io, "* gradient evaluations: ", r.n_eval_g)
    println(io, "* elapsed time: ", prettydifftime(r.seconds_elapsed))
    println(io, "* traced states: ", length(r.trace_state))
    println(io, "* state type: ", S)
    println(io, "* maximum: ", r.maximum)
    print(io, "* maximizer: ")
    show(io, MIME("text/plain"), r.maximizer)
end

function Base.show(
    io::IO,
    r::OptimizationResult{T,U,S},
) where {T<:AbstractPoint,U<:AbstractPointDifference,S<:OptimizerState{T,U}}
    print(io, "maximum: ", r.maximum)
end

# These functions need methods for concrete subtypes of AbstractPoint and
# AbstractPointDifference.
function ap_rand! end
function ap_diff! end
function ap_copy! end
function ap_add! end
function ap_as_difference end
function ap_rand! end
function ap_mul! end
function ap_mul! end
function ap_add! end
function ap_dist end
