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

function check_objective(state::OptimizerState; only_warn::Bool, prefix = "")
    msg = ""
    if isnan(maximum(state))
        msg = prefix * "maximum is NaN"
    elseif isinf(maximum(state))
        msg = prefix * "maximum is not finite: $(maximum(state))"
    end
    x = maximizer(state)
    !isempty(msg) && (only_warn ? @warn(msg, x) : error(msg * ", x = " * string(x)))
    return nothing
end

function check_gradient(state::OptimizerState; only_warn::Bool, prefix = "")
    msg = ""
    if any(isnan, gradient(state))
        msg = prefix * "gradient contains NaN: $(gradient(state))"
    elseif any(isinf, gradient(state))
        msg = prefix * "gradient contains infinities: $(gradient(state))"
    end
    x = maximizer(state)
    !isempty(msg) && (only_warn ? @warn(msg, x) : error(msg * ", x = " * string(x)))
    return nothing
end

# The first argument is either a function AbstractPoint -> Float64
# or a tuple with a function AbstractPoint -> Float64
# and its mutating gradient (Vector{Float64}, AbstractPoint) -> Vector{Float64}
function optimize(
    f_or_f_g!,
    optimizer::Optimizer,
    prototypes::AbstractVector{<:AbstractPoint},
    constraints::AbstractConstraints;
    trace_state = false,
)
    time_start = time()
    assert_constraints(prototypes, constraints)
    state = optimizer_state(f_or_f_g!, optimizer, prototypes, constraints)
    check_objective(state; only_warn = false, prefix = "iteration 1: ")
    t_x = [deepcopy(maximizer(state))]
    t_fx = [maximum(state)]
    t_gx = Vector{Float64}[]
    is_first_order = optimizer isa FirstOrderOptimizer
    if is_first_order
        check_gradient(state; only_warn = false, prefix = "iteration 1: ")
        push!(t_gx, gradient(state))
    end
    t_state = typeof(state)[]
    trace_state && push!(t_state, deepcopy(state))
    for i in 2:iterations(optimizer)
        tick!(f_or_f_g!, state, optimizer, constraints, i)
        push!(t_x, deepcopy(maximizer(state)))
        push!(t_fx, maximum(state))
        is_first_order && push!(t_gx, gradient(state))
        trace_state && push!(t_state, deepcopy(state))
    end
    check_objective(state; only_warn = true, prefix = "final iteration: ")
    is_first_order && check_gradient(state; only_warn = true, prefix = "final iteration: ")
    time_end = time()
    twall = time_end - time_start
    x = maximizer(state)
    fx = maximum(state)
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
