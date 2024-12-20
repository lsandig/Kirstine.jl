# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
    StepSizeStrategy

Supertype for Frank-Wolfe step-size strategies.
"""
abstract type StepSizeStrategy end

"""
    OpenLoop <: StepSizeStrategy

Frank-Wolfe step-size strategy ``γ(t) = l / (t + l)`` for ``t=0,1,2,…`` with shift ``l∈ℕ, l≥1``.

!!! note

    With `OpenLoop`, the algorithm will always land on a vertex of the feasible set at ``t=1``.
    For a probability simplex, these vertices are the one-point designs.
    As a consequence, `OpenLoop` is _not_ suitable for the re-weighting step of [`Exchange`](@ref)
    for design problems where one-point designs are singular.
"""
@kwdef struct OpenLoop <: StepSizeStrategy
    shift::Int64
    function OpenLoop(shift::Integer)
        shift < 1 && throw(ArgumentError("shift must be >= 1"))
        return new(shift)
    end
end

"""
    FrankWolfe <: FirstOrderOptimizer

A basic Frank-Wolfe algorithm.

This version is taken from [^P23].

# Initialization

Frank-Wolfe can only be initialized with a single prototype.
This prototype will be used as the starting value.

[^P23]: S. Pokutta (2023). The Frank-Wolfe algorithm: a short introduction. Jahresbericht der Deutschen Mathematiker-Vereinigung, 126(1), 3–35. [doi:10.1365/s13291-023-00275-x](http://dx.doi.org/10.1365/s13291-023-00275-x)
"""
@kwdef struct FrankWolfe{T<:StepSizeStrategy} <: FirstOrderOptimizer
    iterations::Int64
    step::StepSizeStrategy
    @doc """
        FrankWolfe(; iterations)
        FrankWolfe(iterations)

    Set up parameters for the Frank-Wolfe algorithm.

    This type has both a positional constructor and a keyword-based constructor.
    """
    function FrankWolfe(iterations::Integer, s::StepSizeStrategy)
        iterations <= 0 && throw(ArgumentError("number of iterations must be positive"))
        return new{typeof(s)}(iterations, s)
    end
end

iterations(o::FrankWolfe) = o.iterations

mutable struct FrankWolfeState{T,U} <: OptimizerState{T,U}
    x::T
    v::T
    diff::U
    stepsize::Float64
    fx::Float64
    gx::Vector{<:Float64}
    n_eval_f::Int64
    n_eval_g::Int64
end

maximizer(s::FrankWolfeState) = s.x
maximum(s::FrankWolfeState) = s.fx
gradient(s::FrankWolfeState) = s.gx
n_eval_f(s::FrankWolfeState) = s.n_eval_f
n_eval_g(s::FrankWolfeState) = s.n_eval_g

function stepsize(o::FrankWolfe{OpenLoop}, t::Integer)
    return o.step.shift / (t + o.step.shift)
end

# note: location of prototype only determines which boundary vertex is chosen first
function Kirstine.optimizer_state(
    f,
    g!,
    optimizer::FrankWolfe,
    prototypes::AbstractVector{<:AbstractPoint},
    constraints::AbstractConstraints,
)
    if length(prototypes) > 1
        msg = "FrankWolfe needs exactly 1 prototype, got $(length(prototypes))"
        throw(ArgumentError(msg))
    end
    x = deepcopy(only(prototypes))
    v = deepcopy(x)
    x_diff = deepcopy(ap_as_difference(v))
    gx = zeros(fw_gradient_length(x, constraints))
    g!(gx, x)
    any(isnan, gx) && error("gradient must be finite but contains NaN")
    any(isinf, gx) && error("gradient must be finite but contains Inf or -Inf")
    # Here we perform the update corresponding to step t=0 in [^P23].
    # With OpenLoop this always gives an x1 that is on the boundary of the compact convex set
    # since the initial step size is 1.
    fw_linear_maximization_oracle!(v, gx, constraints)

    step = stepsize(optimizer, 0)
    ap_diff!(x_diff, v, x)
    ap_mul!(x_diff, step)
    ap_add!(x, x_diff, constraints)

    fx = f(x)
    state = FrankWolfeState(x, v, x_diff, step, fx, gx, 1, 1)
    return state
end

function Kirstine.tick!(
    f,
    g!,
    state::FrankWolfeState,
    optimizer::FrankWolfe,
    constraints::AbstractConstraints,
    iter::Integer,
)
    g!(state.gx, state.x)
    state.n_eval_g += 1
    fw_linear_maximization_oracle!(state.v, state.gx, constraints)
    state.n_eval_f += 1
    # In [^P23] the time index starts at t=0.
    # Since we start to count from iter=1,
    # we have to subtract 1 from it in the stepsize computation.
    state.stepsize = stepsize(optimizer, iter - 1)
    ap_diff!(state.diff, state.v, state.x)
    ap_mul!(state.diff, state.stepsize)
    ap_add!(state.x, state.diff, constraints)
    state.fx = f(state.x)
    return state
end

# Methods that need to be implemented for T <: AbstractPoint, C <: AbstractConstraints are:
# * fw_linear_maximization_oracle!(v::T, gx::AbstractVector{<:Real}, c::C) is the linear maximization oracle,
#   where gx contains the gradient of the objective evaluated at x,
#   c are the constraints defining the compact convex region,
#   and v must be overwritten with the point of the boundary of the region that maximizes 〈v, gx〉
#   (given a suitable vector representation of v).
# * fw_gradient_length(x::T, c::C)::Integer should compute length(gx).
function fw_linear_maximization_oracle! end
function fw_gradient_length end
