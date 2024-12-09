# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
    FrankWolfe <: Optimizer

A basic Frank-Wolfe algorithm.

This version is taken from [^P23]
and uses the open-loop step size ``γ(t) = 2 / (2 + t)`` in step ``t=0,1,…``.

[^P23]: S. Pokutta (2023). The Frank-Wolfe algorithm: a short introduction. Jahresbericht der Deutschen Mathematiker-Vereinigung, 126(1), 3–35. [doi:10.1365/s13291-023-00275-x](http://dx.doi.org/10.1365/s13291-023-00275-x)
"""
@kwdef struct FrankWolfe <: Optimizer
    iterations::Int64
    @doc """
        FrankWolfe(; iterations)
        FrankWolfe(iterations)

    Set up parameters for the Frank-Wolfe algorithm.

    This type has both a positional constructor and a keyword-based constructor.
    """
    function FrankWolfe(iterations::Integer)
        iterations <= 0 && throw(ArgumentError("number of iterations must be positive"))
        return new(iterations)
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
    n_eval::Int64
end

maximizer(s::FrankWolfeState) = s.x
maximum(s::FrankWolfeState) = s.fx
n_eval(s::FrankWolfeState) = s.n_eval

# note: location of prototype only determines which boundary vertex is chosen first
function Kirstine.optimizer_state(
    (f, g!),
    o::FrankWolfe,
    prototypes::AbstractVector{<:AbstractPoint},
    constraints::AbstractConstraints,
)
    if length(prototypes) > 1
        msg = "FrankWolfe needs exactly 1 prototype, got $(length(prototypes))"
        throw(ArgumentError(msg))
    end
    x = deepcopy(only(prototypes))
    v = deepcopy(x)
    gx = zeros(fw_gradient_length(x, constraints))
    g!(gx, x)
    any(isnan, gx) && error("gradient must be finite but contains NaN")
    any(isinf, gx) && error("gradient must be finite but contains Inf or -Inf")
    # Here we perform the update corresponding to step t=0 in [^P23].
    # This always gives an x1 that is on the boundary of the compact convex set
    # since the initial step size is 1.
    fw_linear_maximization_oracle!(v, gx, constraints)
    ap_copy!(x, v)
    x_diff = deepcopy(ap_as_difference(v))
    fx = f(x)
    state = FrankWolfeState{typeof(x),typeof(x_diff)}(x, v, x_diff, 1, fx, gx, 1)
    return state
end

function Kirstine.tick!(
    (f, g!),
    state::FrankWolfeState,
    optimizer::FrankWolfe,
    constraints::AbstractConstraints,
    iter::Integer,
)
    g!(state.gx, state.x)
    fw_linear_maximization_oracle!(state.v, state.gx, constraints)
    # In [^P23] the time index starts at t=0 and the open loop stepsize is γ_t = 2 / (2 + t).
    # Since we start to count from iter=1, we have to subtract 1 from t in the stepsize calculation.
    state.stepsize = 2 / (1 + iter)
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
