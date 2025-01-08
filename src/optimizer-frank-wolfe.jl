# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

"""
    StepSizeStrategy

Supertype for Frank-Wolfe step-size strategies.
"""
abstract type StepSizeStrategy end

abstract type FWWorkspace end

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

struct FWDummyWorkspace <: FWWorkspace end

function fw_workspace(step::OpenLoop, x::AbstractPoint, gradient_length::Integer)
    return FWDummyWorkspace()
end

"""
    ConstNumIter <: StepSizeStrategy

Frank-Wolfe step-size strategy ``γ(t) = 1 / \\sqrt{T}`` for a run with ``T`` iterations ``t=0,…,T-1``.
"""
struct ConstNumIter <: StepSizeStrategy end

function fw_workspace(step::ConstNumIter, x::AbstractPoint, gradient_length::Integer)
    return FWDummyWorkspace()
end

"""
    Adaptive <: StepSizeStrategy
    Adaptive(; eta = 0.9, tau = 2.0, initial_smoothness_estimate = Inf, smoothness_threshold = 1e10)

Adaptive Frank-Wolfe step-size strategy that uses a smoothness property of the objective function.

Since we typically do not know the smoothness constant of the objective function ``f``,
i.e. a real number ``L`` such that
``⟨∇f(y) - ∇f(x), y - x⟩ ≤ L\\lVert y-x\\rVert^2``
for all ``x`` and ``y`` in the convex domain of ``f``,
the constant is dynamically approximated via multiplicative search with parameters `eta ≤ 1` and `tau > 1`.
When `initial_smoothness_estimate = Inf`,
the initial estimate is computed from a local approximation at the starting value.

If at any point during the search the smoothness estimate exceeds `smoothness_threshold`,
this is interpreted as the divergence of the multiplicative search,
and the current (non-optimal) step size is returned.

This strategy is an implementation of Algorithm 5 from [^P23],
and the defaults of the parameters are taken from the adaptive line search in
[FrankWolfe.jl](https://doi.org/10.5281/zenodo.14565428).
"""
@kwdef struct Adaptive <: StepSizeStrategy
    eta::Float64 = 0.9
    tau::Float64 = 2.0
    initial_smoothness_estimate::Float64 = Inf
    smoothness_threshold::Float64 = 1e10
    function Adaptive(
        eta::Real,
        tau::Real,
        initial_smoothness_estimate::Real,
        smoothness_threshold::Real,
    )
        eta <= 1 || throw(ArgumentError("eta must be <= 1"))
        tau > 1 || throw(ArgumentError("tau must be > 1"))
        smoothness_threshold > 0 || throw(ArgumentError("smoothness threshold must be > 0"))
        if initial_smoothness_estimate != Inf &&
           !(0 < initial_smoothness_estimate <= smoothness_threshold)
            msg = "initial smoothness estimate must be Inf or in (0, threshold]"
            throw(ArgumentError(msg))
        end
        return new(eta, tau, initial_smoothness_estimate, smoothness_threshold)
    end
end

mutable struct AdaptiveWorkspace{T<:AbstractPoint,U<:AbstractPointDifference} <: FWWorkspace
    smoothness_estimate::Float64
    x_tmp::T
    d_tmp::U
    gx_tmp::Vector{Float64}
end

function fw_workspace(step::Adaptive, x::AbstractPoint, gradient_length::Integer)
    d_tmp = ap_as_difference(deepcopy(x))
    x_tmp = deepcopy(x)
    gx_tmp = fill(NaN, gradient_length)
    return AdaptiveWorkspace(step.initial_smoothness_estimate, x_tmp, d_tmp, gx_tmp)
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

mutable struct FrankWolfeState{T,U,V} <: OptimizerState{T,U}
    x::T
    v::T
    diff::U
    stepsize::Float64
    fx::Float64
    gx::Vector{<:Float64}
    work::V
    n_eval_f::Int64
    n_eval_g::Int64
end

maximizer(s::FrankWolfeState) = s.x
maximum(s::FrankWolfeState) = s.fx
gradient(s::FrankWolfeState) = s.gx
n_eval_f(s::FrankWolfeState) = s.n_eval_f
n_eval_g(s::FrankWolfeState) = s.n_eval_g

function stepsize_openloop(shift::Integer, t::Integer)
    return shift / (t + shift)
end

function stepsize!(
    g!,
    state::FrankWolfeState,
    o::FrankWolfe{OpenLoop},
    c::AbstractConstraints,
    t::Integer,
)
    return stepsize_openloop(o.step.shift, t)
end

function stepsize_constnumiter(iterations::Integer, t::Integer)
    t > iterations && error("can't iterate past maximum number of iterations")
    return 1 / sqrt(iterations)
end

function stepsize!(
    g!,
    state::FrankWolfeState,
    o::FrankWolfe{ConstNumIter},
    c::AbstractConstraints,
    t::Integer,
)
    return stepsize_constnumiter(o.iterations, t)
end

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
    grad_length = fw_gradient_length(x, constraints)
    gx = fill(NaN, grad_length)
    work = fw_workspace(optimizer.step, x, grad_length)
    state = FrankWolfeState(x, v, x_diff, NaN, NaN, gx, work, 0, 0)
    # Here we perform the update corresponding to step t=0 in [^P23].
    # With OpenLoop this always gives an x1 that is on the boundary of the compact convex set
    # since the initial step size is 1.
    tick!(f, g!, state, optimizer, constraints, 1)
    any(isnan, state.gx) && error("gradient must be finite but contains NaN")
    any(isinf, state.gx) && error("gradient must be finite but contains Inf or -Inf")
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
    ap_diff!(state.diff, state.v, state.x)
    state.stepsize = stepsize!(g!, state, optimizer, constraints, iter - 1)
    ap_mul!(state.diff, state.stepsize)
    ap_add!(state.x, state.diff, constraints)
    state.fx = f(state.x)
    return state
end

# Methods that need to be implemented for T <: AbstractPoint, U <: AbstractPointDifference, C <: AbstractConstraints are:
# * fw_linear_maximization_oracle!(v::T, gx::AbstractVector{<:Real}, c::C) is the linear maximization oracle,
#   where gx contains the gradient of the objective evaluated at x,
#   c are the constraints defining the compact convex region,
#   and v must be overwritten with the point of the boundary of the region that maximizes 〈v, gx〉
#   (given a suitable vector representation of v).
# * fw_evaluate_linearization(gx::AbstractVector{<:Real}, y::U, c::C) is the inner product
#   between the gradient of the objective evaluated at x and the vector representation of y.
#   In order to know how to compute this we also need the constraints c here.
# * fw_gradient_length(x::T, c::C)::Integer should compute length(gx).
function fw_linear_maximization_oracle! end
function fw_evaluate_linearization end
function fw_gradient_length end
