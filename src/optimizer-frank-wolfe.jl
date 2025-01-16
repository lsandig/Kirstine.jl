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
    OpenLoop(; shift = 2)

Frank-Wolfe step-size strategy ``γ(t) = l / (t + l)`` for ``t=0,1,2,…`` with shift ``l∈ℕ, l≥1``.

!!! note

    With `OpenLoop`, the algorithm will always land on a vertex of the feasible set at ``t=1``.
    For a probability simplex, these vertices are the one-point designs.
    As a consequence, `OpenLoop` is _not_ suitable for the re-weighting step of [`Exchange`](@ref)
    for design problems where one-point designs are singular.
"""
@kwdef struct OpenLoop <: StepSizeStrategy
    shift::Int64 = 2
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

A Frank-Wolfe optimizer.

!!! warning "Experimental Feature"

    This optimizer is experimental.
    It is not to be considered part of the API yet,
    hence it is not exported and may be subject to breaking changes
    even in minor or patch version updates.

    It is currently intended to be used for the re-weighting step of [`Exchange`](@ref).
    For [`DirectMaximization`](@ref) it can currently only be used
    when all design points are fixed.

Two variants are implemented:
a basic one that always moves towards the best vertex for the linearized problem
(see Algorithm 4 in [^P23]),
and a version with pairwise steps (see Algorithm 2 in [^LJ15])
that is better at finding an optimum on the boundary.

# Initialization

Frank-Wolfe can only be initialized with a single prototype.
This prototype will be used as the starting value.

[^LJ15]: Simon Lacoste-Julien and Martin Jaggi (2015). On the Global Linear Convergence of Frank-Wolfe Optimization Variants. [arXiv:1511.05932](https://arxiv.org/abs/1511.05932v1)
[^P23]: S. Pokutta (2023). The Frank-Wolfe algorithm: a short introduction. Jahresbericht der Deutschen Mathematiker-Vereinigung, 126(1), 3–35. [doi:10.1365/s13291-023-00275-x](http://dx.doi.org/10.1365/s13291-023-00275-x)
"""
@kwdef struct FrankWolfe{T<:StepSizeStrategy} <: FirstOrderOptimizer
    iterations::Int64
    stepsize::StepSizeStrategy
    pairwise::Bool
    @doc """
        FrankWolfe(; iterations::Integer, stepsize::StepSizeStrategy, pairwise::Bool)

    Set up parameters for the Frank-Wolfe algorithm.
    """
    function FrankWolfe(iterations::Integer, s::StepSizeStrategy, pairwise::Bool)
        iterations <= 0 && throw(ArgumentError("number of iterations must be positive"))
        return new{typeof(s)}(iterations, s, pairwise)
    end
end

iterations(o::FrankWolfe) = o.iterations

mutable struct FrankWolfeState{T,U,V} <: OptimizerState{T,U}
    x::T
    v::T
    direction::U
    stepsize::Float64
    fx::Float64
    gx::Vector{<:Float64}
    frank_wolfe_gap::Float64
    alpha::Vector{Float64}
    active_set::Vector{T}
    work::V
    active_points_added::Int64
    active_points_removed::Int64
    n_eval_f::Int64
    n_eval_g::Int64
end

maximizer(s::FrankWolfeState) = s.x
maximum(s::FrankWolfeState) = s.fx
gradient(s::FrankWolfeState) = s.gx
n_eval_f(s::FrankWolfeState) = s.n_eval_f
n_eval_g(s::FrankWolfeState) = s.n_eval_g

function stepsize_openloop(shift::Integer, t::Integer, gamma_max::Real)
    return min(shift / (t + shift), gamma_max)
end

function stepsize!(
    g!,
    state::FrankWolfeState,
    o::FrankWolfe{OpenLoop},
    c::AbstractConstraints,
    t::Integer,
    gamma_max::Real,
)
    return stepsize_openloop(o.stepsize.shift, t, gamma_max)
end

function stepsize_constnumiter(iterations::Integer, t::Integer, gamma_max::Real)
    t > iterations && error("can't iterate past maximum number of iterations")
    return min(1 / sqrt(iterations), gamma_max)
end

function stepsize!(
    g!,
    state::FrankWolfeState,
    o::FrankWolfe{ConstNumIter},
    c::AbstractConstraints,
    t::Integer,
    gamma_max::Real,
)
    return stepsize_constnumiter(o.iterations, t, gamma_max)
end

function fw_estimate_initial_smoothness!(
    g!,
    w::AdaptiveWorkspace,
    x::AbstractPoint,
    gx::AbstractVector{<:Real},
    direction::AbstractPointDifference,
    c::AbstractConstraints,
    gamma_max::Real,
)
    # in-place initial estimate for Lipschitz constant of the gradient,
    # norm(g!(x) - g!(x + ε*direction)) / (ε * norm(direction))
    epsilon = min(1e-3, gamma_max)
    ap_copy!(w.x_tmp, x)
    ap_copy!(w.d_tmp, direction)
    ap_mul!(w.d_tmp, epsilon)
    ap_add!(w.x_tmp, w.d_tmp, c)
    g!(w.gx_tmp, w.x_tmp)
    w.gx_tmp .-= gx
    # Note: norm(direction) does not need to be subject to constraints.
    # For a DesignMeasure with some fixed elements,
    # the difference of these elements will always be zero.
    w.smoothness_estimate = norm(w.gx_tmp) / (epsilon * fw_norm(direction))
    return w
end

# This is modeled after the adaptive line search in the FrankWolfe.jl package.
#
# It assumes that the following fields of the state have been updated for the current x
# before the call: x, gx, direction
#
# Notable differences from how the stepsize computation is described in [^P23]:
#
# (1) In [^P23] the short step estimation has an inner product with the opposite of our
#     direction. However, since we maximize the objective function, the two sign switches
#     cancel out.
# (2) The same holds for the acceptance test.
# (3) In [^P23] they have gamma_max := 1 and -direction := v - x, this version can have more
#     general 0 < gamma_max <= 1 and other directions. This is necessary for the pairwise
#     algorithm variant as in [^LJ15] and in the FrankWolfe.jl package.
function stepsize!(
    g!,
    state::FrankWolfeState,
    o::FrankWolfe{Adaptive},
    c::AbstractConstraints,
    t::Integer,
    gamma_max::Real,
)
    w = state.work
    if isinf(w.smoothness_estimate)
        fw_estimate_initial_smoothness!(
            g!,
            w,
            state.x,
            state.gx,
            state.direction,
            c,
            gamma_max,
        )
        state.n_eval_g += 1
    end
    M = o.stepsize.eta * w.smoothness_estimate
    dd_direction = fw_evaluate_linearization(state.gx, state.direction, c)
    while true
        gamma = min(dd_direction / (M * fw_norm(state.direction)^2), gamma_max)
        gamma < 0 && error("gamma should not become negative")
        # in-place computation of g!(x + gamma * direction)
        ap_copy!(w.d_tmp, state.direction)
        ap_mul!(w.d_tmp, gamma)
        ap_copy!(w.x_tmp, state.x)
        ap_add!(w.x_tmp, w.d_tmp, c)
        try
            g!(w.gx_tmp, w.x_tmp)
        catch e
            if e isa SingularException
                msg = """singular intermediate design in adaptive step-size computation,
               it might help to increase the initial smoothness estimate"""
                @error msg o.stepsize.initial_smoothness_estimate M gamma w.x_tmp
            end
            rethrow(e)
        end
        state.n_eval_g += 1
        if fw_evaluate_linearization(w.gx_tmp, state.direction, c) >= 0
            w.smoothness_estimate = M
            return gamma
        end
        if M > o.stepsize.smoothness_threshold
            @warn "smoothness estimate seems to diverge, terminating step size search"
            return gamma
        end
        M *= o.stepsize.tau
    end
end

function optimizer_state(
    (f, g!),
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
    dir = deepcopy(ap_as_difference(v))
    grad_length = fw_gradient_length(x, constraints)
    gx = fill(NaN, grad_length)
    as = [deepcopy(x)]
    work = fw_workspace(optimizer.stepsize, x, grad_length)
    state = FrankWolfeState(x, v, dir, NaN, NaN, gx, NaN, [1.0], as, work, 0, 0, 0, 0)
    # Here we perform the update corresponding to step t=0 in [^P23].
    # With OpenLoop this always gives an x1 that is on the boundary of the compact convex set
    # since the initial step size is 1.
    tick!((f, g!), state, optimizer, constraints, 1)
    return state
end

function tick!(
    (f, g!),
    state::FrankWolfeState,
    optimizer::FrankWolfe,
    constraints::AbstractConstraints,
    iter::Integer,
)
    try
        g!(state.gx, state.x)
    catch e
        @error "current design is singular" iter state.v state.x
        rethrow(e)
    end
    state.n_eval_g += 1
    fw_linear_maximization_oracle!(state.v, state.gx, constraints)
    ap_diff!(state.direction, state.v, state.x)
    # Note: this is the gap for x from the previous iteration!
    state.frank_wolfe_gap =
        fw_evaluate_linearization(state.gx, state.direction, constraints)
    away = -1
    if optimizer.pairwise
        _, away = findmin(state.active_set) do a
            fw_evaluate_linearization(state.gx, a, constraints)
        end
        ap_diff!(state.direction, state.v, state.active_set[away])
    end
    gamma_max = optimizer.pairwise ? state.alpha[away] : 1.0
    # In [^P23] the time index starts at t=0.
    # Since we start to count from iter=1,
    # we have to subtract 1 from it in the stepsize computation.
    state.stepsize = stepsize!(g!, state, optimizer, constraints, iter - 1, gamma_max)
    if !optimizer.pairwise
        state.alpha .*= (1 - state.stepsize)
        if !(state.v in state.active_set)
            state.active_points_added += 1
            push!(state.active_set, deepcopy(state.v))
            push!(state.alpha, state.stepsize)
        else
            idx = findfirst(==(state.v), state.active_set)
            state.alpha[idx] += state.stepsize
            state.alpha[idx] = min(state.alpha[idx], 1.0)
        end
    else
        if !(state.v in state.active_set)
            state.active_points_added += 1
            push!(state.active_set, deepcopy(state.v))
            push!(state.alpha, state.stepsize)
            state.alpha[away] = max(state.alpha[away] - state.stepsize, 0)
        else
            idx = findfirst(==(state.v), state.active_set)
            state.alpha[away] = max(state.alpha[away] - state.stepsize, 0)
            state.alpha[idx] = min(state.alpha[idx] + state.stepsize, 1)
        end
        if state.alpha[away] == 0
            state.active_points_removed += 1
            deleteat!(state.alpha, away)
            deleteat!(state.active_set, away)
        end
    end
    ap_convex_combination!(state.x, state.alpha, state.active_set, constraints)
    state.fx = f(state.x)
    state.n_eval_f += 1
    return state
end

# Methods that need to be implemented for T <: AbstractPoint, U <: AbstractPointDifference, C <: AbstractConstraints are:
# * fw_linear_maximization_oracle!(v::T, gx::AbstractVector{<:Real}, c::C) is the linear maximization oracle,
#   where gx contains the gradient of the objective evaluated at x,
#   c are the constraints defining the compact convex region,
#   and v must be overwritten with the point of the boundary of the region that maximizes 〈v, gx〉
#   (given a suitable vector representation of v).
# * fw_evaluate_linearization(gx::AbstractVector{<:Real}, y::T, c::C) and
#   fw_evaluate_linearization(gx::AbstractVector{<:Real}, y::U, c::C)
#   is the inner product between the gradient of the objective evaluated at x
#   and the vector representation of y.
#   In order to know how to compute this we also need the constraints c here.
# * fw_gradient_length(x::T, c::C)::Integer should compute length(gx).
function fw_linear_maximization_oracle! end
function fw_evaluate_linearization end
function fw_gradient_length end
