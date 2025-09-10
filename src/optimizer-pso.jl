# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## generic particle swarm optimizer ##

"""
    Pso <: Kirstine.ZerothOrderOptimizer

A constricted particle swarm optimizer.

There is a large number of published PSO variants,
this one was taken from two [^CK02] publications [^ES00].

Note that particle swarm optimization has no convergence guarantees,
hence the `has_converged` field of the [`OptimizationResult`](@ref)
returned by a PSO run is always `false` by definition.

# Initialization

The swarm is initialized from a vector of prototypical `AbstractPoint`s,
which are supplied by the caller via a non-exported interface.
Construction from the vector of `prototypes` proceeds as follows:

 1. The first `length(prototypes)` particles are a deep copy of the `prototypes` vector.

 2. Each of the remaining particles is a deep copy of `prototypes[1]`
    which is subsequently randomized.
    Randomization is subject to constraints, if any are given.

[^CK02]: Maurice Clerc and James Kennedy (2002). The particle swarm - explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation, 6(1), 58–73. [doi:10.1109/4235.985692](http://dx.doi.org/10.1109/4235.985692)
[^ES00]: Russell C. Eberhardt and Yuhui Shi (2000). Comparing inertia weights and constriction factors in particle swarm optimization. Proceedings of the 2000 Congress on Evolutionary Computation. [doi: 10.1109/CEC.2000.870279](https://doi.org/10.1109/CEC.2000.870279)
"""
@kwdef struct Pso <: ZerothOrderOptimizer
    iterations::Int64
    swarmsize::Int64
    c1::Float64 = 2.05
    c2::Float64 = 2.05
    @doc """
            Pso(; iterations, swarmsize, c1 = 2.05, c2 = 2.05)

        Set up parameters for a particle swarm optimizer.

        `c1` controls movement towards a particle's personal best, while `c2` controls
        movement towards the global best. In this implementation, a particle's
        neighborhood is the whole swarm.
        """
    function Pso(iterations::Integer, swarmsize::Integer, c1::Real, c2::Real)
        iterations <= 0 && throw(ArgumentError("number of iterations must be positive"))
        swarmsize <= 1 && throw(ArgumentError("swarmsize must be at least 2"))
        if c1 < 0 || c2 < 0 || c1 + c2 <= 4
            throw(ArgumentError("c1 and c2 must be non-negative, and c1 + c2 > 4"))
        end
        return new(iterations, swarmsize, c1, c2)
    end
end

iterations(optimizer::Pso) = optimizer.iterations

mutable struct PsoState{T,U} <: OptimizerState{T,U}
    x::Vector{T} # current position
    p::Vector{T} # personal best
    g::T # global best
    v::Vector{U} # velocities
    fx::Vector{Float64} # objective function values at x
    fp::Vector{Float64} # objective function values at p
    fg::Float64 # objective function value at g
    r1::U # uniformly random velocity components
    r2::U # uniformly random velocity components
    diffg::U # temporary variable for g - x
    diffp::U # temporary variable for p - x
    n_eval::Int64 # cumulative number of objective evaluations
end

maximizer(state::PsoState) = state.g
Base.maximum(state::PsoState) = state.fg
n_eval_f(state::PsoState) = state.n_eval
n_eval_g(state::PsoState) = 0
has_converged(state::PsoState, o::Pso) = false

# Note: user-visible documentation for this method is attached to `Pso`.
function optimizer_state(
    f::Function,
    o::Pso,
    prototypes::AbstractVector{<:AbstractPoint},
    constraints::AbstractConstraints,
)
    if o.swarmsize < length(prototypes)
        @error "" o.swarmsize prototypes
        throw(ArgumentError("number of prototype particles must be <= swarmsize"))
    end
    n_random = o.swarmsize - length(prototypes)
    x_random = [ap_rand!(deepcopy(first(prototypes)), constraints) for _ in 1:n_random]
    x = [deepcopy(prototypes); x_random]
    # here we get the matching difference type for the concrete type of x[1]
    zero_diff = ap_diff!(ap_as_difference(x[1]), x[1], x[1]) # initialize at apd 0
    v = [deepcopy(zero_diff) for _ in 1:(o.swarmsize)]
    p, g = deepcopy(x), deepcopy(x[1])
    fx, fp, fg = zeros(o.swarmsize), zeros(o.swarmsize), 0.0
    r1, r2 = deepcopy(zero_diff), deepcopy(zero_diff)
    diffg, diffp = deepcopy(zero_diff), deepcopy(zero_diff)
    n_eval = 0
    state = PsoState(x, p, g, v, fx, fp, fg, r1, r2, diffg, diffp, n_eval)
    pso_evaluate_objective!(f, state)
    state.fp .= -Inf # make sure these get updated
    state.fg = -Inf
    pso_update_best!(state)
    # check for broken initial state
    all(isnan.(state.fx)) && error("objective evaluates to NaN on all initial points")
    return state
end

function tick!(f, state::PsoState, optimizer::Pso, constraints::AbstractConstraints)
    phi = optimizer.c1 + optimizer.c2
    chi = 2 / abs(2 - phi - sqrt(phi^2 - 4 * phi))
    for i in 1:length(state.x)
        pso_randomize_velocity_multipliers!(state)
        pso_update_velocity!(state, i, optimizer.c1, optimizer.c2, chi)
    end
    pso_update_position!(state, constraints)
    pso_evaluate_objective!(f, state)
    pso_update_best!(state)
    return state
end

function pso_randomize_velocity_multipliers!(state::PsoState)
    ap_rand!(state.r1, 0, 1)
    ap_rand!(state.r2, 0, 1)
    return state
end

function pso_update_velocity!(state::PsoState, i, c1, c2, chi)
    # In vector notation:
    #
    #   v[i] += c1 .* r1 .* (p[i] - x[i]) .+ c2 .* r2 .* (g - x[i])
    #   v[i] *= chi
    #
    # with elements of r1, r2 uniformly random from [0,1].
    ap_diff!(state.diffp, state.p[i], state.x[i])
    ap_diff!(state.diffg, state.g, state.x[i])
    ap_mul!(state.diffp, state.r1)
    ap_mul!(state.diffg, state.r2)
    ap_mul!(state.diffp, c1)
    ap_mul!(state.diffg, c2)
    ap_add!(state.v[i], state.diffp)
    ap_add!(state.v[i], state.diffg)
    ap_mul!(state.v[i], chi)
    return state
end

function pso_update_position!(state::PsoState, constraints::AbstractConstraints)
    map((x, v) -> ap_add!(x, v, constraints), state.x, state.v)
    return state
end

function pso_evaluate_objective!(f, state::PsoState)
    for i in 1:length(state.x)
        state.fx[i] = f(state.x[i])
    end
    state.n_eval += length(state.x)
    return state
end

function pso_update_best!(state::PsoState)
    for i in 1:length(state.x)
        if state.fx[i] > state.fp[i]
            ap_copy!(state.p[i], state.x[i])
            state.fp[i] = state.fx[i]
            if state.fx[i] > state.fg
                ap_copy!(state.g, state.x[i])
                state.fg = state.fx[i]
            end
        end
    end
    return state
end
