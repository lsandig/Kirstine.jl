# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## generic particle swarm optimizer ##

"""
    Pso <: Optimizer

A constricted particle swarm optimizer.

There is a large number of published PSO variants,
this one was taken from two [^CK02] publications [^ES00].

## Initialization

The swarm is initialized from a vector of prototypical `AbstractPoint`s,
which are supplied by the caller via a non-exported interface.
Construction from the vector of `prototypes` proceeds as follows:

 1. The first `length(prototypes)` particles are a deep copy of the `prototypes` vector.

 2. Each of the remaining particles is a deep copy of `prototypes[1]`
    which is subsequently randomized.
    Randomization is subject to constraints, if any are given.

[^CK02]: M. Clerc and J. Kennedy (2002). The particle swarm - explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation, 6(1), 58–73. [doi:10.1109/4235.985692](http://dx.doi.org/10.1109/4235.985692)
[^ES00]: R.C. Eberhardt and Y. Shi (2000). Comparing inertia weights and constriction factors in particle swarm optimization. Proceedings of the 2000 Congress on Evolutionary Computation. [doi: 10.1109/CEC.2000.870279](https://doi.org/10.1109/CEC.2000.870279)
"""
struct Pso <: Optimizer
    iterations::Int64
    swarmsize::Int64
    c1::Float64
    c2::Float64
    function Pso(iterations, swarmsize, c1, c2)
        if iterations <= 0
            error("number of iterations must be positive")
        end
        if swarmsize <= 1
            error("swarmsize must be at least 2")
        end
        if c1 < 0 || c2 < 0 || c1 + c2 <= 4
            error("c1 and c2 must be non-negative and, c1 + c2 > 4")
        end
        return new(iterations, swarmsize, c1, c2)
    end
end

"""
    Pso(; iterations, swarmsize, c1 = 2.05, c2 = 2.05)

Set up parameters for a particle swarm optimizer.

`c1` controls movement towards a particle's personal best, while `c2` controls
movement towards the global best. In this implementation, a particle's
neighborhood is the whole swarm.
"""
function Pso(; iterations, swarmsize, c1 = 2.05, c2 = 2.05)
    return Pso(iterations, swarmsize, c1, c2)
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
maximum(state::PsoState) = state.fg
n_eval(state::PsoState) = state.n_eval

# Note: user-visible documentation for this method is attached to `Pso`.
function optimizer_state(
    f,
    o::Pso,
    prototypes::AbstractVector{<:AbstractPoint},
    constraints::AbstractConstraints,
)
    if o.swarmsize < length(prototypes)
        error("swarmsize must be a least as large as number of prototype particles")
    end
    x_given = deepcopy(prototypes)
    n_random = o.swarmsize - length(prototypes)
    x_random = [deepcopy(prototypes[1]) for _ in 1:n_random]
    for i in 1:length(x_random)
        ap_random_point!(x_random[i], constraints)
    end
    x = [x_given; x_random]
    # here we get the matching difference type for the concrete type of x[1]
    zero_diff = ap_as_difference(x[1])
    ap_difference!(zero_diff, x[1], x[1]) # initialize at 0
    v = [deepcopy(zero_diff) for _ in 1:(o.swarmsize)]
    p = deepcopy(x)
    g = deepcopy(x[1])
    fx = zeros(o.swarmsize)
    fp = zeros(o.swarmsize)
    fg = 0.0
    r1 = deepcopy(zero_diff)
    r2 = deepcopy(zero_diff)
    diffg = deepcopy(zero_diff)
    diffp = deepcopy(zero_diff)
    n_eval = 0
    state = PsoState(x, p, g, v, fx, fp, fg, r1, r2, diffg, diffp, n_eval)
    pso_evaluate_objective!(f, state)
    state.fp .= -Inf # make sure these get updated
    state.fg = -Inf
    pso_update_best!(state)
    return state
end

function tick!(f, state::PsoState, optimizer::Pso, constraints::AbstractConstraints)
    phi = optimizer.c1 + optimizer.c2
    chi = 2 / abs(2 - phi - sqrt(phi^2 - 4 * phi))
    pso_update_velocity!(state, optimizer.c1, optimizer.c2, chi)
    pso_update_position!(state, constraints)
    pso_evaluate_objective!(f, state)
    pso_update_best!(state)
    return state
end

function pso_update_velocity!(state::PsoState, c1, c2, chi)
    for i in 1:length(state.x)
        # In vector notation:
        #
        #   v[i] += c1 .* r1 .* (g - x[i]) .+ c2 .* r2 .* (p[i] - x[i])
        #   v[i] *= chi
        #
        # with elements of r1, r2 uniformly random from [0,1].
        ap_difference!(state.diffp, state.p[i], state.x[i])
        ap_difference!(state.diffg, state.g, state.x[i])
        ap_random_difference!(state.r1, 0, 1)
        ap_random_difference!(state.r2, 0, 1)
        ap_mul_hadamard!(state.diffp, state.r1)
        ap_mul_hadamard!(state.diffg, state.r2)
        ap_mul_scalar!(state.diffp, c1)
        ap_mul_scalar!(state.diffg, c2)
        ap_add!(state.v[i], state.diffp)
        ap_add!(state.v[i], state.diffg)
        ap_mul_scalar!(state.v[i], chi)
    end
    return state
end

function pso_update_position!(state::PsoState, constraints::AbstractConstraints)
    for i in 1:length(state.x)
        ap_move!(state.x[i], state.v[i], constraints)
    end
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
