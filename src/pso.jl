# generic particle swarm optimizer

# declare functions to be overwritten in unit tests
function randomize! end
function difference! end
function flat_length end
function copy! end
function move! end

"""
    Pso(; iterations, swarmsize, c1 = 2.05, c2 = 2.05)

Constricted particle swarm optimizer.

`c1` controls movement towards a particle's personal best, while `c2` controls
movement towards the global best. In this implementation, a particle's
neighborhood is the whole swarm.

## References

  - Clerc, M., and Kennedy, J. (2002).
    [The particle swarm - explosion, stability, and convergence in a multidimensional complex space](http://dx.doi.org/10.1109/4235.985692).
    IEEE Transactions on Evolutionary Computation, 6(1), 58–73.
  - Eberhardt, R.C. and Shi, Y. (2000).
    [Comparing inertia weights and constriction factors in particle swarm optimization](https://doi.org/10.1109/CEC.2000.870279).
    Proceedings of the 2000 Congress on Evolutionary Computation
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

function Pso(; iterations, swarmsize, c1 = 2.05, c2 = 2.05)
    return Pso(iterations, swarmsize, c1, c2)
end

iterations(optimizer::Pso) = optimizer.iterations

mutable struct PsoState{T} <: OptimizerState{T}
    x::Vector{T} # current position
    p::Vector{T} # personal best
    g::T # global best
    v::Vector{Vector{Float64}} # velocity vectors
    fx::Vector{Float64} # objective function values at x
    fp::Vector{Float64} # objective function values at p
    fg::Float64 # objective function value at g
    r1::Vector{Float64} # uniformly random velocity vector
    r2::Vector{Float64} # uniformly random velocity vector
    diffg::Vector{Float64} # temporary variable for g - x
    diffp::Vector{Float64} # temporary variable for p - x
end

maximizer(state::PsoState) = state.g
maximum(state::PsoState) = state.fg

function optimizer_state(
    o::Pso,
    candidates::AbstractVector{<:AbstractPoint},
    f,
    constraints,
)
    if o.swarmsize < length(candidates)
        error("swarmsize must be a least as large as number of candidate solutions")
    end
    x_given = deepcopy(candidates)
    n_random = o.swarmsize - length(candidates)
    x_random = [deepcopy(candidates[1]) for _ in 1:n_random]
    for i in 1:length(x_random)
        randomize!(x_random[i], constraints)
    end
    x = [x_given; x_random]
    velocity_length = flat_length(x[1])
    v = [zeros(velocity_length) for _ in 1:(o.swarmsize)]
    p = deepcopy(x)
    g = deepcopy(x[1])
    fx = zeros(o.swarmsize)
    fp = zeros(o.swarmsize)
    fg = 0.0
    r1 = zeros(velocity_length)
    r2 = zeros(velocity_length)
    diffg = zeros(velocity_length)
    diffp = zeros(velocity_length)
    state = PsoState(x, p, g, v, fx, fp, fg, r1, r2, diffg, diffp)
    pso_evaluate_objective!(state, f)
    state.fp .= -Inf # make sure these get updated
    state.fg = -Inf
    pso_update_best!(state)
    return state
end

function tick!(state::PsoState, optimizer::Pso, f, constraints)
    phi = optimizer.c1 + optimizer.c2
    chi = 2 / abs(2 - phi - sqrt(phi^2 - 4 * phi))
    pso_update_velocity!(state, optimizer.c1, optimizer.c2, chi)
    pso_update_position!(state, constraints)
    pso_evaluate_objective!(state, f)
    pso_update_best!(state)
    return state
end

function pso_update_velocity!(state::PsoState, c1, c2, chi)
    for i in 1:length(state.x)
        difference!(state.diffp, state.p[i], state.x[i])
        difference!(state.diffg, state.g, state.x[i])
        rand!(state.r1)
        rand!(state.r2)
        state.v[i] .+= c1 .* state.r1 .* state.diffp
        state.v[i] .+= c2 .* state.r2 .* state.diffg
        state.v[i] .*= chi
    end
    return state
end
function pso_update_position!(state::PsoState, constraints)
    for i in 1:length(state.x)
        move!(state.x[i], state.v[i], constraints)
    end
    return state
end
function pso_evaluate_objective!(state::PsoState, f)
    for i in 1:length(state.x)
        state.fx[i] = f(state.x[i])
    end
    return state
end
function pso_update_best!(state::PsoState)
    for i in 1:length(state.x)
        if state.fx[i] > state.fp[i]
            copy!(state.p[i], state.x[i])
            state.fp[i] = state.fx[i]
            if state.fx[i] > state.fg
                copy!(state.g, state.x[i])
                state.fg = state.fx[i]
            end
        end
    end
    return state
end
