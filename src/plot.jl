@recipe function f(r::OptimizationResult)
    niter = length(r.trace_fx)
    iter = collect(0:(niter - 1))
    xguide --> "iteration"
    yguide --> "objective"
    xlims --> (0, niter)
    label --> ""
    iter, r.trace_fx
end

@recipe function f(rs::AbstractVector{OptimizationResult})
    nsteps = length(rs)
    steps = collect(1:nsteps)
    fx = [r.maximum for r in rs]
    xguide --> "step"
    yguide --> "best objective value"
    xlims --> (1, nsteps)
    label --> ""
    steps, fx
end
