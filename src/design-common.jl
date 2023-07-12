# generic design objective functions and Gateaux derivatives

# declare functions to be redefined for user models
#
# (m::NonlinearRegression) -> Real
function unit_length end
# m::NonlinearRegression -> Covariate
function allocate_covariate end
# (jm::AbstractMatrix, m::NonlinearRegression, c::Covariate, p) -> jm
function jacobianmatrix! end
# (c::Covariate, dp::AbstractVector{<:Real}, m::NonlinearRegression, cp::CovariateParameterization) -> c
function update_model_covariate! end
# m -> Real or m -> AbstractMatrix
function invcov end
# p::Parameter -> Integer
function dimension end

## main interface ##


"""
    solve(dp::DesignProblem [, strategy::ProblemSolvingStrategy])

Attempt to solve the design problem.

The default strategy is

```julia
DirectMaximization(;
    optimizer = Pso(; iterations = 50, swarmsize = 20),
    prototype = random_design(dp.ds, parameter_dimension(dp.pk)),
)
```

Returns a simplified and sorted [`DesignMeasure`](@ref) and a full [`OptimizationResult`](@ref).

See also [`DirectMaximization`](@ref).
"""
function solve(
    dp::DesignProblem,
    strategy::ProblemSolvingStrategy = DirectMaximization(;
        optimizer = Pso(; iterations = 50, swarmsize = 20),
        prototype = random_design(dp.ds, parameter_dimension(dp.pk)),
    ),
)
    constraints = DesignConstraints(
        strategy.prototype,
        dp.ds,
        strategy.fixedweights,
        strategy.fixedpoints,
    )
    tc = precalculate_trafo_constants(dp.trafo, dp.pk)
    wm = WorkMatrices(unit_length(dp.m), parameter_dimension(dp.pk), codomain_dimension(tc))
    c = allocate_initialize_covariates(strategy.prototype, dp.m, dp.cp)
    f = d -> objective!(wm, c, dp.dc, d, dp.m, dp.cp, dp.pk, tc, dp.na)
    or = optimize(
        strategy.optimizer,
        f,
        [strategy.prototype],
        constraints;
        trace_state = strategy.trace_state,
    )
    dopt = sort_designpoints(
        simplify(or.maximizer, dp.ds, dp.m, dp.cp; strategy.simplify_args...),
    )
    return dopt, or
end

"""
    refine_design(od::Optimizer,
                  ow::Optimizer,
                  steps::Int64,
                  candidate::DesignMeasure,
                  dp::DesignProblem;
                  trace_state = false,
                  sargs...)

Improve a `candidate` design by ascending its Gateaux derivative.

This repeats the following `steps` times, starting with `r = candidate`:

 1. [`simplify`](@ref) the design `r`, passing along `sargs`.
 2. Use the optimizer `od` to find the direction (one-point design / Dirac measure) `d`
    of steepest [`gateauxderivative`](@ref) at `r`.
    The vector of `prototypes` that is used for initializing `od`
    is constructed from one-point designs at the design points of `r`.
    See the [`Optimizer`](@ref)s for algorithm-specific details.
 3. Use the optimizer `ow` to re-calculcate optimal weights of a [`mixture`](@ref) of `r` and `d`
    for the [`DesignCriterion`](@ref).
    This is implemented as a call to [`solve`](@ref) with the [`DirectMaximization`](@ref) strategy
    and with all of the designpoints of `r` kept fixed.
 4. Set `r` to the result of step 3.

Returns a 3-Tuple:

  - The best [`DesignMeasure`](@ref) found after the last refinement step. As
    postprocessing, [`simplify`](@ref) is called with `sargs` and the design points are
    sorted with [`sort_designpoints`](@ref).
  - A vector of [`OptimizationResult`](@ref)s from the derivative-maximizing steps. If
    `trace_state=true`, the full state of the algorithm is saved for every iteration.
  - A vector of [`OptimizationResult`](@ref)s from the re-weighting steps. If
    `trace_state=true`, the full state of the algorithm is saved for every iteration
"""
function refine_design(
    od::Optimizer,
    ow::Optimizer,
    steps::Int64,
    candidate::DesignMeasure,
    dp::DesignProblem;
    trace_state = false,
    sargs...,
)
    check_compatible(candidate, dp.ds)
    tc = precalculate_trafo_constants(dp.trafo, dp.pk)
    wm = WorkMatrices(unit_length(dp.m), parameter_dimension(dp.pk), codomain_dimension(tc))
    c = allocate_initialize_covariates(
        one_point_design(candidate.designpoint[1]),
        dp.m,
        dp.cp,
    )
    ors_d = OptimizationResult[]
    ors_w = OptimizationResult[]
    constraints = DesignConstraints(dp.ds, [false], [false])

    res = candidate
    for i in 1:steps
        res = simplify(res, dp.ds, dp.m, dp.cp; sargs...)
        dir_prot = map(one_point_design, designpoints(simplify_drop(res, 0)))
        gconst = precalculate_gateaux_constants(dp.dc, res, dp.m, dp.cp, dp.pk, tc, dp.na)
        # find direction of steepest ascent
        gd(d) = gateauxderivative!(wm, c, gconst, d, dp.m, dp.cp, dp.pk, dp.na)
        or_gd = optimize(od, gd, dir_prot, constraints; trace_state = trace_state)
        push!(ors_d, or_gd)
        d = or_gd.maximizer
        # append the new atom
        K = length(res.weight)
        if d.designpoint[1] in designpoints(simplify_drop(res, 0))
            # effectivly run the reweighting from the last round for some more iterations
            res = mixture(0, d, res) # make sure new point is at index 1
            res = simplify_merge(res, dp.ds, 0)
        else
            K += 1
            res = mixture(1 / K, d, res)
        end
        # optimize weights
        wopt_strategy = DirectMaximization(;
            optimizer = ow,
            prototype = res,
            fixedpoints = 1:K,
            trace_state = trace_state,
            simplify_args = Dict(sargs...),
        )
        _, or_w = solve(dp, wopt_strategy)
        push!(ors_w, or_w)
        res = or_w.maximizer
    end
    dopt = sort_designpoints(simplify(res, dp.ds, dp.m, dp.cp; sargs...))
    return dopt, ors_d, ors_w
end

# == various helper functions == #
function DesignConstraints(
    d::DesignMeasure,
    ds::DesignSpace,
    fixedweights::AbstractVector{<:Integer},
    fixedpoints::AbstractVector{<:Integer},
)
    check_compatible(d, ds)
    K = length(weights(d))
    if any(fixedweights .< 1) || any(fixedweights .> K)
        error("indices for fixed weights must be between 1 and $K")
    end
    if any(fixedpoints .< 1) || any(fixedpoints .> K)
        error("indices for fixed points must be between 1 and $K")
    end
    fixw = [k in fixedweights for k in 1:K]
    fixp = [k in fixedpoints for k in 1:K]
    # Fixing all weights but one is equivalent to fixing them all. For
    # numerical stability it is better to explicitly fix them all.
    if count(fixw) == K - 1
        @info "explicitly fixing implicitly fixed weight"
        fixw .= true
    end
    return DesignConstraints(ds, fixw, fixp)
end

function precalculate_trafo_constants(trafo::Identity, pk::DiscretePrior)
    return TCIdentity(parameter_dimension(pk))
end

function check_trafo_jm_dimensions(jm, pk)
    r = parameter_dimension(pk)
    if any(j -> size(j) != size(jm[1]), jm)
        throw(DimensionMismatch("trafo jacobians must be identical in size"))
    end
    # We know all elements of jm have identical sizes, so checking the first is enough
    ncol = size(jm[1], 2)
    if ncol != r
        throw(DimensionMismatch("trafo jacobian must have $(r) columns, got $(ncol)"))
    end
    return nothing
end

function precalculate_trafo_constants(trafo::DeltaMethod, pk::DiscretePrior)
    jm = [trafo.jacobian_matrix(p) for p in pk.p]
    check_trafo_jm_dimensions(jm, pk)
    return TCDeltaMethod(size(jm[1], 1), jm)
end

function parameter_dimension(pk::DiscretePrior)
    return dimension(pk.p[1])
end

function codomain_dimension(tc::TrafoConstants)
    return tc.codomain_dimension
end

function informationmatrix!(
    nim::AbstractMatrix,
    jm::AbstractMatrix,
    w::AbstractVector,
    m::NonlinearRegression,
    invcov::Real,
    c::AbstractVector{<:Covariate},
    p::Parameter,
    na::FisherMatrix,
)
    fill!(nim, 0.0)
    for k in 1:length(w)
        jacobianmatrix!(jm, m, c[k], p)
        # The call to syrk! is equivalent to
        #
        #   nim = w[k] * invcov * jm' * jm + 1 * nim
        #
        # The result is symmetric, and only the 'U'pper triangle of nim is actually written
        # to. The lower triangle is not touched and is allowed to contain arbitrary garbage.
        syrk!('U', 'T', w[k] * invcov, jm, 1.0, nim)
    end
    return nim
end

# For debugging purposes, one will typically want to look at an information matrix
# corresponding to a single parameter value, not to all thousands of them in a prior sample.
function informationmatrix(
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    p::Parameter,
    na::NormalApproximation,
)
    c = Kirstine.allocate_initialize_covariates(d, m, cp)
    r = dimension(p)
    mm = unit_length(m)
    jm = zeros(mm, r)
    nim = zeros(r, r)
    informationmatrix!(nim, jm, d.weight, m, invcov(m), c, p, na)
    return Symmetric(nim)
end

# Calling conventions for `apply_transformation!`
#
# * `wm.r_x_r` holds the information matrix or its inverse, depending on the `is_inv` flag.
# * Only the upper triangle of `wm.r_x_r` is used.
# * `wm.t_x_t` will be overwritten with the transformed information matrix or its inverse.
# * The return value are `wm.t_x_t` and a flag that indicates whether it is inverted.
# * Only the upper triangle of `wm.t_x_t` is guaranteed to make sense, but specific methods
#   are free to return a dense matrix.
# * Whether the returned matrix will be inverted is _not_ controlled by `is_inv`.

function apply_transformation!(wm::WorkMatrices, is_inv::Bool, tc::TCIdentity, index)
    # For the Identity transformation we just pass through the information matrix.
    wm.t_x_t .= wm.r_x_r
    return wm.t_x_t, is_inv
end

function apply_transformation!(wm::WorkMatrices, is_inv::Bool, tc::TCDeltaMethod, index)
    # Denote the Jacobian matrix of T by J and the normalized information matrix by M. We
    # want to efficiently calculate J * inv(M) * J'.
    #
    # A precalculated J is given in tc.jm[index].
    #
    # Depending on whether wm.r_x_r contains (the upper triangle of) M or inv(M) we use
    # different BLAS routines and a different order of multiplication.
    if is_inv
        # Denote the given inverse of M by invM.
        # We first calculate A := (J * invM) and store it in wm.t_x_r.
        #
        # The `symm!` call performes the following in-place update:
        #
        #   wm.t_x_r = 1 * tc.jm[index] * Symmetric(wm.r_x_r) + 0 * wm.t_x_r
        #
        # That is,
        #  * the symmetric matrix wm.r_x_r is the factor on the 'R'ight, and
        #  * the data is contained in the 'U'pper triangle.
        symm!('R', 'U', 1.0, wm.r_x_r, tc.jm[index], 0.0, wm.t_x_r)
        # Next we calculate the result A * J' and store it in wm.t_x_t.
        mul!(wm.t_x_t, wm.t_x_r, tc.jm[index]')
    else
        # When the input is not yet inverted, we don't want to calculate inv(M) explicitly.
        # As a first step we calculate B := inv(M) * J and store it in wm.r_x_t.
        # We do this by solving the linear system M * B == J in place.
        # As we do not want to overwrite J, we copy J' into a work matrix.
        wm.r_x_t .= tc.jm[index]'
        # The `posv!` call performs the following in-place update:
        #
        #  * Overwrite wm.r_x_r with its Cholesky factor `potrf!(wm.r_x_r)`, using the data
        #    in the 'U'pper triangle.
        #  * Overwrite wm.r_x_t by the solution of the linear system.
        posv!('U', wm.r_x_r, wm.r_x_t)
        # Next, we calculate the result J * B and store it in wm.t_x_t.
        mul!(wm.t_x_t, tc.jm[index], wm.r_x_t)
    end
    # Note that for this method, the result is not just an upper triangle, but always a
    # dense matrix.
    return wm.t_x_t, true
end

function allocate_initialize_covariates(d, m, cp)
    K = length(d.weight)
    cs = [allocate_covariate(m) for _ in 1:K]
    for k in 1:K
        update_model_covariate!(cs[k], d.designpoint[k], m, cp)
    end
    return cs
end

# Calculate the Frobenius inner product tr(A*B). Both matrices are implicitly treated as
# symmetric, i.e. only the `uplo` triangles are used.
function tr_prod(A::AbstractMatrix, B::AbstractMatrix, uplo::Symbol)
    if size(A) != size(B)
        throw(ArgumentError("A and B must have identical size"))
    end
    k = size(A, 1)
    acc = 0.0
    if uplo == :U
        for i in 1:k
            acc += A[i, i] * B[i, i]
            for j in (i + 1):k
                acc += 2 * A[i, j] * B[i, j]
            end
        end
    elseif uplo == :L
        for j in 1:k
            acc += A[j, j] * B[j, j]
            for i in (j + 1):k
                acc += 2 * A[i, j] * B[i, j]
            end
        end
    else
        throw(ArgumentError("uplo argument must be either :U or :L, got $uplo"))
    end
    return acc
end

# Calculate `log(det(A))`. `A` is implicitly treated as symmetric, i.e. only the upper
# triangle is used. In the process, `A` is overwritten by its upper Cholesky factor.
# See also
# https://netlib.org/lapack/explore-html/d8/d6c/group__variants_p_ocomputational_ga2f55f604a6003d03b5cd4a0adcfb74d6.html
function log_det!(A::AbstractMatrix)
    potrf!('U', A)
    acc = 0
    for i in 1:size(A, 1)
        acc += A[i, i] > 0 ? log(A[i, i]) : -Inf
    end
    return 2 * acc
end

function objective!(
    wm::WorkMatrices,
    c::AbstractVector{<:Covariate},
    dc::DesignCriterion,
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::DiscretePrior,
    tc::TrafoConstants,
    na::NormalApproximation,
)
    for k in 1:length(c)
        update_model_covariate!(c[k], d.designpoint[k], m, cp)
    end
    # When the information matrix is singular, the objective function is undefined. Lower
    # level calls may throw a PosDefException. This also means that `d` can not be a
    # solution to the maximization problem, hence we return negative infinity in these
    # cases.
    try
        acc = 0
        for i in 1:length(pk.p)
            informationmatrix!(wm.r_x_r, wm.m_x_r, d.weight, m, invcov(m), c, pk.p[i], na)
            _, is_inv = apply_transformation!(wm, false, tc, i)
            acc += pk.weight[i] * criterion_integrand!(wm.t_x_t, is_inv, dc)
        end
        return acc
    catch e
        if isa(e, PosDefException)
            return (-Inf)
        else
            rethrow(e)
        end
    end
end

"""
    objective(d::DesignMeasure, dp::DesignProblem)

Objective function corresponding to the [`DesignProblem`](@ref) evaluated at `d`.
"""
function objective(d::DesignMeasure, dp::DesignProblem)
    tc = precalculate_trafo_constants(dp.trafo, dp.pk)
    wm = WorkMatrices(unit_length(dp.m), parameter_dimension(dp.pk), codomain_dimension(tc))
    c = allocate_initialize_covariates(d, dp.m, dp.cp)
    return objective!(wm, c, dp.dc, d, dp.m, dp.cp, dp.pk, tc, dp.na)
end

function inverse_information_matrices(
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::DiscretePrior,
    na::NormalApproximation,
)
    # Calculate inverse of normalized information matrix for each parameter value.
    # Only the upper triangle of the (symmetric) information matrix is relevant.
    c = allocate_initialize_covariates(d, m, cp)
    # the transformed parameter dimension is a dummy argument here
    wm = WorkMatrices(unit_length(m), parameter_dimension(pk), 1)
    ic = invcov(m)
    # The documentation of `potri!` is not very clear that it expects a Cholesky factor as
    # input, and does _not_ call `potrf!` itself.
    # See also https://netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga9dfc04beae56a3b1c1f75eebc838c14c.html
    inv_nims = map(pk.p) do p
        informationmatrix!(wm.r_x_r, wm.m_x_r, d.weight, m, ic, c, p, na)
        potrf!('U', wm.r_x_r)
        potri!('U', wm.r_x_r)
        return deepcopy(wm.r_x_r)
    end
    return inv_nims
end

function gateauxderivative!(
    wm::WorkMatrices,
    c::AbstractVector{<:Covariate}, # only one element, but passed to `informationmatrix!`
    gconst::GateauxConstants,
    direction::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::DiscretePrior,
    na::NormalApproximation,
)
    update_model_covariate!(c[1], direction.designpoint[1], m, cp)
    acc = 0
    ic = invcov(m)
    for i in 1:length(pk.p)
        informationmatrix!(wm.r_x_r, wm.m_x_r, direction.weight, m, ic, c, pk.p[i], na)
        acc += pk.weight[i] * gateaux_integrand(gconst, wm.r_x_r, i)
    end
    return acc
end

"""
    gateauxderivative(at::DesignMeasure,
                      directions::AbstractArray{DesignMeasure},
                      dp::DesignProblem)

Gateaux derivative of the [`objective`](@ref) function corresponding to `dp`,
`at` the the given design measure into each of the `directions`,
which must be one-point designs.
"""
function gateauxderivative(
    at::DesignMeasure,
    directions::AbstractArray{DesignMeasure},
    dp::DesignProblem,
)
    if any(d -> length(d.weight) != 1, directions)
        error("Gateaux derivatives are only implemented for one-point design directions")
    end
    tc = precalculate_trafo_constants(dp.trafo, dp.pk)
    wm = WorkMatrices(unit_length(dp.m), parameter_dimension(dp.pk), codomain_dimension(tc))
    gconst = try
        precalculate_gateaux_constants(dp.dc, at, dp.m, dp.cp, dp.pk, tc, dp.na)
    catch e
        if isa(e, SingularException)
            # undefined objective implies no well-defined derivative
            return fill(NaN, size(directions))
        else
            rethrow(e)
        end
    end
    cs = allocate_initialize_covariates(directions[1], dp.m, dp.cp)
    gd = map(directions) do d
        gateauxderivative!(wm, cs, gconst, d, dp.m, dp.cp, dp.pk, dp.na)
    end
    return gd
end
