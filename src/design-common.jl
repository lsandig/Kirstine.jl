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

# == main interface == #

"""
    optimize_design(optimizer::Optimizer,
                    dc::DesignCriterion,
                    ds::DesignSpace,
                    m::NonlinearRegression,
                    cp::CovariateParameterization,
                    pk::PriorKnowledge,
                    trafo::Transformation,
                    na::NormalApproximation;
                    candidate::DesignMeasure = random_design(ds, parameter_dimension(pk)),
                    fixedweights = Int64[],
                    fixedpoints = Int64[],
                    trace_state = false,
                    sargs...)

Find an optimal experimental design for the nonlinear regression model `m`.

One particle of the [`Optimizer`](@ref) is initialized at `candidate`, the
remaining ones are randomized. Any weight or design point corresponding to an
index given in `fixedweights` or `fixedpoints` is not randomized and is kept
fixed during optimization. This can speed up computation if some weights or
points are known analytically.

Returns a Tuple:

  - The best [`DesignMeasure`](@ref) found. As postprocessing, [`simplify`](@ref) is called
    with `sargs` and the design points are sorted with [`sort_designpoints`](@ref).

  - The full [`OptimizationResult`](@ref). If `trace_state=true`, the full state of the
    algorithm is saved for every iteration, which can be useful for debugging.
"""
function optimize_design(
    optimizer::Optimizer,
    dc::DesignCriterion,
    ds::DesignSpace,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
    na::NormalApproximation;
    candidate::DesignMeasure = random_design(ds, parameter_dimension(pk)),
    fixedweights = Int64[],
    fixedpoints = Int64[],
    trace_state = false,
    sargs...,
)
    check_compatible(candidate, ds)
    pardim = parameter_dimension(pk)
    tc = precalculate_trafo_constants(trafo, pk)
    tpardim = codomain_dimension(tc)
    nim = zeros(pardim, pardim)
    work = zeros(pardim, tpardim)
    tnim = zeros(tpardim, tpardim)
    jm = zeros(unit_length(m), pardim)
    c = allocate_initialize_covariates(candidate, m, cp)
    f = d -> objective!(tnim, work, nim, jm, c, dc, d, m, cp, pk, tc, na)
    K = length(c)
    # set up constraints
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
        fixw .= true
    end
    constraints = (ds, fixw, fixp)
    or = optimize(optimizer, f, [candidate], constraints; trace_state = trace_state)
    dopt = sort_designpoints(simplify(or.maximizer, ds, m, cp; sargs...))
    return dopt, or
end

"""
    refine_design(od::Optimizer,
                  ow::Optimizer,
                  steps::Int64,
                  candidate::DesignMeasure,
                  dc::DesignCriterion,
                  ds::DesignSpace,
                  m::NonlinearRegression,
                  cp::CovariateParameterization,
                  pk::PriorKnowledge,
                  trafo::Transformation,
                  na::NormalApproximation;
                  trace_state = false,
                  sargs...)

Improve a `candidate` design by ascending its Gateaux derivative.

This repeats the following `steps` times, starting with `r = candidate`:

 1. [`simplify`](@ref) the design `r`, passing along `sargs`.
 2. Find the direction (Dirac measure) `d` of steepest
    [`gateauxderivative`](@ref) at `r` using the [`Optimizer`](@ref) `od`.
 3. Re-calculcate optimal weights of a [`mixture`](@ref) of `r` and `d` for the
    [`DesignCriterion`](@ref) objective using the optimizer `ow`.
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
    dc::DesignCriterion,
    ds::DesignSpace,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
    na::NormalApproximation;
    trace_state = false,
    sargs...,
)
    check_compatible(candidate, ds)
    tc = precalculate_trafo_constants(trafo, pk)
    pardim = parameter_dimension(pk)
    nim = zeros(pardim, pardim)
    jm = zeros(unit_length(m), pardim)
    c = allocate_initialize_covariates(singleton_design(candidate.designpoint[1]), m, cp)
    ors_d = OptimizationResult[]
    ors_w = OptimizationResult[]
    constraints = (ds, [false], [false])

    res = candidate
    for i in 1:steps
        res = simplify(res, ds, m, cp; sargs...)
        dir_cand = map(singleton_design, designpoints(simplify_drop(res, 0)))
        gconst = precalculate_gateaux_constants(dc, res, m, cp, pk, tc, na)
        # find direction of steepest ascent
        gd(d) = gateauxderivative!(nim, jm, c, gconst, d, m, cp, pk, na)
        or_gd = optimize(od, gd, dir_cand, constraints; trace_state = trace_state)
        push!(ors_d, or_gd)
        d = or_gd.maximizer
        # append the new atom
        K = length(res.weight)
        if d.designpoint[1] in designpoints(simplify_drop(res, 0))
            # effectivly run the reweighting from the last round for some more iterations
            res = mixture(0, d, res) # make sure new point is at index 1
            res = simplify_merge(res, ds, 0)
        else
            K += 1
            res = mixture(1 / K, d, res)
        end
        # optimize weights
        _, or_w = optimize_design(
            ow,
            dc,
            ds,
            m,
            cp,
            pk,
            trafo,
            na;
            candidate = res,
            fixedpoints = 1:K,
            trace_state = trace_state,
            sargs...,
        )
        push!(ors_w, or_w)
        res = or_w.maximizer
    end
    dopt = sort_designpoints(simplify(res, ds, m, cp; sargs...))
    return dopt, ors_d, ors_w
end

# == various helper functions == #

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
    return length(pk.p[1])
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
    p,
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

# We just copy nim to tnim and don't care whether it is inverted or not.
function apply_transformation!(tnim, _, nim, is_inv::Bool, tc::TCIdentity, index)
    tnim .= nim
    return tnim, is_inv
end

# Notes:
#  * The first _three_ arguments are modified by this function.
#  * When trafo maps from ℝ^r to ℝ^t, the preallocated matrices need to have dimensions
#    - tnim: (t, t)
#    - work: (r, t)
#    - nim: (r, r)
#  * We always return a dense _inverse_ of the transformed information matrix.
function apply_transformation!(tnim, work, nim, is_inv::Bool, tc::TCDeltaMethod, index)
    if is_inv
        # update work = 1 * Symmetric(nim) * tc.jm[index]' + 0 * work, i.e.
        #  * nim on the 'L'eft of the multiplication
        #  * and only use its 'U'pper triangle.
        symm!('L', 'U', 1.0, nim, permutedims(tc.jm[index]), 0.0, work)
    else
        work .= tc.jm[index]'
        # calculate inv(Symmetric(nim)) * tc.jm[index]' by solving
        # Symmetric(nim) * X = tc.jm[index]' for X
        posv!('U', nim, work) # overwrites nim with potrf!(nim) and work with the solution
    end
    # set tnim to tc.jm[index] * work
    mul!(tnim, tc.jm[index], work)
    return tnim, true
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
    tnim::AbstractMatrix,
    work::AbstractMatrix,
    nim::AbstractMatrix,
    jm::AbstractMatrix,
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
            informationmatrix!(nim, jm, d.weight, m, invcov(m), c, pk.p[i], na)
            _, is_inv = apply_transformation!(tnim, work, nim, false, tc, i)
            acc += pk.weight[i] * criterion_integrand!(tnim, is_inv, dc)
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
    objective(dc::DesignCriterion,
              d::DesignMeasure,
              m::NonlinearRegression,
              cp::CovariateParameterization,
              pk::PriorKnowledge,
              trafo::Transformation,
              na::NormalApproximation,
              )

Objective function corresponding to the [`DesignCriterion`](@ref) evaluated at `d`.
"""
function objective(
    dc::DesignCriterion,
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
    na::NormalApproximation,
)
    pardim = parameter_dimension(pk)
    tc = precalculate_trafo_constants(trafo, pk)
    tpardim = codomain_dimension(tc)
    nim = zeros(pardim, pardim)
    work = zeros(pardim, tpardim)
    tnim = zeros(tpardim, tpardim)
    jm = zeros(unit_length(m), pardim)
    c = allocate_initialize_covariates(d, m, cp)
    return objective!(tnim, work, nim, jm, c, dc, d, m, cp, pk, tc, na)
end

function inverse_information_matrices(
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::DiscretePrior,
    na::NormalApproximation,
)
    # Calculate inverse of normalized information matrix for each parameter Note: the
    # jacobian matrix is re-usable, while the FisherMatrix needs to be allocated anew each
    # time. It is modified in-place by potri!. Only the upper triangle of the (symmetric)
    # information matrix is relevant.
    c = allocate_initialize_covariates(d, m, cp)
    q = parameter_dimension(pk)
    jm = zeros(unit_length(m), q)
    ic = invcov(m)
    nims = [informationmatrix!(zeros(q, q), jm, d.weight, m, ic, c, p, na) for p in pk.p]
    # The documentation of `potri!` is not very clear that it expects a Cholesky factor as
    # input, and does _not_ call `potrf!` itself.
    # See also https://netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga9dfc04beae56a3b1c1f75eebc838c14c.html
    map(M -> potrf!('U', M), nims)
    inv_nims = [potri!('U', M) for M in nims]
    return inv_nims
end

function gateauxderivative!(
    nim::AbstractMatrix,
    jm::AbstractMatrix,
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
    n = length(pk.p)
    for i in 1:n
        informationmatrix!(nim, jm, direction.weight, m, invcov(m), c, pk.p[i], na)
        acc += pk.weight[i] * gateaux_integrand(gconst, nim, i)
    end
    return acc
end

"""
    gateauxderivative(dc::DesignCriterion,
                      at::DesignMeasure,
                      directions::AbstractArray{DesignMeasure},
                      m::NonlinearRegression,
                      cp::CovariateParameterization,
                      pk::PriorKnowledge,
                      trafo::Transformation,
                      na::NormalApproximation,
                      )

Gateaux derivative of the [`objective`](@ref) function `at` the the given design measure
into each of the `directions`, which must be singleton designs.
"""
function gateauxderivative(
    dc::DesignCriterion,
    at::DesignMeasure,
    directions::AbstractArray{DesignMeasure},
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
    na::NormalApproximation,
)
    if any(d -> length(d.weight) != 1, directions)
        error("Gateaux derivatives are only implemented for singleton design directions")
    end
    pardim = parameter_dimension(pk)
    nim = zeros(pardim, pardim)
    jm = zeros(unit_length(m), pardim)
    tc = precalculate_trafo_constants(trafo, pk)
    gconst = try
        precalculate_gateaux_constants(dc, at, m, cp, pk, tc, na)
    catch e
        if isa(e, SingularException)
            # undefined objective implies no well-defined derivative
            return fill(NaN, size(directions))
        else
            rethrow(e)
        end
    end
    cs = allocate_initialize_covariates(directions[1], m, cp)
    gd = map(directions) do d
        gateauxderivative!(nim, jm, cs, gconst, d, m, cp, pk, na)
    end
    return gd
end
