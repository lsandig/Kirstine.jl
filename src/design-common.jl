# generic design objective functions and Gateaux derivatives

# declare functions to be redefined for user models
#
# (m::NonlinearRegression) -> Real
function unit_length end
# (m::NonlinearRegression, K::Integer) -> Vector{<:Covariate}
function allocate_covariates end
# (jm::AbstractMatrix, c::Covariate, m::NonlinearRegression, p) -> jm
function update_jacobian_matrix! end
# (c::Covariate, cp::CovariateParameterization, dp::AbstractVector{<:Real}, m::NonlinearRegression) -> c
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
                    trafo::Transformation;
                    candidate::DesignMeasure = random_design(parameter_dimension(pk), ds),
                    fixedweights = Int64[],
                    fixedpoints = Int64[],
                    trace_state = false)

Find an optimal experimental design for the nonlinear regression model `m`.

One particle of the [`Optimizer`](@ref) is initialized at `candidate`, the
remaining ones are randomized. Any weight or design point corresponding to an
index given in `fixedweights` or `fixedpoints` is not randomized and is kept
fixed during optimization. This can speed up computation if some weights or
points are known analytically.

Returns an [`OptimizationResult`](@ref). If `trace_state=true`, the full state
of the algorithm is saved for every iteration, which can be useful for
debugging.
"""
function optimize_design(
    optimizer::Optimizer,
    dc::DesignCriterion,
    ds::DesignSpace,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation;
    candidate::DesignMeasure = random_design(parameter_dimension(pk), ds),
    fixedweights = Int64[],
    fixedpoints = Int64[],
    trace_state = false,
)
    pardim = parameter_dimension(pk)
    nim = zeros(pardim, pardim)
    jm = zeros(unit_length(m), pardim)
    c = allocate_initialize_covariates(candidate, m, cp)
    f = d -> objective!(nim, jm, c, dc, d, m, cp, pk, trafo)
    # transform index lists into Bool vectors
    K = length(c)
    if any(fixedweights .< 1) || any(fixedweights .> K)
        error("indices for fixed weights must be between 1 and $K")
    end
    if any(fixedpoints .< 1) || any(fixedpoints .> K)
        error("indices for fixed points must be between 1 and $K")
    end
    fixw = [k in fixedweights for k in 1:K]
    fixp = [k in fixedpoints for k in 1:K]
    constraints = (ds, fixw, fixp)
    return optimize(optimizer, f, [candidate], constraints; trace_state = trace_state)
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
                  trafo::Transformation;
                  simpargs...,)

Improve a `candidate` design by ascending its Gateaux derivative.

This repeats the following `steps` times, starting with `r = candidate`:

 1. [`simplify`](@ref) the design `r`, passing along `simpargs`.
 2. Find the direction (Dirac measure) `d` of steepest
    [`gateauxderivative`](@ref) at `r` using the [`Optimizer`](@ref) `od`.
 3. Re-calculcate optimal weights of a [`mixture`](@ref) of `r` and `d` for the
    [`DesignCriterion`](@ref) objective using the optimizer `ow`.
 4. Set `r` to the result of step 3.

Returns a [`DesignMeasure`](@ref).
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
    trafo::Transformation;
    simpargs...,
)
    pardim = parameter_dimension(pk)
    nim = zeros(pardim, pardim)
    jm = zeros(unit_length(m), pardim)
    c = allocate_initialize_covariates(singleton_design(candidate.designpoint[1]), m, cp)
    constraints = (ds, [false], [false])

    res = candidate
    for i in 1:steps
        res = simplify(res, ds, m, cp; simpargs...)
        dir_cand = map(singleton_design, support(res))
        inv_nim = inverse_information_matrices(res, m, cp, pk)
        # find direction of steepest ascent
        gd(d) = gateauxderivative!(nim, jm, c, dc, inv_nim, d, m, cp, pk, trafo)
        or_gd = optimize(od, gd, dir_cand, constraints)
        d = or_gd.maximizer
        # append the new atom
        K = length(res.weight)
        if d.designpoint[1] in support(res)
            # effectivly run the reweighting from the last round for some more iterations
            res = mixture(0, d, res) # make sure new point is at index 1
            res = simplify_merge(res, ds, 0)
        else
            K += 1
            res = mixture(1 / K, d, res)
        end
        # optimize weights
        or_w = optimize_design(
            ow,
            dc,
            ds,
            m,
            cp,
            pk,
            trafo;
            candidate = res,
            fixedpoints = 1:K,
        )
        res = or_w.maximizer
    end
    return simplify(res, ds, m, cp; simpargs...)
end

# == various helper functions == #

function parameter_dimension(pk::PriorSample)
    return length(pk.p[1])
end

function parameter_dimension(pk::PriorGuess)
    return length(pk.p)
end

function parameter_dimension(pk::DiscretePrior)
    return length(pk.p[1])
end

function informationmatrix!(
    nim::AbstractMatrix,
    jm::AbstractMatrix,
    w::AbstractVector,
    m::NonlinearRegression,
    invcov::Real,
    c::AbstractVector{<:Covariate},
    p,
)
    fill!(nim, 0.0)
    for k in 1:length(w)
        update_jacobian_matrix!(jm, c[k], m, p)
        # The call to syrk! is equivalent to
        #
        #   nim = w[k] * invcov * jm' * jm + 1 * nim
        #
        # The result is symmetric, and only the 'U'pper triangle of nim is actually written
        # to. The lower triangle is not touched and is allowed to contain arbitrary garbabe.
        syrk!('U', 'T', w[k] * invcov, jm, 1.0, nim)
    end
    return nim
end

function allocate_initialize_covariates(d, m, cp)
    K = length(d.weight)
    cs = allocate_covariates(m, K)
    for k in 1:K
        update_model_covariate!(cs[k], cp, d.designpoint[k], m)
    end
    return cs
end

# Calculate the Frobenius inner product tr(A*B). Both matrices are implicitly treated as
# symmetric, i.e. only the uppr triangles are used.
function tr_prod(A::AbstractMatrix, B::AbstractMatrix)
    if size(A) != size(B)
        error("A and B must have identical size")
    end
    k = size(A, 1)
    acc = 0.0
    for i in 1:k
        acc += A[i, i] * B[i, i]
        for j in (i + 1):k
            acc += 2 * A[i, j] * B[i, j]
        end
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

# == objective function helpes for each type of `PriorKnowledge` == #
function obj_integral(nim, jm, dc, w, m, c, pk::DiscretePrior, trafo)
    acc = 0
    for i in 1:length(pk.p)
        informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p[i])
        acc += pk.weight[i] * criterion_integrand!(nim, dc, trafo)
    end
    return acc
end

function obj_integral(nim, jm, dc, w, m, c, pk::PriorSample, trafo)
    acc = 0
    n = length(pk.p)
    for i in 1:n
        informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p[i])
        acc += criterion_integrand!(nim, dc, trafo)
    end
    return acc / n
end

function obj_integral(nim, jm, dc, w, m, c, pk::PriorGuess, trafo)
    informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p)
    return criterion_integrand!(nim, dc, trafo)
end

function objective!(
    nim::AbstractMatrix,
    jm::AbstractMatrix,
    c::AbstractVector{<:Covariate},
    dc::DesignCriterion,
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
)
    for k in 1:length(c)
        update_model_covariate!(c[k], cp, d.designpoint[k], m)
    end
    return obj_integral(nim, jm, dc, d.weight, m, c, pk, trafo)
end

"""
    objective(dc::DesignCriterion,
              d::DesignMeasure,
              m::NonlinearRegression,
              cp::CovariateParameterization,
              pk::PriorKnowledge,
              trafo::Transformation,
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
)
    pardim = parameter_dimension(pk)
    nim = zeros(pardim, pardim)
    jm = zeros(unit_length(m), pardim)
    c = allocate_initialize_covariates(d, m, cp)
    return objective!(nim, jm, c, dc, d, m, cp, pk, trafo)
end

# == Gateaux derivative function helpes for each type of `PriorKnowledge` == #
function allocate_infomatrices(q, jm, w, m, invcov, c, pk::DiscretePrior)
    return [informationmatrix!(zeros(q, q), jm, w, m, invcov, c, p) for p in pk.p]
end

function allocate_infomatrices(q, jm, w, m, invcov, c, pk::PriorSample)
    return [informationmatrix!(zeros(q, q), jm, w, m, invcov, c, p) for p in pk.p]
end

function allocate_infomatrices(q, jm, w, m, invcov, c, pk::PriorGuess)
    return [informationmatrix!(zeros(q, q), jm, w, m, invcov, c, pk.p)]
end

function inverse_information_matrices(
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
)
    # Calculate inverse of normalized information matrix for each parameter Note: the
    # jacobian matrix is re-usable, while the FisherMatrix needs to be allocated anew each
    # time. It is modified in-place by potri!. Only the upper triangle of the (symmetric)
    # information matrix is relevant.
    c = allocate_initialize_covariates(d, m, cp)
    q = parameter_dimension(pk)
    jm = zeros(unit_length(m), q)
    nims = allocate_infomatrices(q, jm, d.weight, m, invcov(m), c, pk)
    # The documentation of `potri!` is not very clear that it expects a Cholesky factor as
    # input, and does _not_ call `potrf!` itself.
    # See also https://netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga9dfc04beae56a3b1c1f75eebc838c14c.html
    map(M -> potrf!('U', M), nims)
    inv_nims = [potri!('U', M) for M in nims]
    return inv_nims
end

function gd_integral(nim, jm, c, dc, w, inv_nim_at, m, pk::DiscretePrior, trafo)
    acc = 0
    n = length(pk.p)
    for i in 1:n
        informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p[i])
        acc += pk.weight[i] * gateaux_integrand(inv_nim_at[i], nim, dc, trafo)
    end
    return acc
end

function gd_integral(nim, jm, c, dc, w, inv_nim_at, m, pk::PriorSample, trafo)
    acc = 0
    n = length(pk.p)
    for i in 1:n
        informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p[i])
        acc += gateaux_integrand(inv_nim_at[i], nim, dc, trafo)
    end
    return acc / n
end

function gd_integral(nim, jm, c, dc, w, inv_nim_at, m, pk::PriorGuess, trafo)
    informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p)
    return gateaux_integrand(inv_nim_at[1], nim, dc, trafo)
end

function gateauxderivative!(
    nim::AbstractMatrix,
    jm::AbstractMatrix,
    c::AbstractVector{<:Covariate}, # only one element, but passed to `informationmatrix!`
    dc::DesignCriterion,
    inv_nim_at::AbstractVector{<:AbstractMatrix},
    direction::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
)
    update_model_covariate!(c[1], cp, direction.designpoint[1], m)
    return gd_integral(nim, jm, c, dc, direction.weight, inv_nim_at, m, pk, trafo)
end

"""
    gateauxderivative(dc::DesignCriterion,
                      at::DesignMeasure,
                      directions::AbstractArray{DesignMeasure},
                      m::NonlinearRegression,
                      cp::CovariateParameterization,
                      pk::PriorKnowledge,
                      trafo::Transformation,
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
)
    if any(d -> length(d.weight) != 1, directions)
        error("Gateaux derivatives are only implemented for singleton design directions")
    end
    pardim = parameter_dimension(pk)
    nim = zeros(pardim, pardim)
    jm = zeros(unit_length(m), pardim)
    inv_nim_at = inverse_information_matrices(at, m, cp, pk)
    cs = allocate_initialize_covariates(directions[1], m, cp)
    gd = map(directions) do d
        gateauxderivative!(nim, jm, cs, dc, inv_nim_at, d, m, cp, pk, trafo)
    end
    return gd
end
