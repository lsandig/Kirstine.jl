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
    tpardim = codomain_dimension(trafo, pk)
    nim = zeros(pardim, pardim)
    tnim = zeros(tpardim, tpardim)
    jm = zeros(unit_length(m), pardim)
    c = allocate_initialize_covariates(candidate, m, cp)
    f = d -> objective!(tnim, nim, jm, c, dc, d, m, cp, pk, trafo, na)
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
        dir_cand = map(singleton_design, support(res))
        inv_nim = inverse_information_matrices(res, m, cp, pk, na)
        inmB = calc_inv_nim_at_mul_B(dc, pk, trafo, inv_nim)
        # find direction of steepest ascent
        gd(d) = gateauxderivative!(nim, jm, c, dc, inv_nim, inmB, d, m, cp, pk, trafo, na)
        or_gd = optimize(od, gd, dir_cand, constraints; trace_state = trace_state)
        push!(ors_d, or_gd)
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

function parameter_dimension(pk::PriorSample)
    return length(pk.p[1])
end

function parameter_dimension(pk::PriorGuess)
    return length(pk.p)
end

function parameter_dimension(pk::DiscretePrior)
    return length(pk.p[1])
end

function codomain_dimension(trafo::Identity, pk)
    return parameter_dimension(pk)
end

function informationmatrix!(
    nim::AbstractMatrix,
    jm::AbstractMatrix,
    w::AbstractVector,
    m::NonlinearRegression,
    invcov::Real,
    c::AbstractVector{<:Covariate},
    p,
    na::MLApproximation,
)
    im_helper!(nim, jm, w, m, invcov, c, p)
    return nim
end

function informationmatrix!(
    nim::AbstractMatrix,
    jm::AbstractMatrix,
    w::AbstractVector,
    m::NonlinearRegression,
    invcov::Real,
    c::AbstractVector{<:Covariate},
    p,
    na::MAPApproximation,
)
    im_helper!(nim, jm, w, m, invcov, c, p)
    nim .+= na.scaled_prior_precision
    return nim
end

function im_helper!(nim, jm, w, m, invcov, c, p)
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

function apply_transformation!(tnim, nim, is_inv::Bool, trafo::Identity, index::Int64)
    tnim .= nim
    return tnim, is_inv
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

# == objective function helpers for each type of `PriorKnowledge` == #
function obj_integral(tnim, nim, jm, dc, w, m, c, pk::DiscretePrior, trafo, na)
    acc = 0
    for i in 1:length(pk.p)
        informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p[i], na)
        _, is_inv = apply_transformation!(tnim, nim, false, trafo, i)
        acc += pk.weight[i] * criterion_integrand!(tnim, is_inv, dc)
    end
    return acc
end

function obj_integral(tnim, nim, jm, dc, w, m, c, pk::PriorSample, trafo, na)
    acc = 0
    n = length(pk.p)
    for i in 1:n
        informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p[i], na)
        _, is_inv = apply_transformation!(tnim, nim, false, trafo, i)
        acc += criterion_integrand!(tnim, is_inv, dc)
    end
    return acc / n
end

function obj_integral(tnim, nim, jm, dc, w, m, c, pk::PriorGuess, trafo, na)
    informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p, na)
    _, is_inv = apply_transformation!(tnim, nim, false, trafo, 1)
    return criterion_integrand!(tnim, is_inv, dc)
end

function objective!(
    tnim::AbstractMatrix,
    nim::AbstractMatrix,
    jm::AbstractMatrix,
    c::AbstractVector{<:Covariate},
    dc::DesignCriterion,
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
    na::NormalApproximation,
)
    for k in 1:length(c)
        update_model_covariate!(c[k], d.designpoint[k], m, cp)
    end
    return obj_integral(tnim, nim, jm, dc, d.weight, m, c, pk, trafo, na)
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
    tpardim = codomain_dimension(trafo, pk)
    nim = zeros(pardim, pardim)
    tnim = zeros(tpardim, tpardim)
    jm = zeros(unit_length(m), pardim)
    c = allocate_initialize_covariates(d, m, cp)
    return objective!(tnim, nim, jm, c, dc, d, m, cp, pk, trafo, na)
end

# == Gateaux derivative function helpes for each type of `PriorKnowledge` == #
function allocate_infomatrices(q, jm, w, m, invcov, c, pk::DiscretePrior, na)
    return [informationmatrix!(zeros(q, q), jm, w, m, invcov, c, p, na) for p in pk.p]
end

function allocate_infomatrices(q, jm, w, m, invcov, c, pk::PriorSample, na)
    return [informationmatrix!(zeros(q, q), jm, w, m, invcov, c, p, na) for p in pk.p]
end

function allocate_infomatrices(q, jm, w, m, invcov, c, pk::PriorGuess, na)
    return [informationmatrix!(zeros(q, q), jm, w, m, invcov, c, pk.p, na)]
end

function inverse_information_matrices(
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    na::NormalApproximation,
)
    # Calculate inverse of normalized information matrix for each parameter Note: the
    # jacobian matrix is re-usable, while the FisherMatrix needs to be allocated anew each
    # time. It is modified in-place by potri!. Only the upper triangle of the (symmetric)
    # information matrix is relevant.
    c = allocate_initialize_covariates(d, m, cp)
    q = parameter_dimension(pk)
    jm = zeros(unit_length(m), q)
    nims = allocate_infomatrices(q, jm, d.weight, m, invcov(m), c, pk, na)
    # The documentation of `potri!` is not very clear that it expects a Cholesky factor as
    # input, and does _not_ call `potrf!` itself.
    # See also https://netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga9dfc04beae56a3b1c1f75eebc838c14c.html
    map(M -> potrf!('U', M), nims)
    inv_nims = [potri!('U', M) for M in nims]
    return inv_nims
end

#! format: off
function gd_integral(nim, jm, c, dc, w, inv_nim_at, inv_nim_at_mul_B, m,
                     pk::DiscretePrior, trafo, na)
#! format: on
    acc = 0
    n = length(pk.p)
    for i in 1:n
        informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p[i], na)
        acc +=
            pk.weight[i] *
            gateaux_integrand(dc, inv_nim_at[i], inv_nim_at_mul_B[i], nim, trafo)
    end
    return acc
end

#! format: off
function gd_integral(nim, jm, c, dc, w, inv_nim_at, inv_nim_at_mul_B, m,
                     pk::PriorSample, trafo, na)
#! format: on
    acc = 0
    n = length(pk.p)
    for i in 1:n
        informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p[i], na)
        acc += gateaux_integrand(dc, inv_nim_at[i], inv_nim_at_mul_B[i], nim, trafo)
    end
    return acc / n
end

#! format: off
function gd_integral(nim, jm, c, dc, w, inv_nim_at, inv_nim_at_mul_B, m,
                     pk::PriorGuess, trafo, na)
#! format: on
    informationmatrix!(nim, jm, w, m, invcov(m), c, pk.p, na)
    return gateaux_integrand(dc, inv_nim_at[1], inv_nim_at_mul_B[1], nim, trafo)
end

function gateauxderivative!(
    nim::AbstractMatrix,
    jm::AbstractMatrix,
    c::AbstractVector{<:Covariate}, # only one element, but passed to `informationmatrix!`
    dc::DesignCriterion,
    inv_nim_at::AbstractVector{<:AbstractMatrix},
    inv_nim_at_mul_B::AbstractVector{<:AbstractMatrix},
    direction::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
    na::NormalApproximation,
)
    update_model_covariate!(c[1], direction.designpoint[1], m, cp)
    #! format: off
    return gd_integral(nim, jm, c, dc, direction.weight, inv_nim_at, inv_nim_at_mul_B,
                       m, pk, trafo, na)
    #! format: on
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
    inv_nim_at = inverse_information_matrices(at, m, cp, pk, na)
    inv_nim_at_mul_B = calc_inv_nim_at_mul_B(dc, pk, trafo, inv_nim_at)
    cs = allocate_initialize_covariates(directions[1], m, cp)
    gd = map(directions) do d
        #! format: off
        gateauxderivative!(nim, jm, cs, dc, inv_nim_at, inv_nim_at_mul_B, d,
                           m, cp, pk, trafo, na)
        #! format: on
    end
    return gd
end
