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

function gd_integral(nim_x, jm, dc, w, inv_nim_candidate, m, c, pk::DiscretePrior, trafo)
    acc = 0
    n = length(pk.p)
    for i in 1:n
        informationmatrix!(nim_x, jm, w, m, invcov(m), c, pk.p[i])
        acc += pk.weight[i] * gateaux_integrand(nim_x, inv_nim_candidate[i], dc, trafo)
    end
    return acc
end

function gd_integral(nim_x, jm, dc, w, inv_nim_candidate, m, c, pk::PriorSample, trafo)
    acc = 0
    n = length(pk.p)
    for i in 1:n
        informationmatrix!(nim_x, jm, w, m, invcov(m), c, pk.p[i])
        acc += gateaux_integrand(nim_x, inv_nim_candidate[i], dc, trafo)
    end
    return acc / n
end

function gd_integral(nim_x, jm, dc, w, inv_nim_candidate, m, c, pk::PriorGuess, trafo)
    informationmatrix!(nim_x, jm, w, m, invcov(m), c, pk.p)
    return gateaux_integrand(nim_x, inv_nim_candidate[1], dc, trafo)
end

function gateauxderivative!(
    nim_x::AbstractMatrix,
    jm::AbstractMatrix,
    dc::DesignCriterion,
    dirac::DesignMeasure,
    inv_nim_candidate::AbstractVector{<:AbstractMatrix},
    m::NonlinearRegression,
    c::AbstractVector{<:Covariate}, # passed through to informationmatrix!
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
)
    update_model_covariate!(c[1], cp, dirac.designpoint[1], m)
    return gd_integral(nim_x, jm, dc, dirac.weight, inv_nim_candidate, m, c, pk, trafo)
end

"""
    gateauxderivative(dc::DesignCriterion,
                      x::AbstractArray{DesignMeasure},
                      candidate::DesignMeasure,
                      m::NonlinearRegression,
                      cp::CovariateParameterization,
                      pk::PriorKnowledge,
                      trafo::Transformation,
                      )

Gateaux derivative of the [`objective`](@ref) at the `candidate` design measure into the
direction of each of the singleton designs in `x`.
"""
function gateauxderivative(
    dc::DesignCriterion,
    x::AbstractArray{DesignMeasure},
    candidate::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorKnowledge,
    trafo::Transformation,
)
    pardim = parameter_dimension(pk)
    nim = zeros(pardim, pardim)
    jm = zeros(unit_length(m), pardim)
    inv_nim = inverse_information_matrices(candidate, m, cp, pk)
    cs = allocate_initialize_covariates(x[1], m, cp)
    gd = map(x) do d
        gateauxderivative!(nim, jm, dc, d, inv_nim, m, cs, cp, pk, trafo)
    end
    return gd
end
