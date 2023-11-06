# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## preallocating objects for less memory overhead ##

# preallocated matrices with dimensions that are often used together
struct WorkMatrices
    r_x_r::Matrix{Float64}
    r_x_t::Matrix{Float64}
    t_x_r::Matrix{Float64}
    t_x_t::Matrix{Float64}
    m_x_r::Matrix{Float64}
    m_x_m::Vector{Matrix{Float64}}
    function WorkMatrices(K::Integer, m::Integer, r::Integer, t::Integer)
        mxm = [zeros(m, m) for _ in 1:K]
        new(zeros(r, r), zeros(r, t), zeros(t, r), zeros(t, t), zeros(m, r), mxm)
    end
end

function allocate_initialize_covariates(d, m, cp)
    K = numpoints(d)
    cs = [allocate_covariate(m) for _ in 1:K]
    for k in 1:K
        map_to_covariate!(cs[k], points(d)[k], m, cp)
    end
    return cs
end

## model and criterion agnostic objective and gateaux derivative ##

function objective!(
    wm::WorkMatrices,
    c::AbstractVector{<:Covariate},
    dc::DesignCriterion,
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorSample,
    tc::TrafoConstants,
    na::NormalApproximation,
)
    for k in 1:length(c)
        map_to_covariate!(c[k], points(d)[k], m, cp)
        update_model_vcov!(wm.m_x_m[k], c[k], m)
        potrf!('U', wm.m_x_m[k])
    end
    # When the information matrix is singular, the objective function is undefined. Lower
    # level calls may throw a PosDefException or a SingularException. This also means that
    # `d` can not be a solution to the maximization problem, hence we return negative
    # infinity in these cases.
    try
        acc = 0
        for i in 1:length(pk.p)
            informationmatrix!(wm.r_x_r, wm.m_x_r, weights(d), m, wm.m_x_m, c, pk.p[i], na)
            _, is_inv = apply_transformation!(wm, false, tc, i)
            acc += pk.weight[i] * criterion_integrand!(wm.t_x_t, is_inv, dc)
        end
        return acc
    catch e
        if isa(e, PosDefException) || isa(e, SingularException)
            return (-Inf)
        else
            rethrow(e)
        end
    end
end

function gateauxderivative!(
    wm::WorkMatrices,
    c::AbstractVector{<:Covariate}, # only one element, but passed to `informationmatrix!`
    gconst::GateauxConstants,
    direction::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorSample,
    na::NormalApproximation,
)
    map_to_covariate!(c[1], points(direction)[1], m, cp)
    update_model_vcov!(wm.m_x_m[1], c[1], m)
    potrf!('U', wm.m_x_m[1])
    acc = 0
    for i in 1:length(pk.p)
        informationmatrix!(
            wm.r_x_r,
            wm.m_x_r,
            weights(direction),
            m,
            wm.m_x_m,
            c,
            pk.p[i],
            na,
        )
        acc += pk.weight[i] * gateaux_integrand(gconst, wm.r_x_r, i)
    end
    return acc
end

## normalized information matrix for θ ##

function informationmatrix!(
    nim::AbstractMatrix,
    jm::AbstractMatrix,
    w::AbstractVector,
    m::NonlinearRegression,
    chol_vcov::AbstractVector{<:AbstractMatrix{<:Real}},
    c::AbstractVector{<:Covariate},
    p::Parameter,
    na::FisherMatrix,
)
    fill!(nim, 0.0)
    for k in 1:length(w)
        jacobianmatrix!(jm, m, c[k], p)
        # set jm = solution to `chol_vcov[k]' * X = 1 * jm`
        # i.e. triangular matrix is the 'L'eft factor, with data in 'U'pper triangle and
        # 'T'ransposed, but 'N'o unit diagonal
        trsm!('L', 'U', 'T', 'N', 1.0, chol_vcov[k], jm)
        # update nim = w[k] jm' * jm + 1 * nim
        # i.e. data in 'U'pper triangle and 'T'ransposed first
        syrk!('U', 'T', w[k], jm, 1.0, nim)
    end
    # Note: nim is implicitly symmetric, only the upper triangle contains data.
    return nim
end

# For debugging purposes, one will typically want to look at an information matrix
# corresponding to a single parameter value, not to all thousands of them in a prior sample.
"""
    informationmatrix(
        d::DesignMeasure,
        m::NonlinearRegression,
        cp::CovariateParameterization,
        p::Parameter,
        na::NormalApproximation,
    )

Compute the normalized information matrix for a single `Parameter` value `p`.

This function is useful for debugging.

See also the [mathematical background](math.md#Objective-Function).
"""
function informationmatrix(
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    p::Parameter,
    na::NormalApproximation,
)
    c = Kirstine.allocate_initialize_covariates(d, m, cp)
    # Note: t = 1 is a dummy value, no trafo will be applied
    wm = WorkMatrices(numpoints(d), unit_length(m), dimension(p), 1)
    for k in 1:length(c)
        Kirstine.update_model_vcov!(wm.m_x_m[k], c[k], m)
        potrf!('U', wm.m_x_m[k])
    end
    informationmatrix!(wm.r_x_r, wm.m_x_r, weights(d), m, wm.m_x_m, c, p, na)
    return Symmetric(wm.r_x_r)
end

function inverse_information_matrices(
    d::DesignMeasure,
    m::NonlinearRegression,
    cp::CovariateParameterization,
    pk::PriorSample,
    na::NormalApproximation,
)
    # Calculate inverse of normalized information matrix for each parameter value.
    # Only the upper triangle of the (symmetric) information matrix is relevant.
    c = allocate_initialize_covariates(d, m, cp)
    # The transformed parameter dimension t = 1 is a dummy argument here.
    wm = WorkMatrices(length(weights(d)), unit_length(m), parameter_dimension(pk), 1)
    for k in 1:length(c)
        Kirstine.update_model_vcov!(wm.m_x_m[k], c[k], m)
        potrf!('U', wm.m_x_m[k])
    end
    # The documentation of `potri!` is not very clear that it expects a Cholesky factor as
    # input, and does _not_ call `potrf!` itself.
    # See also https://netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga9dfc04beae56a3b1c1f75eebc838c14c.html
    inv_nims = map(pk.p) do p
        informationmatrix!(wm.r_x_r, wm.m_x_r, weights(d), m, wm.m_x_m, c, p, na)
        potrf!('U', wm.r_x_r)
        potri!('U', wm.r_x_r)
        return deepcopy(wm.r_x_r)
    end
    return inv_nims
end

## normalized information matrix for T(θ) ##

# Calling conventions for `apply_transformation!`
#
# * `wm.r_x_r` holds the information matrix or its inverse, depending on the `is_inv` flag.
# * Only the upper triangle of `wm.r_x_r` is used.
# * `wm.t_x_t` will be overwritten with the transformed information matrix or its inverse.
# * The return value are `wm` and a flag that indicates whether `wm.t_x_t` it is inverted.
# * Only the upper triangle of `wm.t_x_t` is guaranteed to make sense, but specific methods
#   are free to return a dense matrix.
# * Whether the returned matrix will be inverted is _not_ controlled by `is_inv`.

function apply_transformation!(wm::WorkMatrices, is_inv::Bool, tc::TCIdentity, index)
    # For the Identity transformation we just pass through the information matrix.
    wm.t_x_t .= wm.r_x_r
    return wm, is_inv
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
    return wm, true
end

# helper method to be used when computing gateaux constants
function transformed_information_matrices(
    nim::AbstractVector{<:AbstractMatrix},
    is_inv::Bool,
    pk::PriorSample,
    tc::TrafoConstants,
)
    # dummies --------v--v
    wm = WorkMatrices(1, 1, parameter_dimension(pk), codomain_dimension(tc))
    res_is_inv = missing
    tnim = map(1:length(pk.p)) do i
        wm.r_x_r .= nim[i] # will be overwritten by the next call
        _, res_is_inv = apply_transformation!(wm, is_inv, tc, i)
        return deepcopy(wm.t_x_t)
    end
    return tnim, res_is_inv
end

## linear algebra shortcuts ##

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
