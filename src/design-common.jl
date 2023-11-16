# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## preallocating objects for less memory overhead ##

struct NIMWorkspace
    r_x_r::Matrix{Float64}
    r_x_t::Matrix{Float64}
    t_x_r::Matrix{Float64}
    t_x_t::Matrix{Float64}
    function NIMWorkspace(r::Integer, t::Integer)
        if r < 1 || t < 1
            throw(
                ArgumentError("matrix dimensions must be positive, got (r, t) = ($r, $t)"),
            )
        end
        new(zeros(r, r), zeros(r, t), zeros(t, r), zeros(t, t))
    end
end

struct NRWorkspace <: ModelWorkspace
    m_x_r::Matrix{Float64}
    m_x_m::Vector{Matrix{Float64}}
    function NRWorkspace(K::Integer, m::Integer, r::Integer)
        if r < 1
            throw(ArgumentError("matrix dimension must be positive, got r = $r"))
        end
        if K < 1
            throw(ArgumentError("need at least one design point, got K = $K"))
        end
        mxm = [zeros(m, m) for _ in 1:K]
        new(zeros(m, r), mxm)
    end
end

function allocate_model_workspace(K::Integer, m::NonlinearRegression, pk::PriorSample)
    return NRWorkspace(K, unit_length(m), dimension(pk.p[1]))
end

function allocate_initialize_covariates(d, m, cp)
    K = numpoints(d)
    cs = [allocate_covariate(m) for _ in 1:K]
    update_covariates!(cs, d, m, cp)
    return cs
end

## model and criterion agnostic objective and gateaux derivative ##

function update_covariates!(
    c::AbstractVector{<:Covariate},
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
)
    for k in 1:length(c)
        map_to_covariate!(c[k], points(d)[k], m, cp)
    end
    return c
end

# Initialize matrices that do not depend on the parameter,
# but only on covariate values.
#
# For a nonlinear regression model,
# this sets up `mw.m_x_m` as the (upper triangle of) the Cholesky factor
# of the unit covariance matrix Σ.
# This is the setup that `average_fishermatrix!` expects.
function update_model_workspace!(
    mw::NRWorkspace,
    m::NonlinearRegression,
    c::AbstractVector{<:Covariate},
)
    for k in 1:length(c)
        update_model_vcov!(mw.m_x_m[k], m, c[k])
        potrf!('U', mw.m_x_m[k])
    end
    return mw
end

function objective!(
    nw::NIMWorkspace,
    mw::ModelWorkspace,
    c::AbstractVector{<:Covariate},
    dc::DesignCriterion,
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    tc::TrafoConstants,
    na::NormalApproximation,
)
    update_covariates!(c, d, m, cp)
    update_model_workspace!(mw, m, c)
    # When the information matrix is singular, the objective function is undefined. Lower
    # level calls may throw a PosDefException or a SingularException. This also means that
    # `d` can not be a solution to the maximization problem, hence we return negative
    # infinity in these cases.
    try
        acc = 0
        for i in 1:length(pk.p)
            informationmatrix!(nw.r_x_r, mw, weights(d), m, c, pk.p[i], na)
            _, is_inv = apply_transformation!(nw, false, tc, i)
            acc += pk.weight[i] * criterion_integrand!(nw.t_x_t, is_inv, dc)
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
    nw::NIMWorkspace,
    mw::ModelWorkspace,
    c::AbstractVector{<:Covariate}, # only one element, but passed to `informationmatrix!`
    gconst::GateauxConstants,
    direction::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    na::NormalApproximation,
)
    update_covariates!(c, direction, m, cp)
    update_model_workspace!(mw, m, c)
    acc = 0
    for i in 1:length(pk.p)
        informationmatrix!(nw.r_x_r, mw, weights(direction), m, c, pk.p[i], na)
        acc += pk.weight[i] * gateaux_integrand(gconst, nw.r_x_r, i)
    end
    return acc
end

## normalized information matrix for θ ##

# In the simplest case, the normalized information is just the Fisher information matrix
# averaged wrt the design measure.
#
# Calling conventions:
#  * `nim` will be overwritten with the upper triangle of the information matrix
#  * `mw` must be initialized with what `average_fishermatrix!` for `typeof(m)` expects to find.
#
# Returns: a reference to `nim`
function informationmatrix!(
    nim::AbstractMatrix{<:Real},
    mw::ModelWorkspace,
    w::AbstractVector{<:Real},
    m::Model,
    c::AbstractVector{<:Covariate},
    p::Parameter,
    na::FisherMatrix,
)
    return average_fishermatrix!(nim, mw, w, m, c, p)
end

# Normalized information matrix for nonlinear regression with vector units.
#
# See also docs/src/math.md and `update_model_workspace`.
#
# Calling conventions:
#  * `afm` will be overwritten with the upper triangle of the averaged Fisher matrix
#  * `mw.m_x_m` must contain the Cholesky factors of Σ(c_k), k = 1, …, K
#
# Returns: a reference to `afm`
function average_fishermatrix!(
    afm::AbstractMatrix{<:Real},
    mw::ModelWorkspace,
    w::AbstractVector{<:Real},
    m::NonlinearRegression,
    c::AbstractVector{<:Covariate},
    p::Parameter,
)
    fill!(afm, 0.0)
    for k in 1:length(w)
        # Fill in the jacobian matrix of the mean function
        jacobianmatrix!(mw.m_x_r, m, c[k], p)
        # multiply with inverse Cholesky factor of Σ from the left,
        # i.e. set m_x_r = solution to `chol_vcov[k]' * X = 1 * m_x_r`
        # i.e. triangular matrix is the
        #  * 'L'eft factor,
        #  * with data in 'U'pper triangle,
        #  * and 'T'ransposed,
        #  * but 'N'o unit diagonal.
        trsm!('L', 'U', 'T', 'N', 1.0, mw.m_x_m[k], mw.m_x_r)
        # update afm = w[k] * m_x_r' * m_x_r + 1 * afm
        # i.e. data in 'U'pper triangle and 'T'ransposed first
        syrk!('U', 'T', w[k], mw.m_x_r, 1.0, afm)
    end
    # Note: afm is implicitly symmetric, only the upper triangle contains data.
    return afm
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
    nw = NIMWorkspace(dimension(p), 1)
    # wrapping `p` is a bit ugly here
    mw = allocate_model_workspace(numpoints(d), m, PriorSample([p]))
    update_model_workspace!(mw, m, c)
    informationmatrix!(nw.r_x_r, mw, weights(d), m, c, p, na)
    return Symmetric(nw.r_x_r)
end

function inverse_information_matrices(
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    pk::PriorSample,
    na::NormalApproximation,
)
    # Calculate inverse of normalized information matrix for each parameter value.
    # Only the upper triangle of the (symmetric) information matrix is relevant.
    c = allocate_initialize_covariates(d, m, cp)
    # The transformed parameter dimension t = 1 is a dummy argument here.
    nw = NIMWorkspace(dimension(pk.p[1]), 1)
    mw = allocate_model_workspace(numpoints(d), m, pk)
    update_model_workspace!(mw, m, c)
    # The documentation of `potri!` is not very clear that it expects a Cholesky factor as
    # input, and does _not_ call `potrf!` itself.
    # See also https://netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga9dfc04beae56a3b1c1f75eebc838c14c.html
    inv_nims = map(pk.p) do p
        informationmatrix!(nw.r_x_r, mw, weights(d), m, c, p, na)
        potrf!('U', nw.r_x_r)
        potri!('U', nw.r_x_r)
        return deepcopy(nw.r_x_r)
    end
    return inv_nims
end

## normalized information matrix for T(θ) ##

# Calling conventions for `apply_transformation!`
#
# * `nw.r_x_r` holds the information matrix or its inverse, depending on the `is_inv` flag.
# * Only the upper triangle of `nw.r_x_r` is used.
# * `nw.t_x_t` will be overwritten with the transformed information matrix or its inverse.
# * The return value are `nw` and a flag that indicates whether `nw.t_x_t` it is inverted.
# * Only the upper triangle of `nw.t_x_t` is guaranteed to make sense, but specific methods
#   are free to return a dense matrix.
# * Whether the returned matrix will be inverted is _not_ controlled by `is_inv`.

function apply_transformation!(nw::NIMWorkspace, is_inv::Bool, tc::TCIdentity, index)
    # For the Identity transformation we just pass through the information matrix.
    nw.t_x_t .= nw.r_x_r
    return nw, is_inv
end

function apply_transformation!(nw::NIMWorkspace, is_inv::Bool, tc::TCDeltaMethod, index)
    # Denote the Jacobian matrix of T by J and the normalized information matrix by M. We
    # want to efficiently calculate J * inv(M) * J'.
    #
    # A precalculated J is given in tc.jm[index].
    #
    # Depending on whether nw.r_x_r contains (the upper triangle of) M or inv(M) we use
    # different BLAS routines and a different order of multiplication.
    if is_inv
        # Denote the given inverse of M by invM.
        # We first calculate A := (J * invM) and store it in nw.t_x_r.
        #
        # The `symm!` call performes the following in-place update:
        #
        #   nw.t_x_r = 1 * tc.jm[index] * Symmetric(nw.r_x_r) + 0 * nw.t_x_r
        #
        # That is,
        #  * the symmetric matrix nw.r_x_r is the factor on the 'R'ight, and
        #  * the data is contained in the 'U'pper triangle.
        symm!('R', 'U', 1.0, nw.r_x_r, tc.jm[index], 0.0, nw.t_x_r)
        # Next we calculate the result A * J' and store it in nw.t_x_t.
        mul!(nw.t_x_t, nw.t_x_r, tc.jm[index]')
    else
        # When the input is not yet inverted, we don't want to calculate inv(M) explicitly.
        # As a first step we calculate B := inv(M) * J and store it in nw.r_x_t.
        # We do this by solving the linear system M * B == J in place.
        # As we do not want to overwrite J, we copy J' into a work matrix.
        nw.r_x_t .= tc.jm[index]'
        # The `posv!` call performs the following in-place update:
        #
        #  * Overwrite nw.r_x_r with its Cholesky factor `potrf!(nw.r_x_r)`, using the data
        #    in the 'U'pper triangle.
        #  * Overwrite nw.r_x_t by the solution of the linear system.
        posv!('U', nw.r_x_r, nw.r_x_t)
        # Next, we calculate the result J * B and store it in nw.t_x_t.
        mul!(nw.t_x_t, tc.jm[index], nw.r_x_t)
    end
    # Note that for this method, the result is not just an upper triangle, but always a
    # dense matrix.
    return nw, true
end

# helper method to be used when computing gateaux constants
function transformed_information_matrices(
    nim::AbstractVector{<:AbstractMatrix},
    is_inv::Bool,
    pk::PriorSample,
    tc::TrafoConstants,
)
    nw = NIMWorkspace(parameter_dimension(pk), codomain_dimension(tc))
    res_is_inv = missing
    tnim = map(1:length(pk.p)) do i
        nw.r_x_r .= nim[i] # will be overwritten by the next call
        _, res_is_inv = apply_transformation!(nw, is_inv, tc, i)
        return deepcopy(nw.t_x_t)
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
