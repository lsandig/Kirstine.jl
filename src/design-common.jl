# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## preallocating objects for less memory overhead ##

mutable struct NIMWorkspace
    r_x_r::Matrix{Float64}
    r_x_t::Matrix{Float64}
    t_x_r::Matrix{Float64}
    t_x_t::Matrix{Float64}
    r_is_inv::Bool # r_x_r should be considered as inverted NIM
    t_is_inv::Bool # t_x_t should be considered as inverted tranformed NIM
    function NIMWorkspace(r::Integer, t::Integer)
        if r < 1 || t < 1
            throw(
                ArgumentError("matrix dimensions must be positive, got (r, t) = ($r, $t)"),
            )
        end
        new(zeros(r, r), zeros(r, t), zeros(t, r), zeros(t, t), false, false)
    end
end

struct Workspaces
    nw::NIMWorkspace
    mw::ModelWorkspace
    c::Vector{<:Covariate}
end

struct GCPriorSample <: GateauxConstants
    A::Vector{Matrix{Float64}}
    tr_B::Vector{Float64}
end

"""
    implied_covariates(d::DesignMeasure, m::M, cp::Cp)

Return covariates that are implied by `d`
for instances of user types `M <: Model` and `Cp <: CovariateParameterization`.

This function is useful for debugging.

See also [`map_to_covariate!`](@ref).
"""
function implied_covariates(d, m, cp)
    K = numpoints(d)
    cs = [allocate_covariate(m) for _ in 1:K]
    update_covariates!(cs, d, m, cp)
    return cs
end

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

# Note: only the upper triangle of the symmetric matrix A needs to be filled out,
# since `gateaux_integrand` uses `tr_prod` for the multiplication.
# But producing a dense matrix does not hurt either.
function gateaux_integrand(A::AbstractMatrix, nim_direction::AbstractMatrix, tr_B::Real)
    return tr_prod(A, nim_direction, :U) - tr_B
end

## normalized information matrix for θ ##

# In the simplest case, the normalized information is just the Fisher information matrix
# averaged wrt the design measure.
function informationmatrix!(afm::AbstractMatrix{<:Real}, na::FisherMatrix)
    return afm
end

# For debugging purposes, one will typically want to look at an information matrix
# corresponding to a single parameter value, not to all thousands of them in a prior sample.
"""
    informationmatrix(
        d::DesignMeasure,
        m::Model,
        cp::CovariateParameterization,
        p::Parameter,
        na::NormalApproximation,
    )

Compute the normalized information matrix for a single `Parameter` value `p`.

This function is useful for debugging.

See also [`apply_transformation`](@ref)
and the [mathematical background](math.md#Objective-Function).
"""
function informationmatrix(
    d::DesignMeasure,
    m::Model,
    cp::CovariateParameterization,
    p::Parameter,
    na::NormalApproximation,
)
    c = implied_covariates(d, m, cp)
    # Note: t = 1 is a dummy value, no trafo will be applied
    nw = NIMWorkspace(dimension(p), 1)
    # wrapping `p` is a bit ugly here
    mw = allocate_model_workspace(numpoints(d), m, PriorSample([p]))
    update_model_workspace!(mw, m, c)
    average_fishermatrix!(nw.r_x_r, mw, weights(d), m, c, p)
    informationmatrix!(nw.r_x_r, na)
    return Symmetric(nw.r_x_r)
end

"""
    apply_transformation(nim::AbstractMatrix{<:Real}, p::Parameter, trafo::Transformation)

Compute the information matrix for the transformed `Parameter`.

This function is useful for debugging.

See also [`informationmatrix`](@ref)
and the [mathematical background](math.md#Objective-Function).
"""
function apply_transformation(
    M::AbstractMatrix{<:Real},
    p::Parameter,
    trafo::Transformation,
)
    pk_wrap = PriorSample([p])
    tc = trafo_constants(trafo, pk_wrap)
    r = size(M, 1)
    t = codomain_dimension(trafo, pk_wrap)
    nw = NIMWorkspace(r, t)
    nw.r_x_r .= M
    nw.r_is_inv = false
    apply_transformation!(nw, trafo, trafo_jacobianmatrix_for_index(tc, 1))
    if nw.t_is_inv
        return inv(Symmetric(nw.t_x_t))
    else
        return Symmetric(deepcopy(nw.t_x_t))
    end
end

## normalized information matrix for T(θ) ##

# Calling conventions for `apply_transformation!`
#
# * `nw.r_x_r` holds the information matrix or its inverse, depending on the `nw.r_is_inv` flag.
# * Only the upper triangle of `nw.r_x_r` is used.
# * `nw.t_x_t` will be overwritten with the transformed information matrix or its inverse.
# * `nw.t_is_inv` will be set to true if `nw.t_x_t` is inverted.
# * The return value is `nw`.
# * Only the upper triangle of `nw.t_x_t` is guaranteed to make sense, but specific methods
#   are free to return a dense matrix.
# * Whether the returned matrix will be inverted is _not_ controlled by `nw.t_is_inv`.

function apply_transformation!(nw::NIMWorkspace, trafo::Identity, _)
    # For the Identity transformation we just pass through the information matrix.
    nw.t_x_t .= nw.r_x_r
    nw.t_is_inv = nw.r_is_inv
    return nw
end

function apply_transformation!(nw::NIMWorkspace, trafo::DeltaMethod, jm::Matrix{Float64})
    # Denote the Jacobian matrix of T by J and the normalized information matrix by M. We
    # want to efficiently calculate J * inv(M) * J'.
    #
    # A precalculated J is given in jm.
    #
    # Depending on whether nw.r_x_r contains (the upper triangle of) M or inv(M) we use
    # different BLAS routines and a different order of multiplication.
    if nw.r_is_inv
        # Denote the given inverse of M by invM.
        # We first calculate A := (J * invM) and store it in nw.t_x_r.
        #
        # The `symm!` call performes the following in-place update:
        #
        #   nw.t_x_r = 1 * jm * Symmetric(nw.r_x_r) + 0 * nw.t_x_r
        #
        # That is,
        #  * the symmetric matrix nw.r_x_r is the factor on the 'R'ight, and
        #  * the data is contained in the 'U'pper triangle.
        symm!('R', 'U', 1.0, nw.r_x_r, jm, 0.0, nw.t_x_r)
        # Next we calculate the result A * J' and store it in nw.t_x_t.
        mul!(nw.t_x_t, nw.t_x_r, jm')
    else
        # When the input is not yet inverted, we don't want to calculate inv(M) explicitly.
        # As a first step we calculate B := inv(M) * J and store it in nw.r_x_t.
        # We do this by solving the linear system M * B == J in place.
        # As we do not want to overwrite J, we copy J' into a work matrix.
        nw.r_x_t .= jm'
        # The `posv!` call performs the following in-place update:
        #
        #  * Overwrite nw.r_x_r with its Cholesky factor `potrf!(nw.r_x_r)`, using the data
        #    in the 'U'pper triangle.
        #  * Overwrite nw.r_x_t by the solution of the linear system.
        posv!('U', nw.r_x_r, nw.r_x_t)
        # Next, we calculate the result J * B and store it in nw.t_x_t.
        mul!(nw.t_x_t, jm, nw.r_x_t)
    end
    nw.t_is_inv = true
    # Note that for this method, the result is not just an upper triangle, but always a
    # dense matrix.
    return nw
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
