# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## code specific for nonlinear regression models ##

"""
    NonlinearRegression

Supertype for user-defined nonlinear regression models.
"""
abstract type NonlinearRegression <: Model end

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
