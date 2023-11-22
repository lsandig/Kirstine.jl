# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## concrete types for posterior transformations ##

"""
    Identity <: Transformation

The [`Transformation`](@ref) that maps a parameter to itself.
"""
struct Identity <: Transformation end

@doc raw"""
    DeltaMethod <: Transformation
    DeltaMethod(jacobian_matrix::Function)

A nonlinear [`Transformation`](@ref) of the model parameter.

The [delta method](https://en.wikipedia.org/wiki/Delta_method)
uses the Jacobian matrix ``\TotalDiff\Transformation``
to map the asymptotic multivariate normal distribution of ``\Parameter``
to the asymptotic multivariate normal distribution of ``\Transformation(\Parameter)``.

To construct a `DeltaMethod` object,
`jacobian_matrix` must be a function
that maps a [`Parameter`](@ref) `p`
to the Jacobian matrix of ``\Transformation`` evaluated at `p`.

Note that the order of the columns must match the order that you used
when implementing the [`jacobianmatrix!`](@ref) for your model.

# Examples
Suppose `p` has the fields `a` and `b`, and ``\Transformation(\Parameter) = (ab, b/a)'``.
Then the Jacobian matrix of ``\Transformation`` is
```math
\TotalDiff\Transformation(\Parameter) =
  \begin{bmatrix}
    b      & a   \\
    -b/a^2 & 1/a \\
  \end{bmatrix}.
```
In Julia this is equivalent to
```jldoctest; output = false
jm1(p) = [p.b p.a; -p.b/p.a^2 1/p.a]
DeltaMethod(jm1)
# output
DeltaMethod{typeof(jm1)}(jm1)
```
Note that for a scalar quantity,
e.g. ``\Transformation(\Parameter) = \sqrt{ab}``,
the Jacobian matrix is a _row_ vector.
```jldoctest; output = false
jm2(p) = [b a] ./ (2 * sqrt(p.a * p.b))
DeltaMethod(jm2)
# output
DeltaMethod{typeof(jm2)}(jm2)
```
"""
struct DeltaMethod{T<:Function} <: Transformation
    jacobian_matrix::T # parameter -> Matrix{Float64}
end

struct TCIdentity <: TrafoConstants
    codomain_dimension::Int64
    idmat::Matrix{Float64}
end
struct TCDeltaMethod <: TrafoConstants
    codomain_dimension::Int64
    jm::Vector{Matrix{Float64}}
end

function trafo_constants(trafo::Identity, pk::PriorSample)
    r = parameter_dimension(pk)
    return TCIdentity(r, diagm(ones(r)))
end

function trafo_constants(trafo::DeltaMethod, pk::PriorSample)
    jm = [trafo.jacobian_matrix(p) for p in pk.p]
    r = parameter_dimension(pk)
    if any(j -> size(j) != size(jm[1]), jm)
        throw(DimensionMismatch("trafo jacobians must be identical in size"))
    end
    # We know all elements of jm have identical sizes, so checking the first is enough
    ncol = size(jm[1], 2)
    if ncol != r
        throw(DimensionMismatch("trafo jacobian must have $(r) columns, got $(ncol)"))
    end
    return TCDeltaMethod(size(jm[1], 1), jm)
end

function codomain_dimension(tc::TCIdentity)
    return tc.codomain_dimension
end

function codomain_dimension(tc::TCDeltaMethod)
    return tc.codomain_dimension
end

function trafo_jacobian_matrix_for_index(tc::TCIdentity, i::Integer)
    return tc.idmat
end

function trafo_jacobian_matrix_for_index(tc::TCDeltaMethod, i::Integer)
    return tc.jm[i]
end
