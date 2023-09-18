# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## function stubs for user methods ##

"""
    Kirstine.unit_length(m::M) -> Integer

Return the length of one unit of observation.

# Implementation

For a user type `M <: NonlinearRegression` this should return the length ``\\DimUnit``
of one unit of observation ``\\Unit``.

See also the [mathematical background](math.md#Design-Problems).
"""
function unit_length end

"""
    Kirstine.allocate_covariate(m::M) -> c::C

Construct a single `Covariate` for the model.

# Implementation

For user types `M <: NonlinearRegression` and a corresponding `C <: Covariate`,
this should construct a single instance `c` of `C`
with some (model-specific) sensible initial value.

See also the [mathematical background](math.md#Design-Problems).
"""
function allocate_covariate end

"""
    Kirstine.jacobianmatrix!(jm::AbstractMatrix{<:Real}, m::M, c::C, p::P) -> jm

Compute Jacobian matrix of the mean function at `p`.

# Implementation

For user types `M <: NonlinearRegression`, `C <: Covariate`, and `P <: Parameter`,
this should fill in the elements of the pre-allocated Jacobian matrix `jm`,
and finally return `jm`.

Note that you are free how you map the partial derivatives to the columns of `jm`.
The only thing that is required is to use the same order in any [`DeltaMethod`](@ref)
that you want to implement.

See also the [mathematical background](math.md#Objective-Function).
"""
function jacobianmatrix! end

"""
    Kirstine.update_model_covariate!(c::C, dp::AbstractVector{<:Real}, m::M, cp::Cp) -> c

Map a design point to a model covariate.

# Implementation

For user types `C <: Covariate`, `M <: NonlinearRegression`, and
`Cp <: CovariateParameterization` this should set the fields of the single covariate `c`
according to the single design point `dp`. Finally, this method should return `c`.

See also the [mathematical background](math.md#Design-Problems).
"""
function update_model_covariate! end

"""
    Kirstine.update_model_vcov!(Sigma::Matrix, c::C, m::M)

Compute variance-covariance matrix of the nonlinear regression model
for one unit of observation.

# Implementation

For user types `C <: Covariate` and `M <: NonlinearRegression`,
this should fill in the elements of `Sigma`
for a unit of observation with covariate `c`.
The `size` of the matrix is `(unit_length(m), unit_length(m))`.

See also the [mathematical background](math.md#Design-Problems).
"""
function update_model_vcov! end

"""
    Kirstine.dimension(p::P) -> Integer

Return dimension of the parameter space.

# Implementation

For a user type `P <: Parameter`, this should return the dimension of the parameter space.

See also the [mathematical background](math.md#Design-Problems).
"""
function dimension end

## helper macros ##

"""
    @define_scalar_unit_model module_name model_name covariate_field_names...

Generate code for defining a [`NonlinearRegression`](@ref) model, a corresponding
[`Covariate`](@ref), and helper functions for a 1-dimensional unit of observation
with constant measurement variance.

The model constructor will have the measurement standard deviation `sigma` as a mandatory
keyword argument.

# Examples

The call

```julia
julia> @define_scalar_unit_model Kirstine FooMod bar baz

```

is equivalent to manually spelling out

```julia
struct FooMod <: Kirstine.NonlinearRegression
    sigma::Float64
    function FooMod(; sigma::Real)
        return new(sigma)
    end
end
mutable struct FooModCovariate <: Kirstine.Covariate
    bar::Float64
    baz::Float64
end
Kirstine.unit_length(m::FooMod) = 1
function Kirstine.upate_model_vcov!(Sigma::Matrix{Float64}, c::FooModCovariate, m::FooMod)
    Sigma[1, 1] = m.sigma^2
    return Sigma
end
Kirstine.allocate_covariate(m::FooMod) = FooModCovariate(0, 0)
```

!!! note

    It is necessary to tell the macro under which name `Kirstine` is known in the caller's
    namespace. If we had loaded the package with

    ```julia
    import Kirstine as Smith
    ```

    the macro call in the example would have had to be

    ```julia
    Smith.@define_scalar_unit_model Smith FooMod bar baz
    ```
"""
macro define_scalar_unit_model(module_name, model_name, covariate_field_names...)
    if isempty(covariate_field_names)
        error("no covariate field names supplied")
    end
    # programmatically build up the covariate struct definition
    covar_name = Symbol(String(model_name) * "Covariate")
    covar_fields = [Expr(:(::), fn, :Float64) for fn in covariate_field_names]
    covar_struct_expression = Expr(
        :struct,
        true, # mutable
        Expr(:<:, covar_name, :($module_name.Covariate)),
        Expr(:block, covar_fields...),
    )
    # Note: We have to `esc()` the definitions so that they are evaluated in the calling
    # Module. This has the nice side effect of not replacing `m` with a gensym.
    return esc(
        quote
            struct $model_name <: $module_name.NonlinearRegression
                sigma::Float64
                function $model_name(; sigma::Real)
                    return new(sigma)
                end
            end
            $covar_struct_expression
            function $module_name.unit_length(m::$model_name)
                return 1
            end
            function $module_name.update_model_vcov!(
                s::Matrix{Float64},
                c::$covar_name,
                m::$model_name,
            )
                s[1, 1] = m.sigma^2
                return s
            end
            function $module_name.allocate_covariate(m::$model_name)
                return $covar_name($(fill(0, length(covariate_field_names))...))
            end
        end,
    )
end

"""
    @define_vector_parameter module_name parameter_name field_names...

Generate code for defining a subtype of [`Parameter`](@ref) with the given name and fields,
and define its dimension as `length(field_names)`.

A type defined this way has a keyword constructor.

For the `module_name` argument see the note at [`@define_scalar_unit_model`](@ref).

# Examples

The call

```julia
julia> @define_vector_parameter Kirstine BarPar a b

```

is equivalent to

```julia
@kwdef struct BarPar <: Kirstine.Parameter
    a::Float64
    b::Float64
end

Kirstine.dimension(p::BarPar) = 2
```

A `BarPar` object with `a=1` and `b=2` can then be constructed by

```julia
BarPar(; a = 1, b = 2)
```
"""
macro define_vector_parameter(module_name, parameter_name, field_names...)
    if isempty(field_names)
        error("no field names supplied")
    end
    dim = length(field_names)
    par_fields = [Expr(:(::), fn, :Float64) for fn in field_names]
    par_struct_expression = Expr(
        :struct,
        false, # immutable
        Expr(:<:, parameter_name, :($module_name.Parameter)),
        Expr(:block, par_fields...),
    )
    return esc(quote
        @kwdef $par_struct_expression
        $module_name.dimension(p::$parameter_name) = $dim
    end)
end
