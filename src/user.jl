# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## function stubs for user methods ##

"""
    Kirstine.unit_length(m::M) -> Integer

Return the length of one unit of observation.

# Implementation

For a user type `M <: Model` this should return the length ``\\DimUnit``
of one unit of observation ``\\Unit``.

See also the [mathematical background](math.md#Design-Problems).
"""
function unit_length end

"""
    Kirstine.allocate_covariate(m::M) -> c::C

Construct a single `Covariate` for the model.

# Implementation

For user types `M <: Model` and a corresponding `C <: Covariate`,
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
    Kirstine.map_to_covariate!(c::C, dp::AbstractVector{<:Real}, m::M, cp::Cp) -> c

Map a design point to a model covariate.

# Implementation

For user types `C <: Covariate`, `M <: Model`, and
`Cp <: CovariateParameterization` this should set the fields of the single covariate `c`
according to the single design point `dp`. Finally, this method should return `c`.

See also the [mathematical background](math.md#Design-Problems).
"""
function map_to_covariate! end

"""
    Kirstine.update_model_vcov!(Sigma::Matrix, m::M, c::C)

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

## generic identity covariate parameterization ##

"""
    JustCopy <: CovariateParameterization

A convenience type to use when design variables and model covariates are identical.
"""
struct JustCopy <: CovariateParameterization
    field_names::Vector{Symbol}
end

"""
    JustCopy(names::Symbol...)

Construct a [`CovariateParameterization`](@ref) that copies the elements of a design point
to the fields of a [`Covariate`](@ref) with the given `names`.

This parameterization can be used with any [`Covariate`](@ref)
that has only `Float64` fields.
Note that the order of the `names`
should match the the order of `dimnames(region(prob))` for the [`DesignProblem`](@ref)
in which `JustCopy` is used.
No new method for [`map_to_covariate!`](@ref) needs to be implemented for user types
since the package supplies a generic one that works with any [`Model`](@ref)
and any [`Covariate`](@ref).

See also the [tutorial](tutorial.md) for another usage example.

# Examples

When we have

```jldoctest justcopy
julia> @simple_model Foo a b

julia> @simple_parameter Foo θ

```

we can define

```jldoctest justcopy
julia> cp = JustCopy(:a, :b)
JustCopy([:a, :b])
```

and then solve

```jldoctest justcopy
julia> dp = DesignProblem(;
           region = DesignInterval(:a => (0, 1), :b => (-2, 5)),
           covariate_parameterization = cp,
           criterion = DCriterion(),
           model = FooModel(; sigma = 1),
           prior_knowledge = PriorSample([FooParameter(; θ = 42)]),
       );

```

whithout first having to add a method to `map_to_covariate!`
for the `FooModel` and `FooCovariate` types.

Note however that the equivalent manual version

```jldoctest justcopy
julia> struct CopyBoth <: CovariateParameterization end

julia> function Kirstine.map_to_covariate!(c::FooCovariate, dp, m::FooModel, cp::CopyBoth)
           c.a = dp[1]
           c.b = dp[2]
           return c
       end

```

would be slightly more efficient
since it knows the types and field names at compile time.

Also note that since the design points are assigned in the order of the names given,
`cp = JustCopy(:b, :a)` would be equivalent to setting `c.b = dp[1]` and `c.a = dp[2]`.
"""
function JustCopy(names::Symbol...)
    return JustCopy(collect(names))
end

function map_to_covariate!(c::Covariate, dp::AbstractVector{<:Real}, m::Model, cp::JustCopy)
    dimc = length(cp.field_names)
    cnames = fieldnames(typeof(c))
    err = ArgumentError(
        "field names of covariate must match names of JustCopy parameterization",
    )
    if length(cnames) != dimc
        throw(err)
    end
    for fn in cp.field_names
        if !(fn in cnames)
            throw(err)
        end
    end
    if dimc != length(dp)
        throw(DimensionMismatch("dimension of design point must match that of covariate"))
    end
    for i in 1:dimc
        fn = cp.field_names[i]
        # The next call allocates memory
        # since the field names and their types are not know at compile time.
        # As a consequence, `map_to_covariate` is slowed down by a factor of 10
        # compared to a manually written version with hardcoded field names.
        #
        # This is, however, acceptable since we only need to do this once per design measue,
        # and the performance penalty that we incur is dwarfed by the operations
        # that need to be done for each parameter in a prior sample.
        # Moreover, the amount of memory allocated is only 16 bytes.
        setfield!(c, fn, dp[i])
    end
    return c
end

## helper macros ##

function _kirstine_known_as(calling_mod)
    # This takes care of
    #  `import Kirstine`
    # and of
    #   `import Kirstine as SomethingElse`.
    # Ingoring `:ans` is necessary on the REPL.
    filtered_modnames = filter(names(calling_mod; imported = true)) do n
        x = getfield(calling_mod, n)
        x isa Module && fullname(x)[end] == :Kirstine && n != :ans
    end
    if !(isempty(filtered_modnames))
        if length(filtered_modnames) > 1
            @warn "Kirstine seems to be known under mupltiple names" filtered_modnames
        end
        return filtered_modnames[1]
    end
    # When Kirstine was not `import`ed, we must by logical necessity be `using` it, since
    # otherwise this very function could not have been called. We could play around with a
    # `ccall` to `:jl_module_using`, but this would give us only the same information.
    return :Kirstine
end

"""
    @simple_model name covariate_field_names...

Generate code for defining a [`NonlinearRegression`](@ref) model, a corresponding
[`Covariate`](@ref), and helper functions for a 1-dimensional unit of observation
with constant measurement variance.

The model constructor will have the measurement standard deviation `sigma` as a mandatory
keyword argument.

# Examples

The call

```julia
@simple_model Emax dose
```

is equivalent to manually spelling out

```julia
struct EmaxModel <: Kirstine.NonlinearRegression
    sigma::Float64
    function EmaxModel(; sigma::Real)
        return new(sigma)
    end
end
mutable struct EmaxCovariate <: Kirstine.Covariate
    dose::Float64
end
Kirstine.unit_length(m::EmaxModel) = 1
function Kirstine.update_model_vcov!(Sigma::Matrix{Float64}, m::EmaxModel, c::EmaxCovariate)
    Sigma[1, 1] = m.sigma^2
    return Sigma
end
Kirstine.allocate_covariate(m::EmaxModel) = EmaxCovariate(0)
```

Note that this works both when you are `using Kirstine`
and when you are `import`ing it,
even if you do something like

```julia
import Kirstine as Smith
```

Further note that the `@simple_model` macro can be used together with,
or independently of [`@simple_parameter`](@ref).
"""
macro simple_model(name, covariate_field_names...)
    module_name = _kirstine_known_as(__module__)
    model_name = Symbol(string(name) * "Model")
    covariate_name = Symbol(string(name) * "Covariate")
    if isempty(covariate_field_names)
        throw(ArgumentError("no covariate field names supplied"))
    end
    # programmatically build up the covariate struct definition
    covariate_fields = [Expr(:(::), fn, :Float64) for fn in covariate_field_names]
    covariate_struct_expression = Expr(
        :struct,
        true, # mutable
        Expr(:<:, covariate_name, :($module_name.Covariate)),
        Expr(:block, covariate_fields...),
    )
    # Note: We have to `esc()` the definitions so that they are evaluated in the calling
    # module. This has the nice side effect of not replacing `m` with a gensym.
    return esc(
        quote
            struct $model_name <: $module_name.NonlinearRegression
                sigma::Float64
                function $model_name(; sigma::Real)
                    return new(sigma)
                end
            end
            $covariate_struct_expression
            function $module_name.unit_length(m::$model_name)
                return 1
            end
            function $module_name.update_model_vcov!(
                s::Matrix{Float64},
                m::$model_name,
                c::$covariate_name,
            )
                s[1, 1] = m.sigma^2
                return s
            end
            function $module_name.allocate_covariate(m::$model_name)
                return $covariate_name($(fill(0, length(covariate_field_names))...))
            end
        end,
    )
end

"""
    @simple_parameter name field_names...

Generate code for defining a subtype of [`Parameter`](@ref) with the given name and fields,
and define its dimension as `length(field_names)`.

A type defined this way has a keyword constructor.
All fields will have type `Float64`.

# Examples

The call

```julia
@simple_parameter Emax e0 emax ec50
```

is equivalent to

```julia
@kwdef struct EmaxParameter <: Kirstine.Parameter
    e0::Float64
    emax::Float64
    ec50::Float64
end

Kirstine.dimension(p::EmaxParameter) = 2
```

An `EmaxParameter` object with `e0=1`, `emax=2`, and `ec50=4` can then be constructed by

```julia
EmaxParameter(; e0 = 1, emax = 2, ec50 = 4)
```

Note that this works both when you are `using Kirstine`
and when you are `import`ing it,
even if you do something like

```julia
import Kirstine as Smith
```

Further note that the `@simple_parameter` macro can be used together with,
or independently of [`@simple_model`](@ref).
"""
macro simple_parameter(name, field_names...)
    module_name = _kirstine_known_as(__module__)
    parameter_name = Symbol(string(name) * "Parameter")
    if isempty(field_names)
        throw(ArgumentError("no field names supplied"))
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
