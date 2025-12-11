# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

## function stubs for user methods ##

@doc raw"""
    Kirstine.unit_length(m::M) -> Integer

Return the length of one unit of observation.

# Implementation

For a user type `M <: Kirstine.Model` this should return the length ``\DimUnit``
of one unit of observation ``\Unit``.

See also the [mathematical background](math.md#Design-Problems).
"""
function unit_length end

"""
    Kirstine.allocate_covariate(m::M) -> c::C

Construct a single covariate for the model.

# Implementation

For user types `M <: Kirstine.Model` and a corresponding `C <: Kirstine.Covariate`,
this should construct a single instance `c` of `C`
with some (model-specific) sensible initial value.

See also the [mathematical background](math.md#Design-Problems).
"""
function allocate_covariate end

"""
    Kirstine.jacobianmatrix!(jm::AbstractMatrix{<:Real}, m::M, c::C, p::P) -> jm

Compute Jacobian matrix of the mean function at `p`.

# Implementation

For user types `M <: Kirstine.NonlinearRegression`, `C <: Kirstine.Covariate`, and `P <: Kirstine.Parameter`,
this should fill in the elements of the preallocated Jacobian matrix `jm`,
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

For user types `C <: Kirstine.Covariate`, `M <: Kirstine.Model`, and
`Cp <: Kirstine.CovariateParameterization` this should set the fields of the single covariate `c`
according to the single design point `dp`. Finally, this method should return `c`.

See also [`implied_covariates`](@ref)
and the [mathematical background](math.md#Design-Problems).
"""
function map_to_covariate! end

"""
    Kirstine.update_model_vcov!(Sigma::Matrix, m::M, c::C)

Compute variance-covariance matrix of the nonlinear regression model
for one unit of observation.

# Implementation

For user types `C <: Kirstine.Covariate` and `M <: Kirstine.NonlinearRegression`,
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

For a user type `P <: Kirstine.Parameter`, this should return the dimension of the parameter space.

See also the [mathematical background](math.md#Design-Problems).
"""
function dimension end

## generic identity covariate parameterization ##

"""
    CopyTo <: Kirstine.CovariateParameterization

A convenience type to use when design variables and model covariates are identical.
"""
struct CopyTo{N} <: CovariateParameterization
    # Note that we have to give the field names of the _covariate_ explicitly
    # since map_to_covariate!() does not have access to the design region,
    # hence also not to the dimnames() of the design region.
    # The region's dimnames are only used for axis labels in graphics.
    # However, the _order_ of the field names here is arbitrary
    # as long as it matches what the user intends to do with the design region.
    # I.e. the following would work, but be a bit confusing:
    # struct FooCovariate <: Kirstine.Covariate
    #     a::Float64
    #     b::Float64
    # end
    # dr = DesignInterval(:b => (1, 2), :a => (3, 4))
    # CopyTo(:b, :a)
    covariate_field_names::NTuple{N,Symbol}
end

function Base.show(io::IO, cp::CopyTo)
    args = map(fn -> ':' * string(fn), cp.covariate_field_names)
    print(io, "CopyTo(" * join(args, ", "), ")")
end

"""
    CopyTo(names::Symbol...)

Construct a [`CovariateParameterization`](@ref) that copies the elements of a design point
to the fields of a [`Covariate`](@ref) with the given `names`.

This parameterization can be used with any [`Covariate`](@ref)
that has only `Float64` fields.
Note that the order of the `names`
should match the order of `dimnames(region(prob))` for the [`DesignProblem`](@ref)
in which `CopyTo` is used.
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
julia> cp = CopyTo(:a, :b)
CopyTo(:a, :b)
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
julia> struct CopyBoth <: Kirstine.CovariateParameterization end

julia> function Kirstine.map_to_covariate!(c::FooCovariate, dp, m::FooModel, cp::CopyBoth)
           c.a = dp[1]
           c.b = dp[2]
           return c
       end

```

would be slightly more efficient
since it knows the types and field names at compile time.

Also note that since the design points are assigned in the order of the names given,
`cp = CopyTo(:b, :a)` would be equivalent to setting `c.b = dp[1]` and `c.a = dp[2]`.
"""
function CopyTo(names::Symbol...)
    return CopyTo(names)
end

function map_to_covariate!(c::Covariate, dp::AbstractVector{<:Real}, m::Model, cp::CopyTo)
    dimc = length(cp.covariate_field_names)
    cnames = fieldnames(typeof(c))
    msg = "names of $(cp) don't match field names of covariate $(cnames)"
    err = ArgumentError(msg)
    length(cnames) == dimc || throw(err)
    for fn in cp.covariate_field_names
        if !(fn in cnames)
            throw(err)
        end
    end
    if dimc != length(dp)
        throw(DimensionMismatch("dimension of design point must match that of covariate"))
    end
    for i in 1:dimc
        fn = cp.covariate_field_names[i]
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

"""
    @simple_model name covariate_field_names...

Generate code for defining a [`NonlinearRegression`](@ref) model, a corresponding
[`Covariate`](@ref), and helper functions for a 1-dimensional unit of observation
with constant measurement variance.

This macro will define the two types `\$(name)Model` and `\$(name)Covariate`,
hence `name` should start with a capital letter.
The model constructor will have the measurement standard deviation `sigma` as a mandatory
argument.

# Examples

The call

```julia
@simple_model Emax dose
```

is equivalent to manually spelling out

```julia
@kwdef struct EmaxModel <: Kirstine.NonlinearRegression
    sigma::Float64
    function EmaxModel(sigma::Real)
        if sigma <= 0
            throw(ArgumentError("sigma must be positive"))
        end
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
    err_nofields = ArgumentError("no covariate field names supplied")
    !isempty(covariate_field_names) || throw(err_nofields)
    module_path = _kirstine_known_as(__module__)
    model_name = Symbol(string(name) * "Model")
    covariate_name = Symbol(string(name) * "Covariate")
    # programmatically build up the covariate struct definition
    covariate_fields = [Expr(:(::), fn, :Float64) for fn in covariate_field_names]
    covariate_struct = Expr(
        :struct,
        true, # mutable
        Expr(:<:, covariate_name, :($module_path.Covariate)),
        Expr(:block, covariate_fields...),
    )
    # Note: We have to `esc()` the definitions so that they are evaluated in the calling
    # module. This has the nice side effect of not replacing `m` with a gensym.
    return esc(
        quote
            @kwdef struct $model_name <: $module_path.NonlinearRegression
                sigma::Float64
                function $model_name(sigma::Real)
                    if sigma <= 0
                        throw(ArgumentError("sigma must be positive"))
                    end
                    return new(sigma)
                end
            end
            $covariate_struct
            function $module_path.unit_length(m::$model_name)
                return 1
            end
            function $module_path.update_model_vcov!(
                s::Matrix{Float64},
                m::$model_name,
                c::$covariate_name,
            )
                s[1, 1] = m.sigma^2
                return s
            end
            function $module_path.allocate_covariate(m::$model_name)
                return $covariate_name($(fill(0, length(covariate_field_names))...))
            end
        end,
    )
end

"""
    @simple_parameter name field_names...

Generate code for defining a subtype of [`Parameter`](@ref) with the given name and fields,
and define its dimension as `length(field_names)`.

This macro will define the type `\$(name)Parameter`,
hence `name` should start with a capital letter.
The type will have a keyword constructor.
All fields will have the type `Float64`.

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
    module_path = _kirstine_known_as(__module__)
    parameter_name = Symbol(string(name) * "Parameter")
    if isempty(field_names)
        throw(ArgumentError("no field names supplied"))
    end
    dim = length(field_names)
    par_fields = [Expr(:(::), fn, :Float64) for fn in field_names]
    par_struct = Expr(
        :struct,
        false, # immutable
        Expr(:<:, parameter_name, :($module_path.Parameter)),
        Expr(:block, par_fields...),
    )
    return esc(quote
        @kwdef $par_struct
        $module_path.dimension(p::$parameter_name) = $dim
    end)
end

function _path_to_module(from::Module, to::Module, ignore::AbstractVector{Module})
    path = Symbol[]
    visited = append!(Module[], ignore)
    _path_to_module_dfs!(path, visited, from, to)
    return path
end

function _path_to_module_dfs!(
    path::AbstractVector{Symbol},
    visited::AbstractVector{Module},
    current::Module,
    target::Module,
)
    # a module contains a reference to itself, marking it as visited prevents infinite recursion
    push!(visited, current)
    # we need all = true here since a submodule could be defined but not exported
    names_of_submodules = filter(names(current; all = true, imported = true)) do name
        return isdefined(current, name) && getfield(current, name) isa Module
    end
    submodules = [getfield(current, name) for name in names_of_submodules]
    for (name, submodule) in Iterators.zip(names_of_submodules, submodules)
        if submodule == target
            push!(path, name)
            return true
        else
            if !(submodule in visited)
                push!(path, name)
                found = _path_to_module_dfs!(path, visited, submodule, target)
                if found
                    return true
                end
            end
        end
    end
    !isempty(path) && pop!(path)
    return false
end

function _kirstine_known_as(caller::Module)
    target = parentmodule(_kirstine_known_as)
    path_elements = _path_to_module(caller, target, [Base, Core])
    if !isempty(path_elements)
        # structure from looking at dump(:(Foo.Bar.Baz))
        # Expr(:., Expr(:., :Foo, QuoteNode(:Bar)), QuoteNode(:Baz))
        # foldl((x, y) -> Expr(:., x, QuoteNode(y)), [:Foo, :Bar, :Baz])
        path_expression = foldl((x, y) -> Expr(:., x, QuoteNode(y)), path_elements)
        return path_expression
    else
        # If there is no path to Kirstine among the `imported` modules, we must be `using`
        # the package instead (else this function could not have been called).
        return :Kirstine
    end
end
