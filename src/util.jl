# utitliy functions

"""
    @define_scalar_unit_model module_name model_name covariate_field_names...

Generate code for defining a [`NonlinearRegression`](@ref) model, a corresponding
[`Covariate`](@ref), and helper functions for a 1-dimensional unit of observation.

The call

```julia
julia> @define_scalar_unit_model Kirstine FooMod bar baz

```

is equivalent to manually spelling out

```julia
struct FooMod <: Kirstine.NonlinearRegression
    inv_sigma_sq::Float64
end
mutable struct FooModCovariate <: Kirstine.Covariate
    bar::Float64
    baz::Float64
end
Kirstine.unit_length(m::FooMod) = 1
Kirstine.invcov(m::FooMod) = m.inv_sigma_sq
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
    return esc(quote
        struct $model_name <: $module_name.NonlinearRegression
            inv_sigma_sq::Float64
        end
        $covar_struct_expression
        function $module_name.unit_length(m::$model_name)
            return 1
        end
        function $module_name.invcov(m::$model_name)
            return m.inv_sigma_sq
        end
        function $module_name.allocate_covariate(m::$model_name)
            return $covar_name($(fill(0, length(covariate_field_names))...))
        end
    end)
end
