# SPDX-FileCopyrightText: 2023,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module UserImportTests
module Simple
using Test
import Kirstine as Smith

@testset "user.jl (import, simple)" begin
    @testset "@simple_model" begin
        @test_throws "no covariate field names" @macroexpand(Smith.@simple_model(Foo))

        let ex = @macroexpand(Smith.@simple_model(Foo, bar, baz)),
            _ = Base.remove_linenums!(ex),
            (a, b, c, d, e) = ex.args,
            aref = @macroexpand(
                @kwdef struct FooModel <: Smith.NonlinearRegression
                    sigma::Float64
                    function FooModel(sigma::Real)
                        if sigma <= 0
                            throw(ArgumentError("sigma must be positive"))
                        end
                        return new(sigma)
                    end
                end
            ),
            bref = :(mutable struct FooCovariate <: Smith.Covariate
                bar::Float64
                baz::Float64
            end),
            cref = :(function Smith.unit_length(m::FooModel)
                return 1
            end),
            dref = :(
                function Smith.update_model_vcov!(
                    s::Matrix{Float64},
                    m::FooModel,
                    c::FooCovariate,
                )
                    s[1, 1] = m.sigma^2
                    return s
                end
            ),
            eref = :(function Smith.allocate_covariate(m::FooModel)
                return FooCovariate(0, 0)
            end)

            @test a == Base.remove_linenums!(aref)
            @test b == Base.remove_linenums!(bref)
            @test c == Base.remove_linenums!(cref)
            @test d == Base.remove_linenums!(dref)
            @test e == Base.remove_linenums!(eref)
        end
    end

    @testset "@simple_parameter" begin
        @test_throws "no field names" @macroexpand(Smith.@simple_parameter(Foo))

        let ex = @macroexpand(Smith.@simple_parameter(Emax, e0, emax, ec50)),
            _ = Base.remove_linenums!(ex),
            (a, b) = ex.args,
            aref = @macroexpand(@kwdef struct EmaxParameter <: Smith.Parameter
                e0::Float64
                emax::Float64
                ec50::Float64
            end),
            bref = :(Smith.dimension(p::EmaxParameter) = 3)

            @test a == Base.remove_linenums!(aref)
            @test b == Base.remove_linenums!(bref)
        end
    end
end
end

module Complicated
using Test
module Foo
module Bar
import Kirstine as Smith
end
module Kirstine
surname = "Frederiksen" # someone else
end
end
Kirstine = 42
import .Foo as Baz
module Zaphod
using Test
import ..Baz

# This call internally finds the correct one of the many variables named `Kirstine`:
Baz.Bar.Smith.@simple_parameter Beeblebrox c d

@testset "user.jl (import, complicated, inner)" begin
    # The parameter and a method on it is defined inside this module...
    p = BeeblebroxParameter(1, 2)
    @test typeof(p) <: Baz.Bar.Smith.Parameter
    @test Baz.Bar.Smith.dimension(p) == 2
end
end

@testset "user.jl (import, complicated, outer)" begin
    # ... but not outside
    @test !@isdefined BeeblebroxParameter
end
end
end
