# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module UserUsingTests
using Test
using Kirstine

@testset "user.jl (import)" begin
    @testset "simple_model" begin
        @test_throws "no covariate field names" @macroexpand(@simple_model(Foo))

        let ex = @macroexpand(@simple_model(Foo, bar, baz)),
            _ = Base.remove_linenums!(ex),
            (a, b, c, d, e) = ex.args,
            aref = :(struct FooModel <: Kirstine.NonlinearRegression
                sigma::Float64
                function FooModel(; sigma::Real)
                    return new(sigma)
                end
            end),
            bref = :(mutable struct FooCovariate <: Kirstine.Covariate
                bar::Float64
                baz::Float64
            end),
            cref = :(function Kirstine.unit_length(m::FooModel)
                return 1
            end),
            dref = :(
                function Kirstine.update_model_vcov!(
                    s::Matrix{Float64},
                    m::FooModel,
                    c::FooCovariate,
                )
                    s[1, 1] = m.sigma^2
                    return s
                end
            ),
            eref = :(function Kirstine.allocate_covariate(m::FooModel)
                return FooCovariate(0, 0)
            end)

            @test a == Base.remove_linenums!(aref)
            @test b == Base.remove_linenums!(bref)
            @test c == Base.remove_linenums!(cref)
            @test d == Base.remove_linenums!(dref)
            @test e == Base.remove_linenums!(eref)
        end
    end

    @testset "define_vector_parameter" begin
        @test_throws "no field names" @macroexpand(@simple_parameter(Foo))

        let ex = @macroexpand(@simple_parameter(Emax, e0, emax, ec50)),
            _ = Base.remove_linenums!(ex),
            (a, b) = ex.args,
            aref = @macroexpand(@kwdef struct EmaxParameter <: Kirstine.Parameter
                e0::Float64
                emax::Float64
                ec50::Float64
            end),
            bref = :(Kirstine.dimension(p::EmaxParameter) = 3)

            @test a == Base.remove_linenums!(aref)
            @test b == Base.remove_linenums!(bref)
        end
    end
end
end
