# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module UserTests
using Test
using Kirstine

@testset "user.jl" begin
    @testset "define_scalar_unit_model" begin
        let ex = @macroexpand(@define_scalar_unit_model(Kirstine, Foo, bar, baz)),
            _ = Base.remove_linenums!(ex),
            (a, b, c, d, e) = ex.args,
            aref = :(struct Foo <: Kirstine.NonlinearRegression
                sigma_squared::Float64
            end),
            bref = :(mutable struct FooCovariate <: Kirstine.Covariate
                bar::Float64
                baz::Float64
            end),
            cref = :(function Kirstine.unit_length(m::Foo)
                return 1
            end),
            dref = :(
                function Kirstine.update_model_vcov!(
                    s::Matrix{Float64},
                    c::FooCovariate,
                    m::Foo,
                )
                    s[1, 1] = m.sigma_squared
                    return s
                end
            ),
            eref = :(function Kirstine.allocate_covariate(m::Foo)
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
        @test_throws "no field names" @macroexpand(@define_vector_parameter(Kirstine, Foo))

        let ex = @macroexpand(@define_vector_parameter(Kirstine, EmaxPar, e0, emax, ec50)),
            _ = Base.remove_linenums!(ex),
            (a, b) = ex.args,
            aref = @macroexpand(@kwdef struct EmaxPar <: Kirstine.Parameter
                e0::Float64
                emax::Float64
                ec50::Float64
            end),
            bref = :(Kirstine.dimension(p::EmaxPar) = 3)

            @test a == Base.remove_linenums!(aref)
            @test b == Base.remove_linenums!(bref)
        end
    end
end
end
