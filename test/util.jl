@testset "util.jl" begin
    @testset "define_scalar_unit_model" begin
        let ex = @macroexpand(@define_scalar_unit_model(Kirstine, Foo, bar, baz)),
            _ = Base.remove_linenums!(ex),
            (a, b, c, d, e) = ex.args,
            aref = :(struct Foo <: Kirstine.NonlinearRegression
                inv_sigma_sq::Float64
            end),
            bref = :(mutable struct FooCovariate <: Kirstine.Covariate
                bar::Float64
                baz::Float64
            end),
            cref = :(function Kirstine.unit_length(m::Foo)
                return 1
            end),
            dref = :(function Kirstine.invcov(m::Foo)
                return m.inv_sigma_sq
            end),
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
end
