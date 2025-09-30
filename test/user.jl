# SPDX-FileCopyrightText: 2023,2024,2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module UserTests
using Test
using Kirstine

include("example-testmod.jl")

# Note: module definitions can't occur inside a @testset so we place it here
module Foo
module Bar
Quux = 42
end
module Baz
module Quux end
end
import .Baz.Quux as Frob
end

@testset "user.jl" begin
    @testset "_path_to_module" begin
        # depth 1
        path = Kirstine._path_to_module(UserTests, Foo, Module[])
        @test path == [:Foo]

        # depth 3
        target = Foo.Baz.Quux
        path = Kirstine._path_to_module(UserTests, target, Module[])
        @test path == [:Foo, :Baz, :Quux]

        # aliased module, we find the original path
        target = Foo.Frob
        path = Kirstine._path_to_module(UserTests, target, Module[])
        @test path == [:Foo, :Baz, :Quux]

        # aliased module, we ignore the original name
        ignore = [Foo.Baz]
        target = Foo.Frob
        path = Kirstine._path_to_module(UserTests, target, ignore)
        @test path == [:Foo, :Frob]
        # we should not leak visited modules
        @test ignore == [Foo.Baz]

        # partial path
        target = Foo.Baz.Quux
        path = Kirstine._path_to_module(Foo, target, Module[])
        @test path == [:Baz, :Quux]
    end

    @testset "==(CopyTo)" begin
        let ct1 = CopyTo(:a, :b),
            ct2 = CopyTo(:a, :b),
            ct3 = CopyTo(:a, :c),
            ct4 = CopyTo(:b, :a)

            @test ct1 == ct2
            @test ct1 != ct3
            @test ct1 != ct4
        end
    end

    @testset "show(::CopyTo)" begin
        # check that pretty representation is parseable
        let x = CopyTo(:a, :b, :c),
            io = IOBuffer(),
            ioc = IOContext(io, :limit => false),
            _ = show(ioc, "text/plain", x),
            str_pretty = String(take!(io)),
            x_pretty = eval(Meta.parse(str_pretty))

            @test x == x_pretty
        end
    end

    @testset "map_to_covariate!(CopyTo)" begin
        let mtc! = Kirstine.map_to_covariate!,
            c1 = TestCovar2(0, 0),
            c2 = TestCovarNonFloat(0, [0.1, 0.2, 0.3]),
            dp = [42.0, 3.14],
            m = TestMod(), # any model would do
            ct1 = CopyTo(:a, :b),
            ct2 = CopyTo(:a, :b, :c),
            ct3 = CopyTo(:a, :x),
            ret = mtc!(c1, dp, m, ct1)

            # a reference is returned
            @test ret === c1
            # the fields are set correctly
            @test c1.a == 42
            @test c1.b == 3.14
            # design point is not modified
            @test dp == [42.0, 3.14]
            # covariate parameterization is not modified
            @test ct1.covariate_field_names == (:a, :b)

            @test_throws "dimension of design point" mtc!(c1, [1.0], m, ct1)
            @test_throws "don't match field names of covariate" mtc!(c1, dp, m, ct2)
            @test_throws "don't match field names of covariate" mtc!(c1, dp, m, ct3)
            @test_throws TypeError mtc!(c2, dp, m, ct1)
        end
    end
end
end
