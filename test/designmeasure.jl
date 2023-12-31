# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignmeasureTests
using Test
using Random: seed!
using Kirstine

@testset "designmeasure.jl" begin
    @testset "DesignMeasure" begin
        # inner constructor
        @test_throws "must be equal" DesignMeasure([[1], [2]], [0])
        @test_throws "identical lengths" DesignMeasure([[1], [2, 3]], [0.5, 0.5])
        @test_throws "non-negative" DesignMeasure([[1], [2]], [-0.5, 1.5])
        @test_throws "sum to one" DesignMeasure([[1], [2]], [0.1, 0.2])

        let pts = [[1], [42], [9]], d = DesignMeasure(pts, [0.2, 0.3, 0.5])
            @test weights(d) == [0.2, 0.3, 0.5]
            @test points(d) == pts
            # When constructed from a vector of design points, memory should not be shared.
            points(d)[1][1] = 0
            @test pts[1][1] == 1
            @test points(d)[1][1] == 0
        end

        # constructor from point => weight pairs
        let d = DesignMeasure([1] => 0.2, [42] => 0.3, [9] => 0.5),
            ref = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5])

            @test d == ref
        end
    end

    @testset "one_point_design" begin
        let d = one_point_design([42]), ref = DesignMeasure([[42]], [1])
            @test d == ref
        end
    end

    @testset "uniform_design" begin
        let d = uniform_design([[1], [2], [3], [4]]),
            ref = DesignMeasure([[i] for i in 1:4], fill(0.25, 4))

            @test d == ref
        end
    end

    @testset "equidistant_design" begin
        let dr = DesignInterval(:a => (1, 4)),
            d = equidistant_design(dr, 4),
            ref = DesignMeasure([[i] for i in 1:4], fill(0.25, 4))

            @test_throws "at least K = 2" equidistant_design(dr, 1)
            @test d == ref
        end
    end

    @testset "random_design" begin
        let dr = DesignInterval(:a => (2, 4), :b => (-1, 3)),
            _ = seed!(7531),
            d = random_design(dr, 4)

            for k in 1:4
                @test all(points(d)[k] .<= upperbound(dr))
                @test all(points(d)[k] .>= lowerbound(dr))
            end
        end
    end

    @testset "points" begin
        # accessor should return a reference
        let d = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5])
            @test points(d) == [[1], [42], [9]]
            # points() should return a reference, so we can modify d.
            points(d)[1][1] = 0
            @test points(d)[1] == [0]
            points(d)[1] = [2]
            @test points(d)[1] == [2]
        end
    end

    @testset "weights" begin
        # accessor should return a reference
        let d = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5])
            @test weights(d) == [0.2, 0.3, 0.5]
            @test weights(d) === d.weights
        end
    end

    @testset "show" begin
        # check that both compact and pretty representation are parseable
        let d = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5]),
            str_compact = repr(d),
            io = IOBuffer(),
            ioc = IOContext(io, :limit => false),
            _ = show(ioc, "text/plain", d),
            str_pretty = String(take!(io)),
            d_compact = eval(Meta.parse(str_compact)),
            d_pretty = eval(Meta.parse(str_pretty))

            @test d == d_compact
            @test d == d_pretty
        end
    end

    @testset "==" begin
        let d1 = DesignMeasure([[1, 2], [3, 4]], [0.1, 0.9]),
            d2 = DesignMeasure([[3, 4], [1, 2]], [0.9, 0.1])

            @test d1 != d2
            @test d1 == sort_points(d2)
        end
    end

    @testset "mixture" begin
        let d = equidistant_design(DesignInterval(:a => (1, 4)), 4),
            dirac = one_point_design([5]),
            dirac2d = one_point_design([5, 6]),
            supp = [[5], [1], [2], [3], [4]]

            @test mixture(0.2, dirac, d) == uniform_design(supp)
            # results should not be simplified
            @test mixture(0.0, dirac, d) == DesignMeasure(supp, [0.0; fill(0.25, 4)])
            @test_throws "identical length" mixture(0.5, dirac, dirac2d)
            @test_throws "between 0 and 1" mixture(1.1, dirac, d)
        end
    end

    @testset "apportion" begin
        # reference values from Pukelsheim p. 310 (Exhibit 12.2)
        let d = DesignMeasure([[i] for i in 1:3], [1 / 6, 1 / 3, 1 / 2]),
            n = 7:12,
            ref = [[2, 2, 3], [1, 3, 4], [2, 3, 4], [2, 3, 5], [2, 4, 5], [2, 4, 6]]

            for i in 1:length(n)
                @test all(apportion(d, n[i]) .== ref[i])
            end
        end
    end

    @testset "simplify_drop" begin
        let d = DesignMeasure([[1], [2], [3]], [1e-6, 0.5 - 5e-7, 0.5 - 5e-7])
            @test simplify_drop(d, 1e-4) == uniform_design([[2], [3]])
            # drop also on equality
            @test simplify_drop(d, 1e-6) == uniform_design([[2], [3]])
            # return unchanged copy if nothing is dropped
            @test simplify_drop(d, 1e-7) == d
            @test simplify_drop(d, 1e-7) !== d
        end

        # One-point-designs should be returned as an unchanged copy.
        let o = one_point_design([42]), o_simp = simplify_drop(o, 1e-4)
            @test o !== o_simp
            @test o == o_simp
        end
    end

    @testset "simplify_merge" begin
        let d = DesignMeasure([[1, 1], [5, 1], [3, 10]], [0.3, 0.1, 0.6]),
            dr = DesignInterval(:a => (0, 100), :b => (0, 100)),
            ref = DesignMeasure([[2, 1], [3, 10]], [0.4, 0.6])

            @test simplify_merge(d, dr, 0.05) == ref
            # merge also on equality
            @test simplify_merge(d, dr, 0.04) == ref
            # return unchanged copy if nothing is dropped
            @test simplify_merge(d, dr, 0.03) == d
            @test simplify_merge(d, dr, 0.03) !== d
        end

        # One-point-designs should be returned as an unchanged copy.
        let o = one_point_design([42]),
            dr = DesignInterval(:a => (0, 100), :b => (1, 100)),
            o_simp = simplify_merge(o, dr, 0.05)

            @test o !== o_simp
            @test o == o_simp
        end

        # Unmerged points should stay the same.
        let dr = DesignInterval(:a => (0, 0.95)),
            d = uniform_design([[0.1], [0.5], [0.8], [0.81]])

            # We don't want to scale [0.5] into the unit cube and back,
            # since 0.5 / 0.95 * 0.95 != 0.5 numerically.
            @test d == simplify_merge(d, dr, 0)
            @test points(d)[1:2] == points(simplify_merge(d, dr, 0.1))[2:3]
        end
    end

    # sorting
    @testset "sort_points" begin
        let d = DesignMeasure([[3, 4], [2, 1], [1, 1], [2, 3]], [0.4, 0.2, 0.3, 0.1]),
            d_sorted = sort_points(d),
            refp = DesignMeasure([[1, 1], [2, 1], [2, 3], [3, 4]], [0.3, 0.2, 0.1, 0.4])

            @test d_sorted == refp
            # check that a copy is returned
            points(d_sorted)[1][1] = 42
            @test points(d_sorted)[1] == [42, 1]
            @test points(d)[3] == [1, 1]
            @test sort_points(refp) !== refp
            @test weights(sort_points(d; rev = true)) == reverse(weights(refp))
            @test points(sort_points(d; rev = true)) == reverse(points(refp))
        end
    end

    @testset "sort_weights" begin
        let d = DesignMeasure([[3, 4], [2, 1], [1, 1], [2, 3]], [0.4, 0.2, 0.3, 0.1]),
            refw = DesignMeasure([[2, 3], [2, 1], [1, 1], [3, 4]], [0.1, 0.2, 0.3, 0.4])

            @test sort_weights(d) == refw
            # check that a copy is returned
            @test sort_weights(refw) !== refw
            @test weights(sort_weights(d; rev = true)) == reverse(weights(refw))
            @test points(sort_weights(d; rev = true)) == reverse(points(refw))
        end
    end
end
end
