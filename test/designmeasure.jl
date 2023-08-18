# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignmeasureTests
using Test
using Kirstine

@testset "designmeasure.jl" begin
    @testset "DesignMeasure" begin
        # inner constructor
        @test_throws "must be equal" DesignMeasure([[1], [2]], [0])
        @test_throws "identical lengths" DesignMeasure([[1], [2, 3]], [0.5, 0.5])
        @test_throws "non-negative" DesignMeasure([[1], [2]], [-0.5, 1.5])
        @test_throws "sum to one" DesignMeasure([[1], [2]], [0.1, 0.2])

        let d = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5])
            @test d.weight == [0.2, 0.3, 0.5]
            @test d.designpoint == [[1], [42], [9]]
        end

        # constructor from point => weight pairs
        let d = DesignMeasure([1] => 0.2, [42] => 0.3, [9] => 0.5),
            ref = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5])

            @test d == ref
        end

        # constructor from matrix
        @test_throws "at least two rows" DesignMeasure([1 2 3])
        let d = DesignMeasure([[7, 4], [8, 5], [9, 6]], [0.5, 0.2, 0.3]),
            d_as_matrix = [0.5 0.2 0.3; 7 8 9; 4 5 6],
            dirac = one_point_design([2, 3]),
            dirac_as_matrix = reshape([1, 2, 3], :, 1)

            @test d == DesignMeasure(d_as_matrix)
            @test dirac == DesignMeasure(dirac_as_matrix)
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
        let d = equidistant_design(DesignInterval(:a => (1, 4)), 4),
            ref = DesignMeasure([[i] for i in 1:4], fill(0.25, 4))

            @test d == ref
        end
    end

    @testset "random_design" begin
        let dr = DesignInterval(:a => (2, 4), :b => (-1, 3)), d = random_design(dr, 4)
            for k in 1:4
                @test all(d.designpoint[k] .<= upperbound(dr))
                @test all(d.designpoint[k] .>= lowerbound(dr))
            end
        end
    end

    @testset "designpoints" begin
        # accessor should return a copy
        let d = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5])
            @test designpoints(d) == [[1], [42], [9]]
            @test designpoints(d) !== d.designpoint
        end
    end

    @testset "weights" begin
        # accessor should return a copy
        let d = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5])
            @test weights(d) == [0.2, 0.3, 0.5]
            @test weights(d) !== d.weight
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
            @test d1 == sort_designpoints(d2)
        end
    end

    @testset "as_matrix" begin
        let d = DesignMeasure([[7, 4], [8, 5], [9, 6]], [0.5, 0.2, 0.3]),
            d_as_matrix = [0.5 0.2 0.3; 7 8 9; 4 5 6],
            m = [0.1 0.2 0.3 0.4; 1 2 3 4],
            m_as_designmeasure = DesignMeasure([[1], [2], [3], [4]], [0.1, 0.2, 0.3, 0.4]),
            dirac = one_point_design([2, 3]),
            dirac_as_matrix = reshape([1, 2, 3], :, 1)

            @test m == as_matrix(m_as_designmeasure)
            @test dirac_as_matrix == as_matrix(dirac)
            # roundtrips
            @test d == DesignMeasure(as_matrix(d))
            @test m == as_matrix(DesignMeasure(m))
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
    end

    # sorting
    @testset "sort_designpoints" begin
        let d = DesignMeasure([[3, 4], [2, 1], [1, 1], [2, 3]], [0.4, 0.2, 0.3, 0.1]),
            refp = DesignMeasure([[1, 1], [2, 1], [2, 3], [3, 4]], [0.3, 0.2, 0.1, 0.4])

            @test sort_designpoints(d) == refp
            # check that a copy is returned
            @test sort_designpoints(refp) !== refp
            @test sort_designpoints(d; rev = true).weight == reverse(refp.weight)
            @test sort_designpoints(d; rev = true).designpoint == reverse(refp.designpoint)
        end
    end

    @testset "sort_weights" begin
        let d = DesignMeasure([[3, 4], [2, 1], [1, 1], [2, 3]], [0.4, 0.2, 0.3, 0.1]),
            refw = DesignMeasure([[2, 3], [2, 1], [1, 1], [3, 4]], [0.1, 0.2, 0.3, 0.4])

            @test sort_weights(d) == refw
            # check that a copy is returned
            @test sort_weights(refw) !== refw
            @test sort_weights(d; rev = true).weight == reverse(refw.weight)
            @test sort_weights(d; rev = true).designpoint == reverse(refw.designpoint)
        end
    end
end
end
