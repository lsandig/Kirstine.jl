module DesignmeasureTests
using Test
using Random
using Kirstine

@testset "designmeasure.jl" begin
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

    @testset "DesignMeasure" begin
        # Note: the inner constructor is tested in types.jl

        @test_throws "at least two rows" DesignMeasure([1 2 3])

        # constructor from point => weight pairs
        let d = DesignMeasure([1] => 0.2, [42] => 0.3, [9] => 0.5),
            ref = DesignMeasure([[1], [42], [9]], [0.2, 0.3, 0.5])

            @test d == ref
        end

        # constructor from matrix
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

    @testset "DesignInterval" begin
        # Note: the inner constructor is tested in types.jl
        let dr = DesignInterval(:a => (0, 1), :b => (0, 2)),
            ref = DesignInterval((:a, :b), (0, 0), (1, 2))

            @test all(ref.name .== dr.name)
            @test all(ref.lowerbound .== dr.lowerbound)
            @test all(ref.upperbound .== dr.upperbound)
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

    @testset "check_compatible" begin
        let dr1 = DesignInterval(:a => (0, 1)),
            dr2 = DesignInterval(:a => (2, 3), :b => (4, 5)),
            d1 = equidistant_design(dr1, 3),
            d2 = one_point_design([1, 1]),
            check = Kirstine.check_compatible

            @test check(d1, dr1)
            @test_throws "length must match" check(d1, dr2)
            @test_throws "outside design region" check(d2, dr2)
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

    ## abstract point methods ##

    @testset "ap_random_point!" begin
        # check that constraints are handled correctly
        let dr = DesignInterval(:a => (0, 5)),
            d = DesignMeasure([[1], [4.2], [3]], [0.1, 0.42, 0.48]),
            dcon = Kirstine.DesignConstraints,
            _ = Random.seed!(7531),
            arp(dd, c) = Kirstine.ap_random_point!(deepcopy(dd), c),
            # nothng fixed
            c0 = dcon(dr, fill(false, 3), fill(false, 3)),
            r0 = arp(d, c0),
            # fixed weight
            c1 = dcon(dr, [false, true, false], fill(false, 3)),
            r1 = arp(d, c1),
            # fixed design point
            c2 = dcon(dr, fill(false, 3), [false, true, false]),
            r2 = arp(d, c2),
            # both fixed
            c3 = dcon(dr, [false, false, true], [false, false, true]),
            r3 = arp(d, c3)
            # some weights fixed that already sum to one (see below)
            c4 = dcon(dr, [false, true, true, true], fill(false, 4))

            # check that weights are non-negative and sum to one
            @test sum(weights(r0)) ≈ 1
            @test sum(weights(r1)) ≈ 1
            @test sum(weights(r2)) ≈ 1
            @test sum(weights(r3)) ≈ 1
            @test all(weights(r0) .>= 0)
            @test all(weights(r1) .>= 0)
            @test all(weights(r2) .>= 0)
            @test all(weights(r3) .>= 0)

            # check that box constraints are honored
            @test Kirstine.check_compatible(r0, dr)
            @test Kirstine.check_compatible(r1, dr)
            @test Kirstine.check_compatible(r2, dr)
            @test Kirstine.check_compatible(r3, dr)

            # check that fixed weight/point constraints are honored
            @test r1.weight[2] == d.weight[2]
            @test all(r1.weight[[1, 3]] .!= d.weight[[1, 3]])
            @test r2.designpoint[2] == d.designpoint[2]
            @test all(r2.designpoint[[1, 3]] .!= d.designpoint[[1, 3]])
            @test r3.weight[3] == d.weight[3]
            @test r3.designpoint[3] == d.designpoint[3]
            @test all(r3.weight[1:2] .!= d.weight[1:2])
            @test all(r3.designpoint[1:2] .!= d.designpoint[1:2])

            @test_logs(
                (:warn, "fixed weights already sum to one"),
                arp(mixture(0.0, one_point_design([0]), d), c4)
            )
        end
    end

    @testset "ap_difference!" begin
        let d1 = uniform_design([[1], [2], [3]]),
            d2 = DesignMeasure([[6], [5], [4]], [0, 1 / 3, 2 / 3]),
            ref1 = deepcopy(d1),
            ref2 = deepcopy(d2),
            s = Kirstine.SignedMeasure([[0], [0], [0]], zeros(3)),
            r = Kirstine.ap_difference!(s, d1, d2)

            @test s.weight == [1 / 3, 0, -1 / 3]
            @test s.atom == [[-5], [-3], [-1]]
            # check against accidental modification
            @test d1 == ref1
            @test d2 == ref2
            # check that first argument is returned
            @test r === s
        end
    end

    @testset "ap_copy!" begin
        let from = DesignMeasure([[1], [2]], [0.1, 0.9]),
            to = DesignMeasure([[3], [4]], [0.2, 0.8]),
            ref = deepcopy(from),
            r = Kirstine.ap_copy!(to, from)

            # check that source was not accidentally modified
            @test from == ref
            # check equality,...
            @test to == from
            # ... but non-egality
            @test to !== from
            # check that first argument is returned
            @test r === to
        end
    end

    @testset "ap_as_difference" begin
        let d = uniform_design([[0], [1], [2]]), s = Kirstine.ap_as_difference(d)

            # Note: dont use allocating weights()/designpoints() accessors
            @test d.weight == s.weight
            @test d.designpoint == s.atom
            # check that they don't share memory
            @test d.weight !== s.weight
            @test d.designpoint !== s.atom
        end
    end

    @testset "ap_random_difference!" begin
        let s = Kirstine.SignedMeasure([[4], [5], [6]], [-1, 2, 3]),
            _ = Random.seed!(7531),
            r = Kirstine.ap_random_difference!(s)

            @test all(s.weight .<= 1)
            @test all(s.weight .>= 0)
            @test all(dp -> dp[1] < 1, s.atom)
            @test all(dp -> dp[1] > 0, s.atom)
            # check that first argument is returned
            @test r === s
        end
    end

    @testset "ap_mul_hadamard!" begin
        let s1 = Kirstine.SignedMeasure([[4], [5], [6]], [0.1, 0.2, 0.3]),
            s2 = Kirstine.SignedMeasure([[1e0], [1e1], [1e2]], [1e-2, 1e-3, 1e-4]),
            ref = deepcopy(s2),
            r = Kirstine.ap_mul_hadamard!(s1, s2)

            @test s1.weight == [1e-3, 2e-4, 3e-5]
            @test s1.atom == [[4], [50], [600]]
            # check for no accidental modification
            @test s2.weight == ref.weight
            @test s2.atom == ref.atom
            # check that first argument is returned
            @test r === s1
        end
    end

    @testset "ap_mul_scalar!" begin
        let s = Kirstine.SignedMeasure([[4], [5], [6]], [0.1, 0.2, 0.3]),
            r = Kirstine.ap_mul_scalar!(s, 42)

            @test s.weight == [4.2, 8.4, 12.6]
            @test s.atom == [[168], [210], [252]]
            # check that first argument is returned
            @test r === s
        end
    end

    @testset "ap_add!" begin
        let s1 = Kirstine.SignedMeasure([[7], [8], [9]], [-0.1, 0.2, 0.4]),
            s2 = Kirstine.SignedMeasure([[3], [2], [1]], [0.1, -0.2, -0.4]),
            ref = deepcopy(s2),
            r = Kirstine.ap_add!(s1, s2)

            @test s1.weight == [0, 0, 0]
            @test s1.atom == [[10], [10], [10]]
            # check for no accidental modification
            @test s2.weight == ref.weight
            @test s2.atom == ref.atom
            # check that first argument is returned
            @test r === s1
        end
    end

    @testset "ap_move!" begin
        # only high-level checks, collision detection is tested below
        let d1 = DesignMeasure([[0.5], [0.5], [0.5]], [1 / 4, 1 / 4, 1 / 2]),
            d2 = deepcopy(d1),
            dr = DesignInterval(:a => (0, 1)),
            c = Kirstine.DesignConstraints(dr, fill(false, 3), fill(false, 3)),
            #                                                ignored! -----------v
            v1 = Kirstine.SignedMeasure([[0.1], [0.2], [0.3]], [1 / 8, -1 / 8, 1 / 2]),
            v1_copy = deepcopy(v1), #                   ignored! ----v
            v2 = Kirstine.SignedMeasure([[-1], [1], [0.6]], [1, 0, 1 / 2]),
            # stay inside
            r1 = Kirstine.ap_move!(d1, v1, c),
            # be stopped at the boundary, both in box and in simplex
            r2 = Kirstine.ap_move!(d2, v2, c)

            # test against expected position
            @test d1 == DesignMeasure([[0.6], [0.7], [0.8]], [3 / 8, 1 / 8, 1 / 2])
            @test d2 == DesignMeasure([[0], [1], [0.8]], [3 / 4, 1 / 4, 0])
            # test that velocity is unchanged or 0
            @test v1.weight == v1_copy.weight
            @test v1.atom == v1_copy.atom
            @test v2.weight == [0, 0, 0]
            @test v2.atom == [[0], [0], [0]]
            # test that d is modified and returned
            @test r1 === d1
            @test r2 === d2
        end
    end

    @testset "move_handle_fixed!" begin
        let v = Kirstine.SignedMeasure([[1 / 4], [1 / 4], [1 / 4]], [1 / 8, 3 / 8, 1 / 32]),
            mhv! = Kirstine.move_handle_fixed!,
            # one weight fixed
            r1 = mhv!(deepcopy(v), [true, false, false], fill(false, 3)),
            # one design point fixed
            r2 = mhv!(deepcopy(v), fill(false, 3), [true, false, false]),
            # special case: last weight fixed
            r3 = mhv!(deepcopy(v), [false, false, true], fill(false, 3))

            @test r1.weight == [0, 3 / 8, 1 / 32]
            @test r1.atom == [[1 / 4], [1 / 4], [1 / 4]]
            @test r2.weight == [1 / 8, 3 / 8, 1 / 32]
            @test r2.atom == [[0], [0.25], [0.25]]
            @test r3.weight == [-1 / 8, 1 / 8, 1 / 32]
            @test r3.atom == [[1 / 4], [1 / 4], [1 / 4]]
        end
    end

    @testset "move_how_far" begin
        let dr = DesignInterval(:a => (0, 1)),
            d = DesignMeasure([[0.5], [0.5], [0.5]], [1 / 4, 1 / 4, 1 / 2]),
            # Note: the last weight is ignored
            # stay inside
            v1 = Kirstine.SignedMeasure([[0.1], [0.1], [0.1]], [0.1, 0.1, 0.1]),
            # move outside box
            v2 = Kirstine.SignedMeasure([[1], [1], [0.1]], [0.1, 0.1, 0.1]),
            v3 = Kirstine.SignedMeasure([[1], [-1], [0.1]], [0.1, 0.1, 0.1]),
            # move outside simplex
            v4 = Kirstine.SignedMeasure([[0.1], [0.1], [0.1]], [1, 1, 0.1]),
            v5 = Kirstine.SignedMeasure([[0.1], [0.1], [0.1]], [1, -1, 0.1]),
            # move outside both
            v6 = Kirstine.SignedMeasure([[1], [1], [0.1]], [1, 1, 0.1]),
            v7 = Kirstine.SignedMeasure([[1], [-1], [0.1]], [1, -1, 0.1]),
            v8 = Kirstine.SignedMeasure([[1], [1], [0.1]], [1, -1, 0.1]),
            v9 = Kirstine.SignedMeasure([[1], [-1], [0.1]], [1, 1, 0.1])

            @test Kirstine.move_how_far(d, v1, dr) == 1.0
            @test Kirstine.move_how_far(d, v2, dr) == 1 / 2
            @test Kirstine.move_how_far(d, v3, dr) == 1 / 2
            # the remaining ones are always stopped at the simplex boundary
            @test Kirstine.move_how_far(d, v4, dr) == 1 / 4
            @test Kirstine.move_how_far(d, v5, dr) == 1 / 4
            @test Kirstine.move_how_far(d, v6, dr) == 1 / 4
            @test Kirstine.move_how_far(d, v7, dr) == 1 / 4
            @test Kirstine.move_how_far(d, v8, dr) == 1 / 4
            @test Kirstine.move_how_far(d, v9, dr) == 1 / 4
        end
    end

    @testset "how_far_right" begin
        # Can we move all the way from x to x + tv, or are we stopped at ub?
        #
        # 0   1   2   3   4   5   6
        # |---|---|---|---|---|---|
        #             x      ub
        #                            x    t  v ub
        @test Kirstine.how_far_right(3, 1.5, 1, 5) == 1.5
        @test Kirstine.how_far_right(3, 1.0, 4, 5) == 0.5
        @test Kirstine.how_far_right(3, 1.0, 3, 5) == 2 / 3
    end

    @testset "how_far_left" begin
        # Can we move all the way from x to x + tv, or are we stopped at lb?
        #
        # 0   1   2   3   4   5   6
        # |---|---|---|---|---|---|
        #    lb       x
        #                           x    t   v lb
        @test Kirstine.how_far_left(3, 1.5, -1, 1) == 1.5
        @test Kirstine.how_far_left(3, 1.0, -4, 1) == 0.5
        @test Kirstine.how_far_left(3, 1.0, -3, 1) == 2 / 3
    end

    @testset "how_far_simplexdiag" begin
        # Suppose there are three elements, the third being implicit. Can we move from x to
        # x + tv, or are we stopped at the simplex diagonal face?
        #
        #  x_2 direction
        #  |
        #  |     x+tv
        #  |
        #  |\
        #  | \
        #  |  \
        #  |   \
        #  |    \
        #  |     \
        #  | x    \
        # 0|_______\_____ x_1 direction
        #  0
        let x = [0.25, 0.25], v1 = [1, 2], v2 = [0.1, 0.2]

            # This moves up to the intersection at [5/12, 7/12],
            # which is 1/6 the length of v1.
            @test Kirstine.how_far_simplexdiag(sum(x), 1.0, sum(v1)) == 1 / 6
            # This stays inside.
            @test Kirstine.how_far_simplexdiag(sum(x), 1.0, sum(v2)) == 1.0
        end
    end

    @testset "move_add_v!" begin
        let dr = DesignInterval(:a => (1, 3)),
            d = DesignMeasure([[1.5], [2], [2.5]], [3 / 8, 4 / 8, 1 / 8]),
            # Note:
            #
            #  * The last weight index of v will be ignored when moving.
            #  * Movements out of the interval/simplex are usually only off by eps(),
            #    but we can test the logic with larger deviations.
            #
            # stay inside
            v1 = Kirstine.SignedMeasure([[1 / 4], [1 / 4], [1 / 4]], fill(1 / 32, 3)),
            r1 = Kirstine.move_add_v!(deepcopy(d), 1.0, v1, dr, fill(false, 3)),
            # move out in design point right
            v2 = Kirstine.SignedMeasure([[4], [0.25], [0.25]], fill(1 / 32, 3)),
            r2 = Kirstine.move_add_v!(deepcopy(d), 1.0, v2, dr, fill(false, 3)),
            # move out in design point left
            v3 = Kirstine.SignedMeasure([[-2], [0.25], [0.25]], fill(1 / 32, 3)),
            r3 = Kirstine.move_add_v!(deepcopy(d), 1.0, v3, dr, fill(false, 3)),
            # move out in weight right, also set last weight to 0
            v4 = Kirstine.SignedMeasure([[1 / 4], [1 / 4], [1 / 4]], [1, -0.5, 0]),
            r4 = Kirstine.move_add_v!(deepcopy(d), 1.0, v4, dr, fill(false, 3)),
            # move out in weight left
            v5 = Kirstine.SignedMeasure([[1 / 4], [1 / 4], [1 / 4]], [-1, 1 / 8, 2 / 8]),
            r5 = Kirstine.move_add_v!(deepcopy(d), 1.0, v5, dr, fill(false, 3)),
            # special case: fixed last weight
            v6 = Kirstine.SignedMeasure([[1 / 4], [1 / 4], [1 / 4]], [1 / 4, -1 / 4, 0]),
            r6 = Kirstine.move_add_v!(deepcopy(d), 1.0, v6, dr, [false, false, true])

            @test r1 == DesignMeasure([[1.75], [2.25], [2.75]], [13 / 32, 17 / 32, 1 / 16])
            @test r2 == DesignMeasure([[3.00], [2.25], [2.75]], [13 / 32, 17 / 32, 1 / 16])
            @test r3 == DesignMeasure([[1.00], [2.25], [2.75]], [13 / 32, 17 / 32, 1 / 16])
            @test r4 == DesignMeasure([[1.75], [2.25], [2.75]], [1, 0, 0])
            @test r5 == DesignMeasure([[1.75], [2.25], [2.75]], [0, 5 / 8, 3 / 8])
            @test r6 == DesignMeasure([[1.75], [2.25], [2.75]], [5 / 8, 2 / 8, 1 / 8])
        end
    end
end
end
