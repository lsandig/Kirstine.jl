# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module DesignmeasureAbstractpointTests
using Test
using Random
using Kirstine

@testset "designmeasure-abstractpoint.jl" begin
    @testset "SignedMeasure" begin
        @test_throws "must be equal" Kirstine.SignedMeasure([1 2], [0])

        let s = Kirstine.SignedMeasure([1 2 5; 4 3 6], [1, -2, 0.5])
            @test s.weights == [1, -2, 0.5]
            @test s.atoms == [1 2 5; 4 3 6]
        end
    end

    @testset "DesignConstraints" begin
        # inner constructor
        let dr = DesignInterval(:a => (0, 1)), dcon = Kirstine.DesignConstraints
            @test_throws DimensionMismatch dcon(dr, [true], [true, false])
        end

        # outer constructor
        let dc = DOptimality(),
            dr = DesignInterval(:dose => (0, 10)),
            d = equidistant_design(dr, 3),
            dcon = Kirstine.DesignConstraints,
            c = dcon(d, dr, [2], [3])

            # Trigger erros for out-of-range weight / designpoint indices.
            @test_throws "between 1 and 3" dcon(d, dr, [0], Int64[])
            @test_throws "between 1 and 3" dcon(d, dr, [4], Int64[])
            @test_throws "between 1 and 3" dcon(d, dr, Int64[], [-1])
            @test_throws "between 1 and 3" dcon(d, dr, Int64[], [5])
            # If only one weight index is not given as fixed, it is still implicitly fixed
            # because of the simplex constraint. In this case we want to have it fixed
            # explicitly.
            @test_logs(
                (:info, "explicitly fixing implicitly fixed weight"),
                dcon(d, dr, [1, 3], Int64[])
            )
            @test c.fixw == [false, true, false]
            @test c.fixp == [false, false, true]
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
            @test_throws "outside design region" (@test_warn "dp =" check(d2, dr2))
        end
    end

    @testset "ap_rand!(DesignPoint)" begin
        # check that constraints are handled correctly
        let dr = DesignInterval(:a => (0, 5)),
            d = DesignMeasure([[1], [4.2], [3]], [0.1, 0.42, 0.48]),
            dcon = Kirstine.DesignConstraints,
            _ = Random.seed!(7531),
            arp(dd, c) = Kirstine.ap_rand!(deepcopy(dd), c),
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
            @test weights(r1)[2] == weights(d)[2]
            @test all(weights(r1)[[1, 3]] .!= weights(d)[[1, 3]])
            @test points(r2)[2] == points(d)[2]
            @test all(points(r2)[[1, 3]] .!= points(d)[[1, 3]])
            @test weights(r3)[3] == weights(d)[3]
            @test points(r3)[3] == points(d)[3]
            @test all(weights(r3)[1:2] .!= weights(d)[1:2])
            @test all(points(r3)[1:2] .!= points(d)[1:2])

            @test_logs(
                (:warn, "fixed weights already sum to one"),
                arp(mixture(0.0, one_point_design([0]), d), c4)
            )
        end
    end

    @testset "ap_diff!" begin
        let d1 = uniform_design([[1], [2], [3]]),
            d2 = DesignMeasure([[6], [5], [4]], [0, 1 / 3, 2 / 3]),
            ref1 = deepcopy(d1),
            ref2 = deepcopy(d2),
            s = Kirstine.SignedMeasure([0 0 0], zeros(3)),
            r = Kirstine.ap_diff!(s, d1, d2)

            @test s.weights == [1 / 3, 0, -1 / 3]
            @test s.atoms == [-5 -3 -1]
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
            @test d.weights == s.weights
            @test d.points == s.atoms
            # check that they don't share memory
            @test d.weights !== s.weights
            @test d.points !== s.atoms
        end
    end

    @testset "ap_dist" begin
        let p = DesignMeasure([[1], [7], [3]], [0.1, 0.3, 0.6]),
            q = DesignMeasure([[-2], [5], [8]], [0.2, 0.4, 0.4]),
            dist = Kirstine.ap_dist(p, q)

            # This should ignore the last weight in p and q (squared diff 0.04)
            @test dist == sqrt(38.02)
        end
    end

    @testset "ap_rand!(SignedMeasure)" begin
        let s = Kirstine.SignedMeasure([4 5 6], [-1, 2, 3]),
            _ = Random.seed!(7531),
            r = Kirstine.ap_rand!(s, 0, 1)

            @test all(s.weights .<= 1)
            @test all(s.weights .>= 0)
            @test all(dp -> dp[1] < 1, eachcol(s.atoms))
            @test all(dp -> dp[1] > 0, eachcol(s.atoms))
            # check that first argument is returned
            @test r === s
        end
    end

    @testset "ap_mul!(SignedMeasure)" begin
        let s1 = Kirstine.SignedMeasure([4 5 6], [0.1, 0.2, 0.3]),
            s2 = Kirstine.SignedMeasure([1e0 1e1 1e2], [1e-2, 1e-3, 1e-4]),
            ref = deepcopy(s2),
            r = Kirstine.ap_mul!(s1, s2)

            @test s1.weights == [1e-3, 2e-4, 3e-5]
            @test s1.atoms == [4 50 600]
            # check for no accidental modification
            @test s2.weights == ref.weights
            @test s2.atoms == ref.atoms
            # check that first argument is returned
            @test r === s1
        end
    end

    @testset "ap_mul!(Real)" begin
        let s = Kirstine.SignedMeasure([4 5 6], [0.1, 0.2, 0.3]),
            r = Kirstine.ap_mul!(s, 42)

            @test s.weights == [4.2, 8.4, 12.6]
            @test s.atoms == [168 210 252]
            # check that first argument is returned
            @test r === s
        end
    end

    @testset "ap_add!" begin
        let s1 = Kirstine.SignedMeasure([7 8 9], [-0.1, 0.2, 0.4]),
            s2 = Kirstine.SignedMeasure([3 2 1], [0.1, -0.2, -0.4]),
            ref = deepcopy(s2),
            r = Kirstine.ap_add!(s1, s2)

            @test s1.weights == [0, 0, 0]
            @test s1.atoms == [10 10 10]
            # check for no accidental modification
            @test s2.weights == ref.weights
            @test s2.atoms == ref.atoms
            # check that first argument is returned
            @test r === s1
        end
    end

    @testset "ap_add!" begin
        # only high-level checks, collision detection is tested below
        let d1 = DesignMeasure([[0.5], [0.5], [0.5]], [1 / 4, 1 / 4, 1 / 2]),
            d2 = deepcopy(d1),
            dr = DesignInterval(:a => (0, 1)),
            c = Kirstine.DesignConstraints(dr, fill(false, 3), fill(false, 3)),
            #                                        ignored! -----------v
            v1 = Kirstine.SignedMeasure([0.1 0.2 0.3], [1 / 8, -1 / 8, 1 / 2]),
            v1_copy = deepcopy(v1), #           ignored! ----v
            v2 = Kirstine.SignedMeasure([-1 1 0.6], [1, 0, 1 / 2]),
            # stay inside
            r1 = Kirstine.ap_add!(d1, v1, c),
            # be stopped at the boundary, both in box and in simplex
            r2 = Kirstine.ap_add!(d2, v2, c)

            # test against expected position
            @test d1 == DesignMeasure([[0.6], [0.7], [0.8]], [3 / 8, 1 / 8, 1 / 2])
            @test d2 == DesignMeasure([[0], [1], [0.8]], [3 / 4, 1 / 4, 0])
            # test that velocity is unchanged or 0
            @test v1.weights == v1_copy.weights
            @test v1.atoms == v1_copy.atoms
            @test v2.weights == [0, 0, 0]
            @test v2.atoms == [0 0 0]
            # test that d is modified and returned
            @test r1 === d1
            @test r2 === d2
        end
    end

    @testset "move_handle_fixed!" begin
        let v = Kirstine.SignedMeasure([1 / 4 1 / 4 1 / 4], [1 / 8, 3 / 8, 1 / 32]),
            mhv! = Kirstine.move_handle_fixed!,
            # one weight fixed
            r1 = mhv!(deepcopy(v), [true, false, false], fill(false, 3)),
            # one design point fixed
            r2 = mhv!(deepcopy(v), fill(false, 3), [true, false, false]),
            # special case: last weight fixed
            r3 = mhv!(deepcopy(v), [false, false, true], fill(false, 3))

            @test r1.weights == [0, 3 / 8, 1 / 32]
            @test r1.atoms == [0.25 0.25 0.25]
            @test r2.weights == [1 / 8, 3 / 8, 1 / 32]
            @test r2.atoms == [0 0.25 0.25]
            @test r3.weights == [-1 / 8, 1 / 8, 1 / 32]
            @test r3.atoms == [0.25 0.25 0.25]
        end
    end

    @testset "move_how_far" begin
        let dr = DesignInterval(:a => (0, 1)),
            d = DesignMeasure([[0.5], [0.5], [0.5]], [1 / 4, 1 / 4, 1 / 2]),
            # Note: the last weight is ignored
            # stay inside
            v1 = Kirstine.SignedMeasure([0.1 0.1 0.1], [0.1, 0.1, 0.1]),
            # move outside box
            v2 = Kirstine.SignedMeasure([1 1 0.1], [0.1, 0.1, 0.1]),
            v3 = Kirstine.SignedMeasure([1 -1 0.1], [0.1, 0.1, 0.1]),
            # move outside simplex
            v4 = Kirstine.SignedMeasure([0.1 0.1 0.1], [1, 1, 0.1]),
            v5 = Kirstine.SignedMeasure([0.1 0.1 0.1], [1, -1, 0.1]),
            # move outside both
            v6 = Kirstine.SignedMeasure([1 1 0.1], [1, 1, 0.1]),
            v7 = Kirstine.SignedMeasure([1 -1 0.1], [1, -1, 0.1]),
            v8 = Kirstine.SignedMeasure([1 1 0.1], [1, -1, 0.1]),
            v9 = Kirstine.SignedMeasure([1 -1 0.1], [1, 1, 0.1])

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
            v1 = Kirstine.SignedMeasure([0.25 0.25 0.25], fill(1 / 32, 3)),
            r1 = Kirstine.move_add_v!(deepcopy(d), 1.0, v1, dr, fill(false, 3)),
            # move out in design point right
            v2 = Kirstine.SignedMeasure([4 0.25 0.25], fill(1 / 32, 3)),
            r2 = Kirstine.move_add_v!(deepcopy(d), 1.0, v2, dr, fill(false, 3)),
            # move out in design point left
            v3 = Kirstine.SignedMeasure([-2 0.25 0.25], fill(1 / 32, 3)),
            r3 = Kirstine.move_add_v!(deepcopy(d), 1.0, v3, dr, fill(false, 3)),
            # move out in weight right, also set last weight to 0
            v4 = Kirstine.SignedMeasure([0.25 0.25 0.25], [1, -0.5, 0]),
            r4 = Kirstine.move_add_v!(deepcopy(d), 1.0, v4, dr, fill(false, 3)),
            # move out in weight left
            v5 = Kirstine.SignedMeasure([0.25 0.25 0.25], [-1, 1 / 8, 2 / 8]),
            r5 = Kirstine.move_add_v!(deepcopy(d), 1.0, v5, dr, fill(false, 3)),
            # special case: fixed last weight
            v6 = Kirstine.SignedMeasure([0.25 0.25 0.25], [1 / 4, -1 / 4, 0]),
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
