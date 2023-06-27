@testset "designmeasure" begin
    # error handling in constructors
    @test_throws "must be equal" DesignMeasure([0], [[1], [2]])
    @test_throws "identical lengths" DesignMeasure([0.5, 0.5], [[1], [2, 3]])
    @test_throws "non-negative" DesignMeasure([-0.5, 1.5], [[1], [2]])
    @test_throws "sum to one" DesignMeasure([0.1, 0.2], [[1], [2]])
    @test_throws "at least two rows" DesignMeasure([1 2 3])
    @test_throws "must be identical" DesignSpace([:a], [1, 2], [3, 4])
    @test_throws "must be identical" DesignSpace([:a, :b], [2], [3, 4])
    @test_throws "must be identical" DesignSpace([:a, :b], [1, 2], [4])
    @test_throws "strictly larger" DesignSpace([:a, :b], [1, 2], [0, 4])
    @test_throws "strictly larger" DesignSpace([:a, :b], [1, 2], [1, 4])

    # outer constructors
    let ds = DesignSpace(:a => (0, 1), :b => (0, 2)),
        ref = DesignSpace((:a, :b), (0, 0), (1, 2))

        @test all(ref.name .== ds.name)
        @test all(ref.lowerbound .== ds.lowerbound)
        @test all(ref.upperbound .== ds.upperbound)
    end
    let d = one_point_design([42]), ref = DesignMeasure([1], [[42]])
        @test all(d.weight .== ref.weight)
        @test all(d.designpoint .== ref.designpoint)
    end
    let d = uniform_design([[1], [2], [3], [4]]),
        ref = DesignMeasure(fill(0.25, 4), [[i] for i in 1:4])

        @test all(d.weight .== ref.weight)
        @test all(d.designpoint .== ref.designpoint)
    end
    let d = equidistant_design(DesignSpace(:a => (1, 4)), 4),
        ref = DesignMeasure(fill(0.25, 4), [[i] for i in 1:4])

        @test all(d.weight .== ref.weight)
        @test all(d.designpoint .== ref.designpoint)
    end
    let d = DesignMeasure([1] => 0.2, [42] => 0.3, [9] => 0.5),
        ref = DesignMeasure([0.2, 0.3, 0.5], [[1], [42], [9]])

        @test all(d.weight .== ref.weight)
        @test all(d.designpoint .== ref.designpoint)
    end
    let d = DesignMeasure([0.5, 0.2, 0.3], [[7, 4], [8, 5], [9, 6]]),
        d_as_matrix = [0.5 0.2 0.3; 7 8 9; 4 5 6],
        m = [0.1 0.2 0.3 0.4; 1 2 3 4],
        m_as_designmeasure = DesignMeasure([0.1, 0.2, 0.3, 0.4], [[1], [2], [3], [4]]),
        dirac = one_point_design([2, 3]),
        dirac_as_matrix = reshape([1, 2, 3], :, 1)

        # conversion in both directions
        @test d.weight == DesignMeasure(d_as_matrix).weight
        @test d.designpoint == DesignMeasure(d_as_matrix).designpoint
        @test m == as_matrix(m_as_designmeasure)
        # roundtrips
        @test d.weight == DesignMeasure(as_matrix(d)).weight
        @test d.designpoint == DesignMeasure(as_matrix(d)).designpoint
        @test m == as_matrix(DesignMeasure(m))
        # one-point designs work as expected
        @test dirac_as_matrix == as_matrix(dirac)
        @test dirac.designpoint == DesignMeasure(dirac_as_matrix).designpoint
        @test dirac.designpoint == DesignMeasure(dirac_as_matrix).designpoint
    end

    # check that both compact and pretty representation are parseable
    let d = DesignMeasure([0.2, 0.3, 0.5], [[1], [42], [9]]),
        str_compact = repr(d),
        io = IOBuffer(),
        ioc = IOContext(io, :limit => false),
        _ = show(ioc, "text/plain", d),
        str_pretty = String(take!(io)),
        d_compact = eval(Meta.parse(str_compact)),
        d_pretty = eval(Meta.parse(str_pretty))

        @test d.weight == d_compact.weight
        @test d.designpoint == d_compact.designpoint
        @test d.weight == d_pretty.weight
        @test d.designpoint == d_pretty.designpoint
    end

    # mixtures
    let d = equidistant_design(DesignSpace(:a => (1, 4)), 4),
        dirac = one_point_design([5]),
        dirac2d = one_point_design([5, 6]),
        mix = mixture(0.2, dirac, d)

        @test all(mix.weight .== 0.2)
        @test all(mix.designpoint .== [[[5]]; d.designpoint])
        @test_throws "identical length" mixture(0.5, dirac, dirac2d)
        @test_throws "between 0 and 1" mixture(1.1, dirac, d)
    end

    # apportionment, reference values from Pukelsheim p. 310 (Exhibit 12.2)
    let d = DesignMeasure([1 / 6, 1 / 3, 1 / 2], [[i] for i in 1:3]),
        n = 7:12,
        ref = [[2, 2, 3], [1, 3, 4], [2, 3, 4], [2, 3, 5], [2, 4, 5], [2, 4, 6]]

        for i in 1:length(n)
            @test all(apportion(d, n[i]) .== ref[i])
        end
    end

    # simplification
    let d = DesignMeasure([1e-6, 0.5 - 5e-7, 0.5 - 5e-7], [[1], [2], [3]]),
        s = simplify_drop(d, 1e-4),
        ref = uniform_design([[2], [3]])

        @test all(s.weight .== ref.weight)
        @test all(s.designpoint .== ref.designpoint)
    end
    let d = DesignMeasure([0.3, 0.1, 0.6], [[1, 1], [5, 1], [3, 10]]),
        ds = DesignSpace(:a => (0, 100), :b => (1, 100)),
        s = simplify_merge(d, ds, 0.05),
        ref = DesignMeasure([0.4, 0.6], [[2, 1], [3, 10]])

        @test all(s.weight .== ref.weight)
        @test all(s.designpoint .== ref.designpoint)
    end

    # sorting
    let d = DesignMeasure([0.4, 0.2, 0.3, 0.1], [[3, 4], [2, 1], [1, 1], [2, 3]]),
        refp = DesignMeasure([0.3, 0.2, 0.1, 0.4], [[1, 1], [2, 1], [2, 3], [3, 4]]),
        refw = DesignMeasure([0.1, 0.2, 0.3, 0.4], [[2, 3], [2, 1], [1, 1], [3, 4]])

        @test !(sort_designpoints(d) === d)
        @test sort_designpoints(d).weight == refp.weight
        @test sort_designpoints(d).designpoint == refp.designpoint
        @test sort_designpoints(d; rev = true).weight == reverse(refp.weight)
        @test sort_designpoints(d; rev = true).designpoint == reverse(refp.designpoint)

        @test !(sort_weights(d) === d)
        @test sort_weights(d).weight == refw.weight
        @test sort_weights(d).designpoint == refw.designpoint
        @test sort_weights(d; rev = true).weight == reverse(refw.weight)
        @test sort_weights(d; rev = true).designpoint == reverse(refw.designpoint)
    end

    # abstract point methods: randomization with fixed weights and/or points
    let ds = DesignSpace(:a => (0, 1)),
        d = DesignMeasure([0.1, 0.42, 0.48], [[1], [4.2], [3]]),
        fw1 = [false, true, false],
        fp1 = fill(false, 3),
        r1 = Kirstine.randomize!(deepcopy(d), (ds, fw1, fp1)),
        fw2 = fill(false, 3),
        fp2 = [false, true, false],
        r2 = Kirstine.randomize!(deepcopy(d), (ds, fw2, fp2)),
        fw3 = [false, false, true],
        fp3 = [false, false, true],
        r3 = Kirstine.randomize!(deepcopy(d), (ds, fw3, fp3))

        @test r1.weight[2] == d.weight[2]
        @test all(r1.weight[[1, 3]] .!= d.weight[[1, 3]])
        @test r2.designpoint[2] == d.designpoint[2]
        @test all(r2.designpoint[[1, 3]] .!= d.designpoint[[1, 3]])
        @test r3.weight[3] == d.weight[3]
        @test r3.designpoint[3] == d.designpoint[3]
        @test all(r3.weight[1:2] .!= d.weight[1:2])
        @test all(r3.designpoint[1:2] .!= d.designpoint[1:2])
    end
end
