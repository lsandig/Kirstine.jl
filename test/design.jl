@testset "design-common" begin
    # error handling in constructors
    @test_throws "must be equal" DiscretePrior([0], [[1], [2]])
    @test_throws "non-negative" DiscretePrior([-0.5, 1.5], [[1], [2]])

    # helpers
    let A = reshape(collect(1:9), 3, 3),
        B = reshape(collect(11:19), 3, 3),
        C = reshape(collect(1:12), 3, 4)

        @test tr(Symmetric(A) * Symmetric(B)) == Kirstine.tr_prod(A, B, :U)
        @test tr(Symmetric(A, :L) * Symmetric(B, :L)) == Kirstine.tr_prod(A, B, :L)
        @test_throws "identical size" Kirstine.tr_prod(A, C, :U)
        @test_throws "either :U or :L" Kirstine.tr_prod(A, B, :F)
    end

    let A = rand(Float64, 3, 3), B = A * A'

        # note: log_det! overwrites B, so it can't be called first
        @test log(det(B)) ≈ Kirstine.log_det!(B)
    end

    # Applying a DeltaMethod transformation should give the same results whether the
    # incoming normalized information matrix is already inverted or not. For a DeltaMethod
    # that is actually an identity transformation in disguise, the result should simply be
    # the inverse of the argument.
    #
    # Note: we have to recreate the circumstances in which apply_transformation! is called:
    # nim is allowed to be only upper triangular, and is allowed to be overwritten. Hence we
    # must use deepcopys, and Symmetric wrappers where necessary.
    let pk = PriorGuess((dummy = 42,)),
        tid = DeltaMethod(p -> diagm(ones(3)), pk),
        tsc = DeltaMethod(p -> diagm([0.5, 2.0, 4.0]), pk),
        _ = seed!(4321),
        A = reshape(rand(9), 3, 3),
        nim = collect(UpperTriangular(A' * A)),
        inv_nim = collect(UpperTriangular(inv(A' * A))),
        nim1 = deepcopy(nim),
        nim2 = deepcopy(inv_nim),
        nim3 = deepcopy(nim),
        nim4 = deepcopy(inv_nim),
        (tnim1, _) = Kirstine.apply_transformation!(zeros(3, 3), nim1, false, tid, 1),
        (tnim2, _) = Kirstine.apply_transformation!(zeros(3, 3), nim2, true, tid, 1),
        # scaling parameters should be able to be pulled out
        (tnim3, _) = Kirstine.apply_transformation!(zeros(3, 3), nim3, false, tsc, 1),
        (tnim4, _) = Kirstine.apply_transformation!(zeros(3, 3), nim4, true, tsc, 1)

        @test Symmetric(tnim1) ≈ Symmetric(inv_nim)
        @test Symmetric(tnim2) ≈ Symmetric(inv_nim)
        @test Symmetric(tnim3) ≈ Symmetric(tnim4)
        @test det(Symmetric(tnim3)) ≈ 4^2 * det(Symmetric(inv_nim))
    end
end

@testset "doptimal" begin
    struct EmaxModel <: NonlinearRegression
        inv_sigma_sq::Float64
    end
    mutable struct Dose <: Covariate
        dose::Float64
    end
    struct CopyDose <: CovariateParameterization end
    Kirstine.unit_length(m::EmaxModel) = 1
    Kirstine.invcov(m::EmaxModel) = m.inv_sigma_sq
    Kirstine.allocate_covariate(m::EmaxModel) = Dose(0)
    function Kirstine.update_model_covariate!(
        c::Dose,
        dp::AbstractVector{<:Real},
        m::EmaxModel,
        cp::CopyDose,
    )
        c.dose = dp[1]
        return c
    end
    # p must be a NamedTuple with elements `e0`, `emax` `ec50`
    function Kirstine.jacobianmatrix!(jm, m::EmaxModel, c::Dose, p)
        x = c.dose
        jm[1, 1] = 1.0
        jm[1, 2] = x / (x + p.ec50)
        jm[1, 3] = -p.emax * x / (x + p.ec50)^2
        return jm
    end

    function emax_solution(p, ds)
        # Locally D-optimal design for the Emax model has an analytic solution,
        # see Theorem 2 in
        #
        #   Dette, H., Kiss, C., Bevanda, M., & Bretz, F. (2010).
        #   Optimal designs for the emax, log-linear and exponential models.
        #   Biometrika, 97(2), 513–518. http://dx.doi.org/10.1093/biomet/asq020
        #
        a = ds.lowerbound[1]
        b = ds.upperbound[1]
        x_star = (a * (b + p.ec50) + b * (a + p.ec50)) / (a + b + 2 * p.ec50)
        return uniform_design([[a], [x_star], [b]])
    end

    # Three-parameter compartmental model from
    # Atkinson, A. C., Chaloner, K., Herzberg, A. M., & Juritz, J. (1993).
    # Optimum experimental designs for properties of a compartmental model.
    # Biometrics, 49(2), 325–337. http://dx.doi.org/10.2307/2532547
    @define_scalar_unit_model Kirstine TPCMod time
    struct CopyTime <: CovariateParameterization end
    function Kirstine.jacobianmatrix!(jm, m::TPCMod, c::TPCModCovariate, p)
        # names(p) == a, e, s
        A = exp(-p.a * c.time)
        E = exp(-p.e * c.time)
        jm[1, 1] = A * p.s * c.time
        jm[1, 2] = -E * p.s * c.time
        jm[1, 3] = E - A
        return m
    end
    function Kirstine.update_model_covariate!(
        c::TPCModCovariate,
        dp::AbstractVector{<:Real},
        m::TPCMod,
        cp::CopyTime,
    )
        c.time = dp[1]
        return c
    end
    # response, area under the curve, time-to-maximum, maximum concentration
    mu(c, p) = p.s * (exp(-p.e * c.time) - exp(-p.a * c.time))
    auc(p) = p.s * (1 / p.e - 1 / p.a)
    ttm(p) = log(p.a / p.e) / (p.a - p.e)
    cmax(p) = mu(TPCModCovariate(ttm(p)), p)
    # jacobian matrices (transposed gradients) from Appendix 1
    function Dauc(p)
        da = p.s / p.a^2
        de = -p.s / p.e^2
        ds = 1 / p.e - 1 / p.a
        return [da de ds]
    end
    function Dttm(p)
        A = p.a - p.e
        A2 = A^2
        B = log(p.a / p.e)
        da = (A / p.a - B) / A2
        de = (B - A / p.e) / A2
        ds = 0
        return [da de ds]
    end
    function Dcmax(p)
        tmax = ttm(p)
        A = exp(-p.a * tmax)
        E = exp(-p.e * tmax)
        F = p.a * A - p.e * E
        da_ttm, de_ttm, ds_ttm = Dttm(p)
        da = p.s * (tmax * A - F * da_ttm)
        de = p.s * (-tmax * E + F * de_ttm)
        ds = E - A
        return [da de ds]
    end

    # Gateaux derivatives and efficiency
    let dc = DOptimality(),
        na_ml = MLApproximation(),
        na_map = MAPApproximation(zeros(3, 3)),
        trafo = Identity(),
        m = EmaxModel(1),
        cp = CopyDose(),
        p1 = (e0 = 1, emax = 10, ec50 = 5),
        p2 = (e0 = 5, emax = -3, ec50 = 2),
        pk1 = PriorGuess(p1),
        pk2 = DiscretePrior([0.75, 0.25], [p1, p2]),
        pk3 = PriorSample([p1, p2]),
        ds = DesignSpace(:dose => (0, 10)),
        # sol is optimal for pk1
        sol = emax_solution(p1, ds),
        a = ds.lowerbound[1],
        b = ds.upperbound[1],
        x_star = sol.designpoint[2][1],
        # not_sol is not optimal for any of pk1, pk2, pk3
        not_sol = DesignMeasure(
            [0.2, 0.3, 0.5],
            [[a + 0.1 * (b - a)], [x_star * 1.1], [a + (0.9 * (b - a))]],
        ),
        to_dirac(d) = map(singleton_design, support(d)),
        gd(s, d, pk, na) = gateauxderivative(dc, s, to_dirac(d), m, cp, pk, trafo, na),
        ob(d, pk, na) = objective(dc, d, m, cp, pk, trafo, na)

        @test all(abs.(gd(sol, sol, pk1, na_ml)) .<= sqrt(eps()))
        @test all(abs.(gd(sol, not_sol, pk1, na_ml)) .> 0.01)
        @test all(abs.(gd(not_sol, not_sol, pk2, na_ml)) .> 0.1)
        @test all(abs.(gd(not_sol, not_sol, pk3, na_ml)) .> 0.1)
        @test ob(not_sol, pk1, na_ml) < ob(sol, pk1, na_ml)
        # a design with less than 3 support points is singular
        @test isinf(ob(uniform_design([[a], [b]]), pk1, na_ml))
        # sol is better than not_sol for pk1 (by construction)
        @test efficiency(sol, not_sol, m, cp, pk1, trafo, na_ml) > 1
        # it also happens to be better for pk2 and pk3
        @test efficiency(sol, not_sol, m, cp, pk2, trafo, na_ml) > 1
        @test efficiency(sol, not_sol, m, cp, pk3, trafo, na_ml) > 1
        # check that efficiency wrt prior sample divides by length of sample vector
        @test efficiency(sol, not_sol, m, cp, PriorSample([p1, p1]), trafo, na_ml) ==
              efficiency(sol, not_sol, m, cp, PriorSample([p1]), trafo, na_ml)
        # check that MAPApproximation is correct in a special case
        @test ob(sol, pk1, na_ml) ≈ ob(sol, pk1, na_map)
        @test ob(sol, pk2, na_ml) ≈ ob(sol, pk2, na_map)
        @test ob(sol, pk3, na_ml) ≈ ob(sol, pk3, na_map)
        @test gd(sol, not_sol, pk1, na_ml) ≈ gd(sol, not_sol, pk1, na_map)
        @test gd(sol, not_sol, pk2, na_ml) ≈ gd(sol, not_sol, pk2, na_map)
        @test gd(sol, not_sol, pk3, na_ml) ≈ gd(sol, not_sol, pk3, na_map)
    end

    # DeltaMethod for Atkinson et al. examples
    let ds = DesignSpace(:time => [0, 48]),
        g0 = PriorGuess((a = 4.298, e = 0.05884, s = 21.80)),
        m = TPCMod(1),
        cp = CopyTime(),
        dc = DOptimality(),
        na_ml = MLApproximation(),
        na_map_nonreg = MAPApproximation(zeros(3, 3)),
        na_map = MAPApproximation(diagm(fill(1e-5, 3))),
        # first four designs of Table 1, and corresponding transformations
        #! format: off
        a1  = DesignMeasure([0.2288] => 1/3,    [1.3886] => 1/3,    [18.417] => 1/3),
        a2  = DesignMeasure([0.2327] => 0.0135, [17.633] => 0.9865),
        a3  = DesignMeasure([0.1793] => 0.6062, [3.5671] => 0.3938),
        a4  = DesignMeasure([1.0122] => 1.0),
        a5 = DesignMeasure([0.2176] => 0.2337, [1.4343] => 0.3878, [18.297] => 0.3785),
        a0_time = [1/6, 1/3, 1/2, 2/3, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 24, 30, 48],
        a0 = uniform_design([[t] for t in a0_time]),
        #! format: on
        t1 = Identity(),
        t1_delta = DeltaMethod(p -> diagm(ones(3)), g0),
        t2 = DeltaMethod(Dauc, g0),
        t3 = DeltaMethod(Dttm, g0),
        t4 = DeltaMethod(Dcmax, g0),
        dir = [singleton_design([t]) for t in range(0, 48; length = 21)],
        #! format: off
        teff = [100     34.31  66.02  36.10;
                  0    100      0      0   ;
                  0      0    100      0   ;
                  0      0      0    100   ;
                 97.36  38.97  52.89  41.26;
                 67.61  24.00  28.60  36.77],
        #! format: on
        ob(a, t, na) = objective(dc, a, m, cp, g0, t, na),
        gd(a, t, na) = gateauxderivative(dc, a, dir, m, cp, g0, t, na),
        ef(a, zs, ts) = map((z, t) -> 100 * efficiency(a, z, m, cp, g0, t, na_map), zs, ts)

        # check Atkinson's solutions
        @test ob(a1, t1, na_map) ≈ 7.3887 rtol = 1e-4
        @test exp(-ob(a2, t2, na_map)) ≈ 2194 rtol = 1e-4
        @test exp(-ob(a3, t3, na_map)) ≈ 0.02815 rtol = 1e-3
        @test exp(-ob(a4, t4, na_map)) ≈ 1.000 rtol = 1e-3
        @test maximum(gd(a1, t1, na_map)) <= 0
        # Gateaux derivative is sensitive to more than the four published decimal places
        @test_broken maximum(gd(a2, t2, na_map)) <= 0
        @test_broken maximum(gd(a3, t3, na_map)) <= 0
        @test_broken maximum(gd(a4, t4, na_map)) <= 0
        # compare DeltaMethod identity with actual Identity
        @test ob(a1, t1, na_map) ≈ ob(a1, t1_delta, na_map)
        @test gd(a1, t1, na_map) ≈ gd(a1, t1_delta, na_map)
        @test ob(a1, t1, na_ml) ≈ ob(a1, t1_delta, na_ml)
        @test gd(a1, t1, na_ml) ≈ gd(a1, t1_delta, na_ml)
        # full-rank information matrices should work with a non-regularizing approximation
        @test ob(a1, t1, na_map_nonreg) ≈ ob(a1, t1_delta, na_map_nonreg)
        @test gd(a1, t1, na_map_nonreg) ≈ gd(a1, t1_delta, na_map_nonreg)
        # Table 4 from the article ([2, 1] and [3, 1] are off)
        @test teff[1, :] ≈ ef(a1, [a1, a2, a3, a4], [t1, t2, t3, t4]) rtol = 1e-3
        @test_broken teff[2, 1] ≈ ef(a2, [a1], [t1])[1] rtol = 1e-3
        @test teff[2, 2:4] ≈ ef(a2, [a2, a3, a4], [t2, t3, t4]) rtol = 1e-3
        @test_broken teff[3, 1] ≈ ef(a3, [a1], [t1])[1] rtol = 1e-3
        @test teff[3, 2:4] ≈ ef(a3, [a2, a3, a4], [t2, t3, t4]) rtol = 1e-3
        @test teff[4, :] ≈ ef(a4, [a1, a2, a3, a4], [t1, t2, t3, t4]) rtol = 1e-3
        @test teff[5, :] ≈ ef(a5, [a1, a2, a3, a4], [t1, t2, t3, t4]) rtol = 1e-3
        @test teff[6, :] ≈ ef(a0, [a1, a2, a3, a4], [t1, t2, t3, t4]) rtol = 1e-3
    end

    # Can we find the locally D-optimal design?
    let dc = DOptimality(),
        na = MLApproximation(),
        trafo = Identity(),
        m = EmaxModel(1),
        cp = CopyDose(),
        p = (e0 = 1, emax = 10, ec50 = 5),
        pk = PriorGuess(p),
        ds = DesignSpace(:dose => (0, 10)),
        sol = emax_solution(p, ds),
        pso = Pso(; iterations = 50, swarmsize = 20),
        optim(; kwargs...) = optimize_design(pso, dc, ds, m, cp, pk, trafo, na; kwargs...),
        # search from a random starting design
        _ = seed!(4711),
        (d1, o1) = optim(),
        # search with lower and upper bound already known and fixed, uniform weights
        _ = seed!(4711),
        cand = grid_design(ds, 3),
        (d2, o2) = optim(; candidate = cand, fixedweights = 1:3, fixedpoints = [1, 3])

        @test d1.weight ≈ sol.weight rtol = 1e-3
        for k in 1:3
            @test d1.designpoint[k] ≈ sol.designpoint[k] atol = 1e-2
        end

        @test d2.weight ≈ sol.weight rtol = 1e-3
        for k in 1:3
            @test d2.designpoint[k] ≈ sol.designpoint[k] atol = 1e-3
        end
        # fixed weights should not change, using ≈ because numerically 1-2/3 != 1/3
        @test all(map(d -> all(d.weight .≈ 1 / 3), o2.trace_x))
        # fixed points should not move
        @test all(map(d -> d.designpoint[1][1], o2.trace_x) .== ds.lowerbound[1])
        @test all(map(d -> d.designpoint[3][1], o2.trace_x) .== ds.upperbound[1])
        # the middle point should only get closer to the optimal one
        # (note that we don't need to sort the atoms of d)
        @test issorted(
            map(d -> abs(d.designpoint[2][1] - sol.designpoint[2][1]), o2.trace_x),
            rev = true,
        )

        @test_throws "between 1 and 3" optim(fixedweights = [0])
        @test_throws "between 1 and 3" optim(fixedweights = [4])
        @test_throws "between 1 and 3" optim(fixedpoints = [-1])
        @test_throws "between 1 and 3" optim(fixedpoints = [5])
        @test_throws "outside design space" optim(candidate = uniform_design([[0], [20]]))
        @test_throws "must match" optim(candidate = uniform_design([[0, 1], [1, 0]]))
        # `minposdist` doesn't exist, the correct argument name is `mindist`. Because we
        # have not implemented `simplify_unique()` for EmaxModel, the generic method should
        # complain about gobbling up `minposdist` in its varargs. (We can't test this in
        # designmeasure.jl because we need a Model and a CovariateParameterization in order
        # to call `simplify()`.)
        @test_logs(
            (:warn, "unused keyword arguments given to generic `simplify_unique` method"),
            optim(minposdist = 1e-2)
        )
    end

    # fixed weights and / or points should never change
    let dc = DOptimality(),
        na = MLApproximation(),
        trafo = Identity(),
        m = EmaxModel(1),
        cp = CopyDose(),
        p = (e0 = 1, emax = 10, ec50 = 5),
        pk = PriorGuess(p),
        ds = DesignSpace(:dose => (0, 10)),
        pso = Pso(; iterations = 2, swarmsize = 5),
        # this is not the optimal solution
        candidate = DesignMeasure([0.1, 0.5, 0.0, 0.0, 0.4], [[0], [5], [7], [8], [10]]),
        #! format: off
        opt(; fw = Int64[], fp = Int64[]) = optimize_design(
            pso, dc, ds, m, cp, pk, trafo, na;
            candidate = candidate, fixedweights = fw, fixedpoints = fp, trace_state = true,
        ),
        #! format: on
        is_const_w(o, ref, k) =
            all([all(map(d -> d.weight[k] == ref.weight[k], s.x)) for s in o.trace_state]),
        is_const_d(o, ref, k) = all([
            all(map(d -> d.designpoint[k] == ref.designpoint[k], s.x)) for
            s in o.trace_state
        ]),
        _ = seed!(4711),
        (_, o1) = opt(; fw = [2], fp = [2]),
        (_, o2) = opt(; fw = [2]),
        (_, o3) = opt(; fp = [2]),
        (_, o4) = opt(; fw = [5], fp = [5]),
        (_, o5) = opt(; fw = [1, 5], fp = [5])

        @test is_const_w(o1, candidate, 2)
        @test is_const_d(o1, candidate, 2)
        @test is_const_w(o2, candidate, 2)
        @test is_const_d(o3, candidate, 2)
        @test is_const_w(o4, candidate, 5)
        @test is_const_d(o4, candidate, 5)
        @test is_const_w(o5, candidate, 1)
        @test is_const_w(o5, candidate, 5)
        @test is_const_d(o5, candidate, 5)

        @test_logs(
            (:warn, "fixed weights already sum to one"),
            match_mode = :any,
            opt(fw = [1, 2, 5])
        )
    end

    # Does refinement work?
    let dc = DOptimality(),
        na = MLApproximation(),
        trafo = Identity(),
        m = EmaxModel(1),
        cp = CopyDose(),
        p = (e0 = 1, emax = 10, ec50 = 5),
        pk = PriorGuess(p),
        ds = DesignSpace(:dose => (0, 10)),
        sol = emax_solution(p, ds),
        od = Pso(; iterations = 50, swarmsize = 100),
        ow = Pso(; iterations = 50, swarmsize = 50),
        _ = seed!(1234),
        cand = uniform_design([[[5]]; support(sol)[[1, 3]]]),
        (r, rd, rw) = refine_design(od, ow, 3, cand, dc, ds, m, cp, pk, trafo, na)

        @test abs(support(r)[2][1] - support(sol)[2][1]) <
              abs(support(cand)[1][1] - support(sol)[2][1])
        @test issorted([r.maximum for r in rw])
        @test all([r.maximum > 0 for r in rd])
    end
end
