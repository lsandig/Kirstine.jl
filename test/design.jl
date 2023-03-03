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

    # Gateaux derivatives
    let dc = DOptimality(),
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
        gd(s, d, pk) = gateauxderivative(dc, s, to_dirac(d), m, cp, pk, trafo),
        ob(d, pk) = objective(dc, d, m, cp, pk, trafo)

        @test all(abs.(gd(sol, sol, pk1)) .<= sqrt(eps()))
        @test all(abs.(gd(sol, not_sol, pk1)) .> 0.01)
        @test all(abs.(gd(not_sol, not_sol, pk2)) .> 0.1)
        @test all(abs.(gd(not_sol, not_sol, pk3)) .> 0.1)
        @test ob(not_sol, pk1) < ob(sol, pk1)
        # a design with less than 3 support points is singular
        @test isinf(ob(uniform_design([[a], [b]]), pk1))
        # sol is better than not_sol for pk1 (by construction)
        @test efficiency(sol, not_sol, m, cp, pk1, trafo) > 1
        # it also happens to be better for pk2 and pk3
        @test efficiency(sol, not_sol, m, cp, pk2, trafo) > 1
        @test efficiency(sol, not_sol, m, cp, pk3, trafo) > 1
    end

    # Can we find the locally D-optimal design?
    let dc = DOptimality(),
        trafo = Identity(),
        m = EmaxModel(1),
        cp = CopyDose(),
        p = (e0 = 1, emax = 10, ec50 = 5),
        pk = PriorGuess(p),
        ds = DesignSpace(:dose => (0, 10)),
        sol = emax_solution(p, ds),
        pso = Pso(; iterations = 50, swarmsize = 20),
        optim(; kwargs...) = optimize_design(pso, dc, ds, m, cp, pk, trafo; kwargs...),
        # search from a random starting design
        _ = seed!(4711),
        o1 = optim(),
        d1 = sort_designpoints(o1.maximizer),
        # search with lower and upper bound already known and fixed, uniform weights
        _ = seed!(4711),
        cand = grid_design(ds, 3),
        o2 = optim(; candidate = cand, fixedweights = 1:3, fixedpoints = [1, 3]),
        d2 = sort_designpoints(o2.maximizer)

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
    end

    # fixed weights and / or points should never change
    let dc = DOptimality(),
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
            pso, dc, ds, m, cp, pk, trafo;
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
        o1 = opt(; fw = [2], fp = [2]),
        o2 = opt(; fw = [2]),
        o3 = opt(; fp = [2]),
        o4 = opt(; fw = [5], fp = [5]),
        o5 = opt(; fw = [1, 5], fp = [5])

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
        r = refine_design(od, ow, 3, cand, dc, ds, m, cp, pk, trafo)

        @test abs(support(r)[1][1] - support(sol)[2][1]) <
              abs(support(cand)[1][1] - support(sol)[2][1])
    end
end
