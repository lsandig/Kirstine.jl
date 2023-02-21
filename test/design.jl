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
    function Kirstine.allocate_covariates(m::EmaxModel, K::Integer)
        cs = [Dose(0) for _ in 1:K]
        return cs
    end
    function Kirstine.update_model_covariate!(c::Dose,
                                              cp::CopyDose,
                                              dp::AbstractVector{<:Real},
                                              m::EmaxModel)
        c.dose = dp[1]
        return c
    end
    # p must be a NamedTuple with elements `e0`, `emax` `ec50`
    function Kirstine.update_jacobian_matrix!(jm, c::Dose, m::EmaxModel, p)
        x = c.dose
        jm[1, 1] = 1.0
        jm[1, 2] = x / (x + p.ec50)
        jm[1, 3] = - p.emax * x / (x + p.ec50)^2
        return jm
    end

    # error handling in constructors
    @test_throws "must be equal" DiscretePrior([0], [[1], [2]])
    @test_throws "non-negative" DiscretePrior([-0.5, 1.5], [[1], [2]])

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
        a = ds.lowerbound[1],
        b = ds.upperbound[1],
        # Locally D-optimal design for the Emax model has an analytic solution,
        # see Theorem 2 in
        #
        #   Dette, H., Kiss, C., Bevanda, M., & Bretz, F. (2010).
        #   Optimal designs for the emax, log-linear and exponential models.
        #   Biometrika, 97(2), 513–518. http://dx.doi.org/10.1093/biomet/asq020
        #
        x_star = (a * (b + p1.ec50) + b * (a + p1.ec50)) / (a + b + 2 * p1.ec50),
        # sol is optimal for pk1
        sol = uniform_design([[a], [x_star], [b]]),
        # not_sol is not optimal for any of pk1, pk2, pk3
        not_sol = DesignMeasure([0.2, 0.3, 0.5], [[a + 0.1 * (b-a)], [x_star * 1.1], [a + (0.9 * (b-a))]]),
        to_dirac(d) = map(singleton_design, support(d)),
        gd(g, s, pk) = gateauxderivative(dc, to_dirac(g), s, m, cp, pk, trafo),
        ob(d, pk) = objective(dc, d, m, cp, pk, trafo);
        @test all(abs.(gd(sol, sol, pk1)) .<= sqrt(eps()))
        @test all(abs.(gd(not_sol, sol, pk1)) .> 0.01)
        @test all(abs.(gd(not_sol, not_sol, pk2)) .> 0.1)
        @test all(abs.(gd(not_sol, not_sol, pk3)) .> 0.1)
        @test ob(not_sol, pk1) < ob(sol, pk1)
        # a design with less than 3 support points is singular
        @test isinf(ob(uniform_design([[a], [b]]), pk1))
    end
end
