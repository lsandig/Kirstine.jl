module Kirstine

using LinearAlgebra: diagind, norm, BLAS.syrk!, LAPACK.potrf!, LAPACK.potri!
using LinearAlgebra: Symmetric, diagm, tr, LAPACK.posv!, mul!, BLAS.symm!
using Random: rand, rand!
using RecipesBase

export @define_scalar_unit_model,
    Covariate,
    CovariateParameterization,
    DOptimality,
    DeltaMethod,
    DesignMeasure,
    DesignSpace,
    DiscretePrior,
    Identity,
    MAPApproximation,
    MLApproximation,
    NonlinearRegression,
    OptimizationResult,
    PriorGuess,
    PriorSample,
    Pso,
    apportion,
    as_matrix,
    designpoints,
    dimnames,
    efficiency,
    gateauxderivative,
    grid_design,
    lowerbound,
    mixture,
    objective,
    optimize_design,
    plot_gateauxderivative,
    random_design,
    refine_design,
    simplify,
    simplify_drop,
    simplify_merge,
    simplify_unique,
    singleton_design,
    sort_designpoints,
    sort_weights,
    support,
    uniform_design,
    upperbound,
    weights

include("types.jl")
include("designmeasure.jl")
include("optimize.jl")
include("pso.jl")
include("design-common.jl")
include("design-doptimal.jl")
include("util.jl")
include("plot.jl")

end
