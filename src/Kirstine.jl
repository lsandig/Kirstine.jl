module Kirstine

using LinearAlgebra: diagind, norm, BLAS.syrk!, LAPACK.potrf!, LAPACK.potri!
using Random: rand, rand!
using RecipesBase

export Covariate,
    CovariateParameterization,
    DOptimality,
    DesignMeasure,
    DesignSpace,
    DiscretePrior,
    Identity,
    NonlinearRegression,
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
include("plot.jl")

end
