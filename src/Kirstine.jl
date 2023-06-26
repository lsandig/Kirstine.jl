module Kirstine

using LinearAlgebra:
    BLAS.symm!,
    BLAS.syrk!,
    LAPACK.posv!,
    LAPACK.potrf!,
    LAPACK.potri!,
    PosDefException,
    SingularException,
    Symmetric,
    diagind,
    mul!,
    norm
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
    FisherMatrix,
    Identity,
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
