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
    DesignInterval,
    DesignMeasure,
    DiscretePrior,
    FisherMatrix,
    Identity,
    NonlinearRegression,
    OptimizationResult,
    Pso,
    apportion,
    as_matrix,
    designpoints,
    dimnames,
    efficiency,
    equidistant_design,
    gateauxderivative,
    informationmatrix,
    lowerbound,
    mixture,
    objective,
    one_point_design,
    optimize_design,
    plot_gateauxderivative,
    random_design,
    refine_design,
    simplify,
    simplify_drop,
    simplify_merge,
    simplify_unique,
    sort_designpoints,
    sort_weights,
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
