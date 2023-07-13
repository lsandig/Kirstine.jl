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
    @define_vector_parameter,
    Covariate,
    CovariateParameterization,
    DOptimality,
    DeltaMethod,
    DesignCriterion,
    DesignInterval,
    DesignMeasure,
    DesignProblem,
    DesignSpace,
    DirectMaximization,
    DirectMaximizationResult,
    DiscretePrior,
    Exchange,
    ExchangeResult,
    FisherMatrix,
    Identity,
    Model,
    NonlinearRegression,
    NormalApproximation,
    OptimizationResult,
    Optimizer,
    Parameter,
    PriorKnowledge,
    ProblemSolvingResult,
    ProblemSolvingStrategy,
    Pso,
    Transformation,
    apportion,
    as_matrix,
    designpoints,
    dimnames,
    efficiency,
    equidistant_design,
    gateauxderivative,
    informationmatrix,
    lowerbound,
    maximizer,
    mixture,
    objective,
    one_point_design,
    plot_gateauxderivative,
    random_design,
    simplify,
    simplify_drop,
    simplify_merge,
    simplify_unique,
    solve,
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
