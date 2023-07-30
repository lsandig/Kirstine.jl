# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

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
    norm,
    tr
using Random: rand, rand!
using RecipesBase

export @define_scalar_unit_model,
    @define_vector_parameter,
    Covariate,
    CovariateParameterization,
    AOptimality,
    DOptimality,
    DeltaMethod,
    DesignCriterion,
    DesignInterval,
    DesignMeasure,
    DesignProblem,
    DesignRegion,
    DirectMaximization,
    DirectMaximizationResult,
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
    PriorSample,
    ProblemSolvingResult,
    ProblemSolvingStrategy,
    Pso,
    Transformation,
    apportion,
    as_matrix,
    designpoints,
    dimension,
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
include("design-aoptimal.jl")
include("util.jl")
include("plot.jl")

end
