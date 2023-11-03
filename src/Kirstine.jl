# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module Kirstine

using LinearAlgebra:
    BLAS.symm!,
    BLAS.syrk!,
    BLAS.trsm!,
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

export @simple_model,
    @simple_parameter,
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
    covariate_parameterization,
    criterion,
    dimension,
    dimnames,
    efficiency,
    equidistant_design,
    gateauxderivative,
    informationmatrix,
    lowerbound,
    maximizer,
    mixture,
    model,
    normal_approximation,
    numpoints,
    objective,
    one_point_design,
    plot_expected_function,
    plot_gateauxderivative,
    points,
    prior_knowledge,
    random_design,
    region,
    shannon_information,
    simplify,
    simplify_drop,
    simplify_merge,
    simplify_unique,
    solve,
    sort_points,
    sort_weights,
    transformation,
    uniform_design,
    upperbound,
    weights

include("abstracttypes.jl")
include("optimize.jl")
include("pso.jl")
include("designregion.jl")
include("designmeasure.jl")
include("designmeasure-abstractpoint.jl")
include("priorknowledge.jl")
include("transformation.jl")
include("normalapproximation.jl")
include("user.jl")
include("design-common.jl")
include("criterion-d.jl")
include("criterion-a.jl")
include("designproblem.jl")
include("solve-directmaximization.jl")
include("solve-exchange.jl")
include("plot.jl")

end
