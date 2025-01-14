# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module Kirstine

using LinearAlgebra:
    BLAS.gemm!,
    BLAS.symm!,
    BLAS.syrk!,
    BLAS.trsm!,
    LAPACK.posv!,
    LAPACK.potrf!,
    LAPACK.potri!,
    PosDefException,
    SingularException,
    Symmetric,
    cond,
    diagind,
    diagm,
    dot,
    mul!,
    norm,
    rank,
    tr
using Random: rand, rand!
using RecipesBase: RecipesBase, @layout, @recipe, @series
using Statistics: quantile

import Base: maximum

export @simple_model,
    @simple_parameter,
    ACriterion,
    CompositeDesignProblem,
    DCriterion,
    DeltaMethod,
    DesignInterval,
    DesignMeasure,
    DesignProblem,
    DirectMaximization,
    Exchange,
    FisherMatrix,
    Identity,
    JustCopy,
    PriorSample,
    Pso,
    apportion,
    covariate_parameterization,
    criterion,
    dimension,
    dimnames,
    efficiency,
    equidistant_design,
    gateauxderivative,
    implied_covariates,
    lowerbound,
    mixture,
    model,
    normal_approximation,
    numpoints,
    objective,
    one_point_design,
    optimization_result,
    optimization_results_direction,
    optimization_results_weight,
    parameters,
    plot_expected_function,
    plot_gateauxderivative,
    points,
    prior_knowledge,
    problems,
    random_design,
    region,
    shannon_information,
    simplify,
    simplify_drop,
    simplify_merge,
    simplify_unique,
    solution,
    solve,
    sort_points,
    sort_weights,
    transformation,
    uniform_design,
    upperbound,
    weights

include("abstracttypes.jl")
include("optimize.jl")
include("optimizer-pso.jl")
include("optimizer-grid.jl")
include("optimizer-frank-wolfe.jl")
include("designregion.jl")
include("designmeasure.jl")
include("designmeasure-abstractpoint.jl")
include("priorknowledge.jl")
include("transformation.jl")
include("normalapproximation.jl")
include("user.jl")
include("design-common.jl")
include("model-nonlinear-regression.jl")
include("criterion-d.jl")
include("criterion-a.jl")
include("designproblem.jl")
include("composite-problem.jl")
include("solve-directmaximization.jl")
include("solve-exchange.jl")
include("plot.jl")

end
