module Kirstine

using LinearAlgebra: diagind, norm, BLAS.syrk!, LAPACK.potrf!, LAPACK.potri!
using Random: rand

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
    apportion,
    gateauxderivative,
    grid_design,
    mixture,
    objective,
    random_design,
    simplify,
    simplify_drop,
    simplify_merge,
    simplify_unique,
    singleton_design,
    support,
    uniform_design,
    weights

include("types.jl")
include("designmeasure.jl")
include("design-common.jl")
include("design-doptimal.jl")

end
