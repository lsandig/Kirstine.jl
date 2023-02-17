module Kirstine

using LinearAlgebra: diagind, norm
using Random: rand

export DesignMeasure,
    DesignSpace,
    apportion,
    grid_design,
    mixture,
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

end
