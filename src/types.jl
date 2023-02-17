abstract type Model end
abstract type Covariate end
abstract type CovariateParameterization end
abstract type PriorKnowledge end
abstract type Parameter end
abstract type Transformation end
abstract type DesignCriterion end

abstract type AbstractPoint end
abstract type Optimizer end
abstract type OptimizerState end

"""
    DesignMeasure

A discrete probability measure with finite support
representing a continuous experimental design.

See p.62 in Fedorov, V. V., & Leonov, S. L. (2013). [Optimal design for
nonlinear response models](https://doi.org/10.1201/b15054).
"""
struct DesignMeasure <: AbstractPoint
    "the weights of the individual design points"
    weight::Vector{Float64}
    "the design points"
    designpoint::Vector{Vector{Float64}}
    @doc """
        DesignMeasure(w, dp)

    Construct a design measure with the given weights and a vector of design
    points.

    ## Examples
    ```@example
    DesignMeasure([0.5, 0.2, 0.3], [[1, 2], [3, 4], [5, 6]])
    ```
    """
    function DesignMeasure(weight, designpoint)
        if length(weight) != length(designpoint)
            error("number of weights and design points must be equal")
        end
        if !allequal(map(length, designpoint))
            error("design points must have identical lengths")
        end
        if any(weight .< 0) || !(sum(weight) ≈ 1)
            error("weights must be non-negative and sum to one")
        end
        new(weight, designpoint)
    end
end

"""
    DesignSpace{N}

A (hyper)rectangular subset of ``\\mathbb{R}^N`` representing the set in which
which the design points of a `DesignMeasure` live.
The dimensions of a `DesignSpace` are named.
"""
struct DesignSpace{N}
    name::NTuple{N,Symbol}
    lowerbound::NTuple{N,Float64}
    upperbound::NTuple{N,Float64}
    @doc """
        DesignSpace(name, lowerbound, upperbound)

    Construct a design space with the given dimension names (supplied as
    symbols) and lower / upper bounds.

    ## Examples
    ```@example
     DesignSpace([:dose, :time], [0, 0], [300, 20])
    ```
    """
    function DesignSpace(name, lowerbound, upperbound)
        n = length(name)
        if !(n == length(lowerbound) == length(upperbound))
            error("lengths of name, upper and lower bounds must be identical")
        end
        if any(upperbound .<= lowerbound)
            error("upper bounds must be strictly larger than lower bounds")
        end
        new{n}(
            tuple(name...),
            tuple(convert.(Float64, lowerbound)...),
            tuple(convert.(Float64, upperbound)...),
        )
    end
end
