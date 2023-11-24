# Extending Kirstine

The code of Kirstine.jl is written in a modular fashion.
This means that you can write drop-in replacements for many of the types that constitute a [`DesignProblem`](@ref)
in order to implement any functionality that you might be missing.
You can even implement your own high- or low-level optimization algorithms.

The following pages show how to implement custom versions of the following objects:

  - [design criterion](extend-criterion.md)
  - [design region](extend-region.md)
  - [model supertype](extend-model.md)
  - [normal approximation](extend-approximation.md)
  - [optimizer](extend-optimizer.md)
  - [problem solving strategy](extend-strategy.md)

!!! warning
    
    Writing extensions necessarily involves working with non-exported functions.
    In addition to reading the vignettes linked above,
    it is recommended to also read the relevant parts of [the package sources](https://git.sr.ht/%7Elsandig/Kirstine.jl/tree/main/item/src/).
    While the public API will be kept stable between minor releases,
    breaking changes to the internals can occur.
