# Extending Kirstine

The code of Kirstine.jl is written in a modular fashion.
This means that you can write drop-in replacements for many of the types that constitute a [`DesignProblem`](@ref)
in order to implement any functionality that you might be missing.
You can even implement your own high- or low-level optimization algorithms.

The following pages show how to implement custom versions of the following objects:

  - design criterion
  - design region
  - model
  - transformation
  - normal approximation
  - optimizer
  - problem solving strategy

!!! warning
    
    Writing extensions necessarily involves working with non-exported functions.
    While stability between versions is one of Kirstines development goals,
    you should expect them to break more frequently than the public API.
