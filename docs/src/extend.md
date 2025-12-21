# Extending Kirstine

Since Kirstine.jl's code base is quite modular,
you can extend the package's functionality in multiple ways.
All of the high-level logic is implemented in terms of abstract supertypes.
For example,
a general `DesignRegion` is used wherever possible,
and the specifics of a `DesignInterval`
are implemented in a few short functions.
Hence you can,
in your own code _outside_ the package,
define a custom subtype of `DesignRegion` and the handful of required methods,
and it will seamlessly blend in.
Such is the magic of multiple dispatch.
This does not only work with the parts of a `DesignProblem`,
but also with the optimization algorithms.
They are represented by subtypes of `Kirstine.Optimizer` (for the low-level work)
and `Kirstine.ProblemSolvingStrategy` (for high-level design-specific stuff).
Moreover, the finicky logic for moving around `DesignMeasures` is encapsulated,
so your custom `Kirstine.Optimizer` subtype will only have to deal with `AbstractPoints`.
The following example vignettes will show you how to implement

  - a nonlogarithmized determinant [design criterion](extend-criterion.md)
  - an ellipsoidal [design region](extend-region.md)
  - a nonlinear logistic regression [model](extend-model.md)
  - a regularizing [normal approximation](extend-approximation.md)
  - a variant of the imperialist competitive algorithm as a low-level [optimizer](extend-optimizer.md)
  - a multi-run [problem solving strategy](extend-strategy.md)

!!! warning
    
    Writing extensions necessarily involves working with nonexported functions.
    In addition to reading the vignettes linked above,
    it is recommended to also read the relevant parts of [the package sources](https://git.sr.ht/%7Elsandig/Kirstine.jl/tree/main/item/src/).
    While the [public API](api.md) will be kept stable between minor releases after v1.0,
    breaking changes can occur in the internals.
    If you need to extend Kirstine.jl for your work,
    consider setting an explicit version in your `Project.toml`.
