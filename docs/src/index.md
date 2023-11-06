# Kirstine.jl

*Bayesian Optimal Design for Nonlinear Regression.*

```@eval
import Pkg, Markdown
ver = Pkg.project().version
Markdown.MD("Version $(ver)")
```

## Project Status

Beta.
Mostly stable, breaking changes will be mostly cosmetic.
Documentation is still sparse in some places.

## Features

  - arbitrary nonlinear regression models
  - scalar or vector-valued response variable
  - variance-covariance matrix may depend on covariate
  - Bayesian and locally optimal design
  - design criteria: D, A
  - separation of design variables and model covariates
  - particle swarm optimization
  - direct maximization and exchange algorithm
  - modular and extendable
  - minimal dependencies

## Installation

Kirstine.jl will be registered once it is stable.
Until then, you can install it with

```julia
Pkg.add(url = "https://git.sr.ht/~lsandig/Kirstine.jl")
```

## Documentation

To get started, read the [tutorial](tutorial.md).

Familiarity with the following topics is required:

  - Nonlinear design theory, e.g.
    
      + Chapter 2 from Fedorov/Leonov's book[^FL13] for the general theory
      + Chaloner/Verdinelli's review article[^CV95] for the Bayesian approach
    
    (There is also an overview of the [notation](math.md) used in this package.)

  - Julia's [type system](https://docs.julialang.org/en/v1/manual/types/)
    and adding [methods](https://docs.julialang.org/en/v1/manual/methods/)
    for user-defined types.

## License

Kirstine.jl is free and open source software.
The code is licensed under GPL-3.0 or later,
and the documentation under GFDL-1.3 or later.

## Rationale

Why another package for optimal design?
Currently, `R` packages for design with nonlinear regression have to make a compromise between flexibility
(allow arbitrary models written in `R`)
and execution speed
(implement only a few models in `C`).
In Julia, we can write both concise high-level code and efficient low-level code in the same language.
This way,
Kirstine.jl attempts to provide applied statisticians with a tool
for efficiently finding designs for arbitrarily complex nonlinear regression models.

Note that the `Julia` package [ExperimentalDesign.jl](https://github.com/phrb/ExperimentalDesign.jl)
and most of the [design packages on CRAN](https://cran.r-project.org/view=ExperimentalDesign)
are for finding block, factorial, or response-surface designs.
These are different tasks than what Kirstine.jl attempts to do.

## Contributing

There is a [git repository and issue tracker at sourcehut](https://sr.ht/%7Elsandig/Kirstine.jl/).

If you have already solved design problems with other software packages,
you can try to replicate your solutions with Kirstine.jl.
I'd be grateful to hear about your results,
especially if they differ from what you expect.
Ditto if you think Kirstine.jl is missing some crucial feature.

[^FL13]: Valerii V. Fedorov and Sergei L. Leonov (2013). Optimal design for nonlinear response models. CRC Press. [doi:10.1201/b15054](https://dx.doi.org/10.1201/b15054)
[^CV95]: Kathryn Chaloner and Isabella Verdinelli (1995). Bayesian experimental design: a review. Statistical Science, 10(3), 273–304. [doi:10.1214/ss/1177009939](http://dx.doi.org/10.1214/ss/1177009939)
