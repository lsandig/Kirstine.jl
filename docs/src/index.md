# Kirstine.jl

*Bayesian Optimal Design for Nonlinear Regression.*

```@eval
import Pkg
ver = Pkg.project().version
"Version $(ver)"
```

Kirstine.jl is free and open source software.
The code is licensed under GPL-3.0 or later,
and the documentation under GFDL-1.3 or later.

## Project Status

Alpha.
Expect bugs and breaking changes.
Documentation is still sparse in some places.

## Features

  - arbitrary nonlinear regression models
  - vector-valued response variable
  - Bayesian and locally optimal design
  - design criteria: D, A
  - separation of design variables and model covariates
  - particle swarm optimization
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

## Contributing

There is a [git repository and issue tracker at sourcehut](https://sr.ht/%7Elsandig/Kirstine.jl/).

If you have already solved design problems with other software packages,
you can try to replicate your solutions with Kirstine.jl.
I'd be grateful to hear about your results,
especially if they differ from what you expect.
Ditto if you think Kirstine.jl is missing some crucial feature.

[^FL13]: Valerii V. Fedorov and Sergei L. Leonov (2013). Optimal design for nonlinear response models. CRC Press. [doi:10.1201/b15054](https://dx.doi.org/10.1201/b15054)
[^CV95]: Kathryn Chaloner and Isabella Verdinelli (1995). Bayesian experimental design: a review. Statistical Science, 10(3), 273–304. [doi:10.1214/ss/1177009939](http://dx.doi.org/10.1214/ss/1177009939)
