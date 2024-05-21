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

## Features

  - arbitrary nonlinear regression models
  - scalar or vector-valued response variable
  - variance-covariance matrix may depend on covariate
  - Bayesian and locally optimal design
  - design criteria: D, A
  - composite design problems
  - separation of design variables and model covariates
  - particle swarm optimization
  - direct maximization and exchange algorithm
  - modular and extendable
  - minimal dependencies

## Installation

Kirstine.jl is in the [General](https://github.com/JuliaRegistries/General) Julia package registry.
You can install it with

```julia
Pkg.add("Kirstine")
```

Alternatively,
you can get it directly from sourcehut:

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

For a change log,
see the list of [annotated tags](https://git.sr.ht/%7Elsandig/Kirstine.jl/refs).

There is also a separate repository with [additional examples](https://git.sr.ht/%7Elsandig/kirstine-examples).

## License

Kirstine.jl is free and open source software.
The code is licensed under GPL-3.0 or later,
and the documentation under GFDL-1.3 or later.

## Citing

Kirstine.jl is [published in the Journal of Open Source Software](https://joss.theoj.org/papers/10.21105/joss.06424).
You can cite the package like this:

```
@article{,
  author =  {Ludger Sandig},
  title =   {Kirstine.jl: A Julia Package for Bayesian Optimal Design of Experiments},
  journal = {Journal of Open Source Software},
  volume =  9,
  number =  97,
  pages =   6424,
  year =    2024,
  doi =     {10.21105/joss.06424},
}
```

## Rationale

Why yet another package for optimal design?

In `R`, there is already the [ICAOD](https://cran.r-project.org/package=ICAOD) package
for finding optimal designs in any model
of which you can implement the Fisher information matrix.
However,
it is not as efficient as it could be
and the code base is hard to extend for more complicated design problems.
There are also various other packages (e.g., [DoseFinding](https://cran.r-project.org/package=DoseFinding) or [PopED](https://cran.r-project.org/package=PopED))
for optimal design with special criteria or special kinds of nonlinear regression models.
In `Julia`, there is already [ExperimentalDesign.jl](https://github.com/phrb/ExperimentalDesign.jl)
for block, factorial and response-surface designs in linear regression/ANOVA and polynomial regression models.

Kirstine.jl is an attempt to provide applied statisticians
with a `Julia` package for optimal design in arbitrary nonlinear regression models.
Its development goals are speed and modularity,
as well as a small, stable and well-documented code base.

## Contributing

The source code of Kirstine.jl is managed on [sourcehut](https://sr.ht/%7Elsandig/Kirstine.jl/).

Please post usage questions and general discussion on the [kirstine-users](https://lists.sr.ht/%7Elsandig/kirstine-users) mailing list.

Only bug reports should be filed on the [issue tracker](https://todo.sr.ht/%7Elsandig/Kirstine.jl/).

Patches are welcome and should be submitted via [`git send-email`](https://git-send-email.io)
to the [kirstine-devel](https://lists.sr.ht/%7Elsandig/kirstine-devel) mailing list.
Please make sure that your patch does not break any existing tests
and includes new tests for any functionality it adds.

[^FL13]: Valerii V. Fedorov and Sergei L. Leonov (2013). Optimal design for nonlinear response models. CRC Press. [doi:10.1201/b15054](https://dx.doi.org/10.1201/b15054)
[^CV95]: Kathryn Chaloner and Isabella Verdinelli (1995). Bayesian experimental design: a review. Statistical Science, 10(3), 273–304. [doi:10.1214/ss/1177009939](http://dx.doi.org/10.1214/ss/1177009939)
