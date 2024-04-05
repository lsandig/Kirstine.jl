# Kirstine.jl

<!-- SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de> -->
<!-- SPDX-License-Identifier: GFDL-1.3-or-later -->

A [Julia][julia-url] package for Bayesian optimal experimental design with nonlinear regression models.

[repo][repo-url] | [docs][docs-url] | [change log][change-url]

[julia-url]: https://julialang.org
[repo-url]: https://sr.ht/~lsandig/Kirstine.jl/
[docs-url]: https://lsandig.srht.site/Kirstine.jl/index.html
[change-url]: https://git.sr.ht/~lsandig/Kirstine.jl/refs

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

[Markdown](docs/src/), [HTML][docs-url]

To get started, read the [tutorial](https://lsandig.srht.site/Kirstine.jl/tutorial.html).

For a change log,
see the list of [annotated tags][change-url].

There is also a separate repository with [additional examples](https://git.sr.ht/~lsandig/kirstine-examples).

## License

Kirstine.jl is free and open source software.
The code is licensed under GPL-3.0 or later,
and the documentation under GFDL-1.3 or later.

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
