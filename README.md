# Kirstine.jl

<!-- SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de> -->
<!-- SPDX-License-Identifier: GFDL-1.3-or-later -->

A [Julia][julia-url] package for Bayesian optimal experimental design with nonlinear regression models.

[julia-url]: https://julialang.org

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

Kirstine.jl is in the [General](https://github.com/JuliaRegistries/General) Julia package registry.
You can install it with

```julia
Pkg.add("Kirstine")
```

## Documentation

[Markdown](docs/src/), [HTML](https://lsandig.srht.site/Kirstine.jl/index.html)

To get started, read the [tutorial](https://lsandig.srht.site/Kirstine.jl/tutorial.html).

For a change log,
see the list of [annotated tags](https://git.sr.ht/~lsandig/Kirstine.jl/refs).

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

Note that the `Julia` package [ExperimentalDesign.jl][edjl-url]
and most of the [design packages on CRAN][craned-url]
are for finding block, factorial, or response-surface designs.
These are different tasks than what Kirstine.jl attempts to do.

[edjl-url]: https://github.com/phrb/ExperimentalDesign.jl
[craned-url]: https://cran.r-project.org/view=ExperimentalDesign

## Contributing

There is a [git repository and issue tracker at sourcehut](https://sr.ht/~lsandig/Kirstine.jl/).

If you have already solved design problems with other software packages,
you can try to replicate your solutions with Kirstine.jl.
I'd be grateful to hear about your results,
especially if they differ from what you expect.
Ditto if you think Kirstine.jl is missing some crucial feature.
