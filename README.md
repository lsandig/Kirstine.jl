# Kirstine.jl

<!-- SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de> -->
<!-- SPDX-License-Identifier: GFDL-1.3-or-later -->

A [Julia][julia-url] package for Bayesian optimal experimental design with nonlinear regression models.

[julia-url]: https://julialang.org

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
- variance-covariance matrix may depend on covariate
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

[Markdown](docs/src/), [HTML](https://lsandig.srht.site/Kirstine.jl/index.html)

To get started, read the [tutorial](https://lsandig.srht.site/Kirstine.jl/tutorial.html).

## Contributing

There is a [git repository at sourcehut](https://git.sr.ht/~lsandig/Kirstine.jl).

If you have already solved design problems with other software packages,
you can try to replicate your solutions with Kirstine.jl.
I'd be grateful to hear about your results,
especially if they differ from what you expect.
Ditto if you think Kirstine.jl is missing some crucial feature.
