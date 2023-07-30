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

## Contributing

There is a [git repository at sourcehut](https://git.sr.ht/%7Elsandig/Kirstine.jl).

If you have already solved design problems with other software packages,
you can try to replicate your solutions with Kirstine.jl.
I'd be grateful to hear about your results,
especially if they differ from what you expect.
Ditto if you think Kirstine.jl is missing some crucial feature.
