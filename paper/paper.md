---
title: 'Kirstine.jl: A Julia Package for Bayesian Optimal Design of Experiments'
tags:
  - Julia
  - design of experiments
  - Bayesian statistics
authors:
  - name: Ludger Sandig
    orcid: 0000-0002-3174-3275
    affiliation: 1
affiliations:
 - name: Department of Statistics, TU Dortmund University, Germany
   index: 1
date: 8 January 2024
bibliography: paper.bib
---

# Summary

Good design of planned experiments increases the precision of parameter estimates.
In a dose-response study,
for example,
the credible intervals for the response curve's parameters are shorter
when the dose levels have been chosen carefully.
The general mathematical framework
that guides the choice of values and weights for the covariates in a regression model
is called optimal design theory.
In the special case of nonlinear regression models,
a researcher must specify prior knowledge about the model parameters.
Accounting here for a full distribution of parameter values
produces designs
that are more robust
than designs for a single best guess.
To help computing such designs efficiently for general nonlinear regression models,
we propose the `Julia` [@bezanson-2017-julia] package `Kirstine.jl`.

# Mathematical Background

Consider a nonlinear regression model with mean function $\mu : X \times \Theta \to \mathbb{R}^m$,
known covariance matrix $\Sigma$,
and compact _design region_ $X \subset \mathbb{R}^d$,
where we want to design an experiment for estimating the unknown parameter $\theta$.
In nonlinear optimal design theory [@fedorov-2013-optim-desig],
we represent the design by a probability measure $\xi$ on $X$.
For every such _design measure_ we define the _normalized information matrix_
$$
\mathrm{M}(\xi, \theta)
=
\int (\mathrm{D}_{\theta}\mu(x, \theta))'
\Sigma^{-1}
(\mathrm{D}_{\theta} \mu(x, \theta))\, \xi(\mathrm{d}x) ,
$$
where $\mathrm{D}_{\theta}\mu$ denotes the Jacobian matrix of $\mu$ with respect to $\theta$.
To obtain, on average, small confidence or posterior credible intervals,
we aim to construct a design $\xi^*$
that maximizes a functional $\phi$ of the normalized information matrix,
with popular choices being the *D-* or *A-criterion*
$$
\phi_{\mathrm{D}}(\mathrm{M}(\xi, \theta))=\log\det(\mathrm{M}(\xi, \theta)),
\quad
\phi_{\mathrm{A}}(\mathrm{M}(\xi, \theta))=-\operatorname{tr}(\mathrm{M}(\xi, \theta)^{-1}).
$$
Since $\mathrm{M}(\xi, \theta)$ still depends on the unknown $\theta$,
we either plug in a best guess $\theta_0$
and obtain a _locally optimal_ design problem,
or we try to find a _Bayesian optimal_ design
that maximizes the average of $\phi$ with respect to a prior distribution [@chaloner-1995-bayes-exper-desig].
Having obtained a candidate design $\xi^*$,
we then apply an equivalence theorem from infinite-dimensional convex analysis
to verify
that the design $\xi^*$ is indeed optimal.

To find a candidate design in practice,
we must make three simplifications.
We first have to approximate the prior expectation,
since the integral is not tractable analytically.
*Monte-Carlo (MC) integration*
$$
\int \phi(\mathrm{M}(\xi, \theta))\,p(\theta)\,\mathrm{d}\theta
\approx
\frac{1}{S} \sum_{s=1}^{S} \phi(\mathrm{M}(\xi, \theta^{(s)}))
$$
is a versatile method for that
because we can use it with a sample
$\theta^{(1)},\dots,\theta^{(S)}$, $S\in\mathbb{N}$
from an arbitrary distribution with density $p(\theta)$.
Next,
we reduce the search space from all probability measures on $X$
to the subset of those
that are discrete
and have $K\in\mathbb{N}$ design points.
The optimal $K^*$ is usually not known beforehand,
but as long as we do not enforce unique design points,
we may guess at a $K > K^*$ and will still be able to find the solution.
Finally,
we have to choose one of the many proposed algorithms [@ryan-2015-review-modern]
for maximizing the objective function numerically.

# Statement of Need

Currently, most open-source experimental design software is implemented in `R`.
There is also a `Julia` package that focuses on
[factorial design problems](https://github.com/phrb/ExperimentalDesign.jl)
but does not address nonlinear regression.
Among the `R` packages [on CRAN](https://cran.r-project.org/view=ExperimentalDesign),
only four deal with nonlinear regression models,
and all of them have to make a tradeoff between speed and flexibility.
With MC integration,
thousands of information matrices have to be computed for one evaluation of the objective function.
These matrix operations are a performance bottleneck,
since function arguments in `R` are passed by value.
To avoid the memory overhead,
package authors can implement internals in `C`
and pass around pointers to pre-allocated matrices.
However,
this requires the users to be proficient in `C`
in order to supply the Jacobian matrices $\mathrm{D}_{\theta}\mu$ of their models.
Consequently,
packages either just accept the slowdown
[@masoudi-2020-icaod],
recommend using `C++`
[@overstall-2020-acebayes],
or come with a small set of models pre-specified
[@bornkamp-2023-dosef;@nyberg-2012-poped;@foracchia-2004-poped].
Hence a design package is needed
where knowledge of only one language is required
for efficiently implementing arbitrary nonlinear regression models.

`Kirstine.jl` attempts to fill this gap in the design software ecosystem.
The package achieves modeling flexibility through `Julia`'s multiple dispatch mechanism,
and performs matrix operations efficiently by passing object references to `BLAS` and `LAPACK` routines.
It currently implements
the D- and A-criterion,
vector-valued measurements,
posterior transformations of $\theta$ via the Delta method,
box-shaped design regions of arbitrary dimension,
particle swarm optimization [@kennedy-1995-partic],
and a variant of Fedorov's coordinate exchange algorithm [@yang-2013-optim-desig].
Locally optimal design is supported implicitly.
Since user-defined `Julia` code does not inherently incur performance penalties,
specific regression models are not supplied.
Instead,
users should first define subtypes of `NonlinearRegression`, `Covariate`, `Parameter`, and `CovariateParameterization`,
and then add methods for a few functions
that dispatch on them.
Optionally,
one of `Julia`'s automatic differentiation packages can be used.
`Kirstine.jl` has a modular and readable code base,
which enables users to extend the package's functionality
with drop-in replacements for new criteria,
design regions,
or even custom optimization algorithms.
Thanks to multiple dispatch,
no changes are required in the package internals.
This way, `Kirstine.jl` provides an additional level of flexibility without sacrificing efficiency.

# Acknowledgments

This work has been supported by the Research Training Group
_Biostatistical Methods for High-Dimensional Data in Toxicology_
(RTG 2624, Project P2), funded by the Deutsche Forschungsgemeinschaft
(DFG, German Research Foundation – Project Number 427806116).

# References
