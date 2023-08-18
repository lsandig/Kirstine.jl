# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

using Documenter, Kirstine

DocMeta.setdocmeta!(Kirstine, :DocTestSetup, :(using Kirstine); recursive=true)

math_macros = Dict(
    # Integers
    raw"\DimCovariate" => raw"c",
    raw"\DimDesignRegion" => raw"d",
    raw"\DimParameter" => raw"r",
    raw"\DimTransformedParameter" => raw"t",
    raw"\DimUnit" => raw"m",
    raw"\IndexDesignPoint" => raw"k",
    raw"\IndexUnit" => raw"i",
    raw"\NumDesignPoints" => raw"K",
    raw"\SampleSize" => raw"n",
    # Fixed model parameters
    raw"\ScalarUnitVariance" => raw"σ^2",
    raw"\UnitCovariance" => raw"Σ",
    # Sets
    raw"\AllDesignMeasures" => raw"Δ",
    raw"\CovariateSet" => raw"X",
    raw"\DesignRegion" => raw"Z",
    raw"\IndexSet" => raw"I",
    raw"\ParameterSet" => raw"Θ",
    raw"\Reals" => raw"ℝ",
    raw"\SNNDMatrices" => raw"\mathrm{S}_{+}^{#1}",
    # Variables
    raw"\Covariate" => raw"x",
    raw"\DesignMeasure" => raw"ζ",
    raw"\DesignMeasureDirection" => raw"δ",
    raw"\DesignPoint" => raw"z",
    raw"\DesignWeight" => raw"w",
    raw"\Parameter" => raw"θ",
    raw"\SomeMatrix" => raw"M",
    raw"\Unit" => raw"y",
    # Functions
    raw"\CovariateParameterization" => raw"C",
    raw"\DesignCriterion" => raw"Ψ",
    raw"\FisherMatrix" => raw"\operatorname{F}",
    raw"\GateauxDerivative" => raw"ψ",
    raw"\MeanFunction" => raw"μ",
    raw"\NIMatrix" => raw"\operatorname{M}",
    raw"\Objective" => raw"f",
    raw"\PriorDensity" => raw"p",
    raw"\RelEff" => raw"\operatorname{RelEff}",
    raw"\TNIMatrix" => raw"\operatorname{M}_T",
    raw"\Trace" => raw"\operatorname{tr}",
    raw"\Transformation" => raw"T",
    # Distributions
    raw"\DiracDist" => raw"\operatorname{Dirac}",
    raw"\MvNormDist" => raw"\operatorname{MvNorm}",
    # Miscellaneous
    raw"\Int" => raw"∫_{#1}#2\operatorname{d}\!#3",
    raw"\IntD" => raw"∫_{#1}#2\,#3\operatorname{d}\!#4",
    raw"\IntM" => raw"∫_{#1}#2\,#3(\operatorname{d}\!#4)",
    raw"\MatDeriv" => raw"\frac{∂#1}{∂#2}(#3)",
    raw"\TotalDiff" => raw"\operatorname{D\!}",
    raw"\simappr" => raw"\overset{\mathrm{appr}}{\sim}",
    raw"\simiid" => raw"\overset{\mathrm{iid}}{\sim}",
    # Abbreviations
    raw"\PosteriorDensity" => raw"p(\Parameter\mid\Unit_1,…,\Unit_{\SampleSize})",
)

makedocs(modules = [Kirstine],
         sitename = "Kirstine.jl",
         strict = true,
         # doctest = :fix,
         format = Documenter.HTML(
             prettyurls = false,
             edit_link = nothing,
             footer = nothing,
             mathengine = KaTeX(Dict(:macros => math_macros)),
         ),
         pages = ["Home" => "index.md",
                  "Getting Started" => "tutorial.md",
                  "Examples" => [
                      "Transformations" => "transformations.md",
                      "Two Design Variables" => "dtr.md",
                      "Locally Optimal Design" => "locally-optimal.md",
                      "Discrete Prior" => "discrete-prior.md",
                  ],
                  "API Reference" => "api.md",
                  "Mathematical Background" => "math.md",
                  ],
         )
