# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

using Documenter, Kirstine

struct SourceHut <: Remotes.Remote
    user::String
    repo::String
    tracker::String
end

function Remotes.repourl(remote::SourceHut)
    return "https://git.sr.ht/$(remote.user)/$(remote.repo)"
end

function Remotes.fileurl(remote::SourceHut, ref, filename, linerange)
    url = "$(Remotes.repourl(remote))/tree/$(ref)/item/$(filename)"
    if isnothing(linerange)
        return url
    end
    a, b = first(linerange), last(linerange)
    return (a == b) ? "$(url)#L$(a)" : "$(url)#L$(a)-$(b)"
end

function Remotes.issueurl(remote::SourceHut, issuenumber)
    return "https://todo.sr.ht/$(remote.user)/$(remote.tracker)/$(issuenumber)"
end

ENV["GKSwstype"] = "100"

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
    raw"\SignedMeasure" => raw"η",
    raw"\SomeMatrix" => raw"M",
    raw"\Unit" => raw"y",
    # Functions
    raw"\AverageFisherMatrix" => raw"\operatorname{\bar{F}}",
    raw"\CovariateParameterization" => raw"C",
    raw"\DesignCriterion" => raw"Ψ",
    raw"\FisherMatrix" => raw"\operatorname{F}",
    raw"\GateauxDerivative" => raw"ψ",
    raw"\LogLikelihood" => raw"\ell",
    raw"\MeanFunction" => raw"μ",
    raw"\NIMatrix" => raw"\operatorname{M}",
    raw"\Objective" => raw"f",
    raw"\PriorDensity" => raw"p",
    raw"\RelEff" => raw"\operatorname{RelEff}",
    raw"\Sensitivity" => raw"φ",
    raw"\TNIMatrix" => raw"\operatorname{M}_T",
    raw"\Trace" => raw"\operatorname{tr}",
    raw"\Transformation" => raw"T",
    # Distributions
    raw"\DiracDist" => raw"\operatorname{Dirac}",
    raw"\MvNormDist" => raw"\operatorname{MvNorm}",
    # Miscellaneous
    raw"\Covariance" => raw"ℂ",
    raw"\Expectation" => raw"𝔼",
    raw"\Hessian" => raw"\operatorname{H\!}",
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
         repo = SourceHut("~lsandig", "Kirstine.jl", "Kirstine.jl"),
         # doctest = :fix,
         # draft = true,
         # warnonly = true,
         format = Documenter.HTML(
             prettyurls = false,
             edit_link = nothing,
             footer = nothing,
             mathengine = KaTeX(Dict(:macros => math_macros)),
         ),
         pages = [
             "Home" => "index.md",
             "Getting Started" => "tutorial.md",
             "Examples" => [
                 "Transformations" => "transformations.md",
                 "Multiple Design Variables" => "dtr.md",
                 "Locally Optimal Design" => "locally-optimal.md",
                 "Discrete Prior" => "discrete-prior.md",
                 "Automatic Differentiation" => "autodiff.md",
             ],
             "Mathematical Background" => "math.md",
             "API Reference" => "api.md",
             "Extending Kirstine" => [
                 "Overview" => "extend.md",
                 "Design Criterion" => "extend-criterion.md",
                 "Design Region" => "extend-region.md",
                 "Normal Approximation" => "extend-approximation.md",
                 "Optimizer" => "extend-optimizer.md",
             ],
         ],
         )
