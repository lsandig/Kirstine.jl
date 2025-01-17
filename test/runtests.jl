# SPDX-FileCopyrightText: 2023,2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

using Test

@testset "Kirstine" begin
    include("optimize.jl")
    include("optimizer-pso.jl")
    include("optimizer-grid.jl")
    include("optimizer-frank-wolfe.jl")
    include("designregion.jl")
    include("designmeasure.jl")
    include("designmeasure-abstractpoint.jl")
    include("priorknowledge.jl")
    include("transformation.jl")
    include("user-import.jl")
    include("user-using.jl")
    include("design-common.jl")
    include("model-nonlinear-regression.jl")
    include("criterion-d.jl")
    include("criterion-a.jl")
    include("designproblem.jl")
    include("composite-problem.jl")
    include("solve-directmaximization.jl")
    include("solve-exchange.jl")
    include("plot.jl")
end
