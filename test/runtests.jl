# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

using Test

@testset "Kirstine" begin
    include("pso.jl")
    include("designmeasure.jl")
    include("types.jl")
    include("design-common.jl")
    include("design-doptimal.jl")
    include("design-aoptimal.jl")
    include("util.jl")
    include("plot.jl")
end
