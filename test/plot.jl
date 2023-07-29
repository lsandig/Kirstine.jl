# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module PlotTests
using Test
using Kirstine
using Plots: plot

@testset "plot.jl" begin
    @testset "plot" begin
        @test_throws "1- or 2-dimensional" plot(one_point_design([1, 2, 3]))
    end
end
end
