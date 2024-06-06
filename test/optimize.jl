# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module OptimizeTests
using Test
using Kirstine

@testset "optimize.jl" begin
    @testset "prettydifftime" begin
        let h = 1, m = 42, s = 3, ms = 141.592, t = 3600 * h + 60 * m + s + ms / 1000
            @test Kirstine.prettydifftime(t) == "1:42:03.141"
            @test Kirstine.prettydifftime(-t) == "-1:42:03.141"
            @test Kirstine.prettydifftime(0) == "0:00:00.000"
        end
    end
end
end
