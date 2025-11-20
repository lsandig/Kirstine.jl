# SPDX-FileCopyrightText: 2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module NormalApproximationTests
using Test
using Kirstine
using Random: seed!

@testset "normalapproximation.jl" begin
    @testset "informationmatrix!" begin
        # The `FisherMatrix` normal approximation should just give the average Fisher
        # information matrix as NIM.
        let _ = seed!(1234),
            afm = rand(3, 3),
            afm_copy = deepcopy(afm),
            na = FisherMatrix(),
            res = Kirstine.informationmatrix!(afm, na)

            @test res === afm
            @test res == afm_copy
        end
    end
end
end
