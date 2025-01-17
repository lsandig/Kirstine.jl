# SPDX-FileCopyrightText: 2025 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

using BenchmarkTools: @ballocated

macro test_noalloc(expr)
    @test(@ballocated($expr; evals = 1, samples = 1) == 0)
end
