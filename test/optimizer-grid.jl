# SPDX-FileCopyrightText: 2024 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

module GridSearchTests
using Test
using Kirstine

include("example-pnt.jl")

function Kirstine.construct_grid(
    gs::ProductGrid,
    c::BoxConstraints,
    prototypes::AbstractVector{Pnt},
)
    ranges = map((l, u, n) -> range(l, u; length = n), c.lb.x, c.ub.x, gs.n_points)
    grid_points = reshape([Pnt(collect(x)) for x in Iterators.product(ranges...)], :)
    return grid_points
end

@testset "optimizer-grid.jl" begin
    @testset "ProductGrid" begin
        @test ProductGrid(4, 2) == ProductGrid{2}((4, 2))
        @test_throws "at least 2" ProductGrid(1, 2, 3)
    end

    @testset "GridSearchState" begin
        let x = [Pnt([1, 2]), Pnt([3, 4])], fx1 = [5, 6], fx2 = [7, 8, 9]
            gss = Kirstine.GridSearchState(x, fx1)

            @test gss isa Kirstine.GridSearchState{Pnt,PntDiff}
            @test_throws "must be identical" Kirstine.GridSearchState(x, fx2)
            @test_throws "can't be empty" Kirstine.GridSearchState(Pnt[], Float64[])
        end
    end

    @testset "construct_grid" begin
        let bbg = ProductGrid(2, 3),
            c = BoxConstraints(Pnt([0, 0]), Pnt([1, 1])),
            ref_x = [[0, 0], [0, 0.5], [0, 1], [1, 0], [1, 0.5], [1, 1]]

            pts = Kirstine.construct_grid(bbg, c, Pnt[])
            pts_x = [p.x for p in pts]

            @test pts isa Vector{Pnt}
            @test isempty(setdiff(pts_x, ref_x))
            @test length(pts) == 6
        end
    end

    @testset "optimizer_state" begin
        let xstar = ones(2),
            f(p) = -sum((p.x .- xstar) .^ 2),
            gs = GridSearch(ProductGrid(3, 3)),
            p = Pnt([0, 0]),
            c = BoxConstraints(Pnt([-1, -1]), Pnt([4, 4]))

            gss1 = Kirstine.optimizer_state(f, gs, Pnt[], c)
            gss2 = Kirstine.optimizer_state(f, gs, [Pnt([0, 0]), Pnt([-1, -1])], c)

            @test gss1 isa Kirstine.GridSearchState
            @test length(gss1.x) == 9
            # prototypes are appended, the whole list is made unique
            @test length(gss2.x) == 10
        end
    end

    @testset "optimize" begin
        let xstar = ones(2),
            f(p) = -sum((p.x .- xstar) .^ 2),
            # grid that includes xstar
            gs1 = GridSearch(ProductGrid(11, 11)),
            # grid that does not include xstar
            gs2 = GridSearch(ProductGrid(3, 3)),
            constr = BoxConstraints(Pnt([-1, -1]), Pnt([4, 4]))

            r1 = Kirstine.optimize(f, gs1, Pnt[], constr; trace_state = true)
            r2 = Kirstine.optimize(f, gs2, Pnt[], constr)

            @test r1.maximizer.x == xstar
            @test r1.maximum == 0
            @test length(r1.trace_state) == 1
            @test r1.trace_x[1].x == xstar
            @test r1.trace_fx[1] == 0
            @test r1.n_eval == 121

            @test r2.maximizer.x == [1.5, 1.5]
            @test r2.maximum == -0.5
        end
    end
end
end
