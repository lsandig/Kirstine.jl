# SPDX-FileCopyrightText: 2023 Ludger Sandig <sandig@statistik.tu-dortmund.de>
# SPDX-License-Identifier: GPL-3.0-or-later

using Documenter, Kirstine

DocMeta.setdocmeta!(Kirstine, :DocTestSetup, :(using Kirstine); recursive=true)

makedocs(modules = [Kirstine],
         sitename = "Kirstine.jl",
         strict = true,
         # doctest = :fix,
         format = Documenter.HTML(prettyurls = true,
                                  edit_link = nothing,
                                  footer = nothing,
                                  ),
         pages = ["Home" => "index.md",
                  "Getting Started" => "tutorial.md",
                  "Locally Optimal Design" => "locally-optimal.md",
                  "Weighted Prior Sample" => "prior-weight.md",
                  "Using Transformations" => "transformations.md",
                  "API Reference" => "api.md",
                  ],
         )
