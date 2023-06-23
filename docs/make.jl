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
                  "Getting Started" => "getting-started.md",
                  "Handling Transformations" => "handling-transformations.md",
                  ],
         )
