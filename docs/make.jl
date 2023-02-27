using Documenter, Kirstine

DocMeta.setdocmeta!(Kirstine, :DocTestSetup, :(using Kirstine); recursive=true)

makedocs(modules = [Kirstine],
         sitename = "Kirstine.jl",
         strict = true,
         format = Documenter.HTML(prettyurls = false,
                                  edit_link = nothing,
                                  footer = nothing,
                                  ),
         pages = ["Home" => "index.md"],
         )
