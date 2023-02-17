using Documenter, Kirstine

makedocs(sitename="Kirstine.jl",
         format = Documenter.HTML(prettyurls = false, edit_link = nothing,
                                  footer = nothing),
         pages = ["Home" => "index.md"])
