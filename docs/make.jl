push!(LOAD_PATH,"../src/")
using Documenter, NumericalRelaxation

makedocs(sitename="NumericalRelaxation.jl",
         modules=[NumericalRelaxation],
         authors="Maximilian Köhler",
         pages=["Home"=> "index.md",
                "One-Dimensional Convexification" => "api/oned-convexification.md",
                "Rank-One Convexification" => "api/r1convexification.md"])
deploydocs(
    repo = "github.com/koehlerson/NumericalRelaxation.jl.git",
    push_preview=true,
)
