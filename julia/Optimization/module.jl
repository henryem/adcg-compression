module Optimization

include("./Loss.jl")
include("./Encoders.jl")
include("./Sparsifiers.jl")
include("./LassoSolvers.jl")
include("./TrivialEncoder.jl")
include("./LsEncoder.jl")
include("./AlignedDirectionFinder.jl")
include("./FiniteDimQpSolver.jl")
include("./LocalImprovementFinder.jl")
include("./AdcgEncoder.jl")

end