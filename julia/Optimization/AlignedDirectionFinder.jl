export AlignedDirectionFinder, leastAlignedDirection
export WarmstartFinder
export GridFinder
export NLoptDirectionFinder

using NLopt

abstract AlignedDirectionFinder

# Finds a parameter vector p in @space so that <synthesize(T(p),1.0), direction>
# is (approximately) minimized.  (Typically will be negative.)
function leastAlignedDirection{T <: ParameterizedTransform}(this:: AlignedDirectionFinder, direction:: VectorizedImage, space:: ParameterSpace{T})
  leastAlignedDirection(this, direction, space, zeros(dimension(space)))
end

# As leastAlignedDirection, but with a starting location.
function leastAlignedDirection{T <: ParameterizedTransform}(this:: AlignedDirectionFinder, direction:: VectorizedImage, space:: ParameterSpace{T}, start:: Vector{Float64})
  raiseAbstract("leastAlignedDirection", this)
end


# An issue with ordinary gradient methods is that the gradient may be very
# small (or round to 0 in floating point) if we start too far away from a good
# parameter point.  Also, since the problem is nonconvex, we may find a bad
# local minimum.  We attempt to alleviate both problems by starting the search
# at an okay starting location.
immutable WarmstartFinder <: AlignedDirectionFinder
  first:: AlignedDirectionFinder
  next:: AlignedDirectionFinder
end

function leastAlignedDirection{T <: ParameterizedTransform}(this:: WarmstartFinder, direction:: VectorizedImage, space:: ParameterSpace{T}, ignored:: Vector{Float64})
  println("Warm-starting...")
  @time const start = leastAlignedDirection(this.first, direction, space)
  println("Optimizing after warm start...")
  @time const result = leastAlignedDirection(this.next, direction, space, start)
  result
end


type GridFinder <: AlignedDirectionFinder
  gridDims:: Union(Int64,Vector{Int64})
  hasCache:: Bool
  cachedCandidates:: Vector{Transform}
  cachedSpace:: ParameterSpace
  
  function GridFinder(gridDims:: Union(Int64,Vector{Int64}))
    new(gridDims, false)
  end
end

function leastAlignedDirection{T <: ParameterizedTransform}(this:: GridFinder, direction:: VectorizedImage, space:: ParameterSpace{T})
  # A bit of a hack; we optimize for the common case where we reuse the 
  # GridFinder with the same space and several directions.
  if !this.hasCache || this.cachedSpace != space
    this.hasCache = true
    this.cachedCandidates = grid(space, this.gridDims)
    this.cachedSpace = space
  end
  bestCandidateIdx = 0
  bestCandidateValue = Inf
  for (candidateIdx, candidate) in enumerate(this.cachedCandidates)
    value = analyze(candidate, direction)
    if value < bestCandidateValue
      bestCandidateIdx = candidateIdx
      bestCandidateValue = value
    end
  end
  parameters(this.cachedCandidates[bestCandidateIdx])
end


immutable NLoptDirectionFinder <: AlignedDirectionFinder
end

function leastAlignedDirection{T <: ParameterizedTransform}(this:: NLoptDirectionFinder, direction:: VectorizedImage, space:: ParameterSpace{T}, initialDirection:: Vector{Float64})
  opt = Opt(:LD_SLSQP, dimension(space))
  # Could make these parameters.
  println("Starting from $(initialDirection)")
  ftol_abs!(opt, 1e-6)
  xtol_rel!(opt, 0.0)
  maxeval!(opt, 400)
  lower_bounds!(opt, Float64[bounds(space, i)[1] for i in 1:dimension(space)])
  upper_bounds!(opt, Float64[bounds(space, i)[2] for i in 1:dimension(space)])
  
  # NLopt requires a function that evaluates the objective (which is just
  # <makeTransform(point), direction>) and computes its gradient with
  # respect to the point parameter at point, placing it in gradientOutput.
  function inprodAndGradient!(point:: Vector{Float64}, gradientOutput:: Vector{Float64})
    #FIXME: Avoid allocation here using the streaming version of analyze(),
    # which is already implemented.  But actually it seems that the bottleneck
    # is in the cubature for computing the filter values, not in memory
    # allocation.
    const t:: T = makeTransform(space, point)
    inprod = analyze(t, direction)
    # println("Point: $(point)")
    if length(gradientOutput) > 0
      fill!(gradientOutput, 0.0)
      parameterGradient!(t, direction, gradientOutput)
      # println("Gradient: $(gradientOutput) (norm: $(norm(gradientOutput)))")
    end
    # println("Value: $(inprod)")
    return inprod:: Float64
  end
  
  min_objective!(opt, inprodAndGradient!)
  (minValue, bestDirection, terminationStatus) = optimize(opt, initialDirection)
  return bestDirection
end