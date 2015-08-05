export AlignedDirectionFinder, mostAlignedDirection
export WarmstartFinder
export GridFinder
export NLoptDirectionFinder

using NLopt

abstract AlignedDirectionFinder

# Finds a parameter vector p in @space so that <synthesize(T(p),1.0), direction>
# is (approximately) maximized.
function mostAlignedDirection{T <: ParameterizedTransform}(this:: AlignedDirectionFinder, direction:: VectorizedImage, space:: ParameterSpace{T})
  mostAlignedDirection(this, direction, space, zeros(dimension(space)))
end

# As mostAlignedDirection, but with a starting location.
function mostAlignedDirection{T <: ParameterizedTransform}(this:: AlignedDirectionFinder, direction:: VectorizedImage, space:: ParameterSpace{T}, start:: Vector{Float64})
  raiseAbstract("mostAlignedDirection", this)
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

function mostAlignedDirection{T <: ParameterizedTransform}(this:: WarmstartFinder, direction:: VectorizedImage, space:: ParameterSpace{T}, ignored:: Vector{Float64})
  const start = mostAlignedDirection(this.first, direction, space)
  mostAlignedDirection(this.next, direction, space, start)
end


immutable GridFinder <: AlignedDirectionFinder
  gridSize:: Int64
end

function mostAlignedDirection{T <: ParameterizedTransform}(this:: GridFinder, direction:: VectorizedImage, space:: ParameterSpace{T})
  candidates = grid(space, this.gridSize)
  bestCandidateIdx = 0
  bestCandidateValue = -Inf
  for (candidateIdx, candidate) in enumerate(candidates)
    #TODO: May want to implement an analyzeWithStorage() or analyzeStreaming()
    # so that the filters don't have to be separately allocated and then 
    # immediately GCed after they are used for one dot product.  Similarly, we
    # may want to avoid allocating even the parameter vector for each
    # candidate; we'd be okay if grid() returned an iterator (with the further
    # constraint that next() destroys the current iterate) instead of a
    # materialized list of candidates.
    value = analyze(candidate, direction)
    if value > bestCandidateValue
      bestCandidateIdx = candidateIdx
      bestCandidateValue = value
    end
  end
  parameters(candidates[bestCandidateIdx])
end


immutable NLoptDirectionFinder <: AlignedDirectionFinder
end

function mostAlignedDirection{T <: ParameterizedTransform}(this:: NLoptDirectionFinder, direction:: VectorizedImage, space:: ParameterSpace{T}, initialDirection:: Vector{Float64})
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
  
  max_objective!(opt, inprodAndGradient!)
  (maxValue, bestDirection, terminationStatus) = optimize(opt, initialDirection)
  return bestDirection
end