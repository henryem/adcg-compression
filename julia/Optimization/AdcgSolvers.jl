# Solver subroutines for AdcgEncoder.
export Loss, loss, gradient
export AlignedDirectionFinder, mostAlignedDirection
export FiniteDimQpSolver, bestWeights
export LocalImprovementFinder, findLocalImprovements
export L2Loss
export WarmstartFinder
export GridFinder
export NLoptDirectionFinder
export AdmmLasso
export InteriorPointNonnegativeLasso
export FixedStep
export NullImprovementFinder
export GradientStepImprovementFinder
export NLoptImprovementFinder

using NLopt
using ImageUtils


abstract Loss

function loss(this:: Loss, residual:: Vector{Float64})
  raiseAbstract("loss", this)
end

function gradient(this:: Loss, residual:: Vector{Float64})
  out = zeros(residual)
  gradient!(this, residual, out)
  out
end

function gradient!(this:: Loss, residual:: Vector{Float64}, out:: Vector{Float64})
  raiseAbstract("gradient!", this)
end

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

abstract FiniteDimQpSolver

# Finds (approximately) nonnegative weights summing to at most maxWeight such
# that the squared distance between the synthesized image
#   sum_i(synthesize(T(p_i),w_i))
# and @image is (approximately) minimized.  (The parameters p_i are fixed.)
function bestWeights(this:: FiniteDimQpSolver, currentAtoms:: Vector{TransformAtom}, maxWeight:: Float64, image:: VectorizedImage, imageParams:: ImageParameters)
  raiseAbstract("findBestWeights", this)
end

abstract LocalImprovementFinder

# Find new parameters (p_i)_{i=1}^{length(currentAtoms)} that reduce the 
# squared distance between the synthesized image
#   sum_i(synthesize(T(p_i),w_i))
# and @image.  (The weights w_i are fixed.)  Note that this problem is
# nonconvex and potentially high-dimensional, so we cannot actually solve it;
# we only try to ensure improvement of the objective.  (Even if we could solve
# the problem, it wouldn't be enough, because we also need to optimize the 
# weights anyway.)
function findLocalImprovements{T <: ParameterizedTransform}(this:: LocalImprovementFinder, currentAtoms:: Vector{TransformAtom}, image:: VectorizedImage, space:: ParameterSpace{T})
  raiseAbstract("findLocalImprovements", this)
end


immutable L2Loss <: Loss end

function loss(this:: L2Loss, residual:: Vector{Float64})
  .5*norm(residual, 2)^2
end

function gradient!(this:: Loss, residual:: Vector{Float64}, out:: Vector{Float64})
  out[:] = residual
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
  opt = Opt(:LD_MMA, dimension(space))
  # Could make these parameters.
  ftol_abs!(opt, 1e-6)
  xtol_rel!(opt, 0.0)
  maxeval!(opt, 200)
  lower_bounds!(opt, Float64[bounds(space, i)[1] for i in 1:dimension(space)])
  upper_bounds!(opt, Float64[bounds(space, i)[2] for i in 1:dimension(space)])
  
  # NLopt requires a function that evaluates the objective (which is just
  # <makeTransform(point), direction>) and computes its gradient with
  # respect to the point parameter at point, placing it in gradientOutput.
  function inprodAndGradient!(point:: Vector{Float64}, gradientOutput:: Vector{Float64})
    #FIXME: Really really should avoid allocation here.  Should have a mutating
    # version of makeTransform and a streaming version of analyze.
    const t:: T = makeTransform(space, point)
    inprod = analyze(t, direction)
    #FIXME: May need to multiply by -1 or something here?
    parameterGradient!(t, direction, gradientOutput)
    return inprod:: Float64
  end
  
  max_objective!(opt, inprodAndGradient!)
  (maxValue, bestDirection, terminationStatus) = optimize(opt, initialDirection)
  return bestDirection
end


# Not a real fully-corrective step -- just takes a convex combination of the
# previous atoms (which are assumed to be the first n-1 atoms given) and
# the new atom (which is assumed to be the last atom given).  Probably
# stupid.  For use as a baseline.
immutable FixedStep <: FiniteDimQpSolver
  gamma:: Float64
end

function bestWeights(this:: FixedStep, currentAtoms:: Vector{TransformAtom}, maxWeight:: Float64, image:: VectorizedImage, imageParams:: ImageParameters)
  const n = length(currentAtoms)
  const newAtoms:: Vector{TransformAtom} = if n == 1
    [TransformAtom(currentAtoms[1].transform, maxWeight)]
  else
    map(1:n) do atomIdx
      const a:: TransformAtom = currentAtoms[atomIdx]
      if atomIdx != n
        TransformAtom(a.transform, (1.0-this.gamma)*a.weight)
      else
        TransformAtom(a.transform, this.gamma*maxWeight)
      end
    end
  end
  newAtoms
end

immutable AdmmLasso <: FiniteDimQpSolver end

function bestWeights(this:: AdmmLasso, currentAtoms:: Vector{TransformAtom}, maxWeight:: Float64, image:: VectorizedImage, imageParams:: ImageParameters)
  #FIXME: This is a crazy and slow way to solve this problem.  Also, it allows
  # negative atoms and uses only a soft l1 penalty.  Both are probably okay for 
  # images, but it's a little weird.
  const solver = AdmmLassoSolver()
  const regularizedXTX = makeRegularizedXTX(imageParams, [t.transform for t in currentAtoms], LASSO_INVERSE_STEPSIZE)
  # solve(solver, )
  #FIXME
end

immutable InteriorPointNonnegativeLasso <: FiniteDimQpSolver end

function bestWeights(this:: InteriorPointNonnegativeLasso, currentAtoms:: Vector{TransformAtom}, maxWeight:: Float64, image:: VectorizedImage, imageParams:: ImageParameters)
  #FIXME
end


# Does no local improvement.  For use as a baseline.
immutable NullImprovementFinder <: LocalImprovementFinder end

function findLocalImprovements{T <: ParameterizedTransform}(this:: NullImprovementFinder, currentAtoms:: Vector{TransformAtom}, image:: VectorizedImage, space:: ParameterSpace{T})
  currentAtoms
end

immutable GradientStepImprovementFinder <: LocalImprovementFinder
  stepSize:: Float64
end

function findLocalImprovements{T <: ParameterizedTransform}(this:: GradientStepImprovementFinder, currentAtoms:: Vector{TransformAtom}, image:: VectorizedImage, space:: ParameterSpace{T})
  newParameters = zeros(dimension(space), length(currentAtoms))
  for (i, a) in enumerate(currentAtoms)
    parameterGradient!(a.transform, image, sub(newParameters, :, i))
    newParameters[:,i] .*= -this.stepSize
    newParameters[:,i] += parameters(a.transform)
  end
  #FIXME: May want to consider modifying atoms in place.
  [TransformAtom(sub(newParameters, :, i), a.weight) for (i, a) in enumerate(currentAtoms)]
end

immutable NLoptImprovementFinder <: LocalImprovementFinder
end

function findLocalImprovements{T <: ParameterizedTransform}(this:: NLoptImprovementFinder, currentAtoms:: Vector{TransformAtom}, image:: VectorizedImage, space:: ParameterSpace{T})
  #FIXME
end