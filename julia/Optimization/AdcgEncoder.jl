export AdcgEncoder, encode

#FIXME
using Images

immutable AdcgEncoder{T <: ParameterizedTransform} <: Encoder
  im:: ImageParameters
  motherTransformSpace:: ParameterSpace{T}
  loss:: Loss
  bestAtomFinder:: AlignedDirectionFinder
  bestWeightsFinder:: FiniteDimQpSolver
  localAtomImprover:: LocalImprovementFinder
end

function encode{T}(this:: AdcgEncoder{T}, image:: VectorizedImage)
  const d = pixelCount(this.im)
  const tau = norm(image, 2) #FIXME: This seems roughly right, since atoms are 
  # supposed to have norm roughly 1.  If we measured everything in the 1-norm,
  # and we believed overshooting the total pixel weight wouldn't ever be a
  # good idiea, we could use exactly norm(image, 1) to set tau.  Since we
  # measure in the 2-norm, I'm not sure we can say anything.  In any case
  # this is probably okay, except that we might want to divide by \sqrt{d}...
  const space:: ParameterSpace{T} = this.motherTransformSpace
  atoms:: Vector{TransformAtom} = []
  currentImageEncoded = TransformedImage(this.im, atoms)
  currentImageDecoded = zeros(Float64, d)
  residual = zeros(Float64, d)
  currentLossGradient = zeros(Float64, d)
  iterationCount = 0
  while true
    # Find the atom that (approximately) best explains the current residual.
    # This means searching over the parameter space, a non-convex but low-
    # dimensional and smooth problem.  Note that what we are really doing
    # here is computing the gradient of the squared loss with respect to the
    # measure.  Such an object can always be an atomic measure, where the atom
    # is on a parameter vector whose synthesis gives the best infinitessimal 
    # improvement in the least-squares objective.  To get the best improvement,
    # we want the synthesis to have a small angle with the current residual.
    # 
    # Note that this step changes if we use a different loss function; in 
    # general we use the gradient of the loss (with respect to the residual),
    # which coincides with the residual for the squared loss.
    #TODO: Not sure why this next line is needed.
    currentImageEncoded = TransformedImage(this.im, atoms)
    decodeInto!(currentImageEncoded, currentImageDecoded)
    imwrite(toImage(this.im, currentImageDecoded), "inprogress_$(iterationCount).jpg")
    #FIXME: This will allocate memory.
    residual[:] = image - currentImageDecoded
    imwrite(toImage(this.im, abs(residual)), "inprogress_residual_$(iterationCount).jpg")
    residualNorm = norm(residual, 2)
    iterationCount += 1
    println("Residual norm before iteration $(iterationCount): $(residualNorm)")
    if iterationCount > 3
      println("Finished encoding: Ran to max number of iterations.")
      break
    elseif residualNorm < 1e-5
      println("Finished encoding: Residual is small.")
      break
    end
    lossGradient!(this.loss, residual, currentLossGradient)
    #FIXME: Check if the direction is actually positive!  If the gradient is
    # ~0 then we are done.  Also, should compute the Frank-Wolfe lower bound
    # here.
    const nextAtomParameters = mostAlignedDirection(this.bestAtomFinder, currentLossGradient, space)
    if analyze(makeTransform(space, nextAtomParameters), residual) <= 0
      println("Finished encoding: Cannot find an atom to improve the residual.")
      break
    end
    push!(atoms, TransformAtom(makeTransform(space, nextAtomParameters), 0.0))
    
    # Now we do heuristic descent over the weights and parameters for awhile.
    # We alternate between exact block coordinate descent over the weights,
    # an atom pruning step, and local search over the parameter values.
    # Note that the standard Frank-Wolfe algorithm would only take a convex
    # combination of the current measure and the new atom, and these steps
    # (since they include exact minimization over the weights) always give us
    # a better objective value.  There are no proofs about the efficacy of
    # these steps (except to the effect that they harm nothing), but in
    # practice they are often important.
    while true
      # Now optimize over the convex hull of tau*thetas to produce atoms whose
      # mass sums to (at most) tau.  This is just solving a convex QP (a 
      # modified Lasso problem).
      #FIXME: Probably should act in place, but TransformAtom is immutable (and
      # therefore not really appropriate for use in this solver).  Same for the
      # next two steps.  On the other hand, these steps can be kinda expensive,
      # so maybe it doesn't matter.
      #FIXME: Assumes squared error loss.
      atoms = bestWeights(this.bestWeightsFinder, atoms, tau, image, this.im)
    
      # Now remove any atoms with 0 weight.
      atoms = filter(a -> a.weight > 0.0, atoms)
    
      # Now do local heuristic search to improve atom locations.
      #FIXME: Assumes squared error loss.
      atoms = findLocalImprovements(this.localAtomImprover, atoms, image, space)
      
      #FIXME: Need a real stopping condition.
      break
    end
    #FIXME: Need a real stopping condition.
    println("Added atom at $(nextAtomParameters) with weight $(atoms[end].weight).")
  end
  TransformedImage(this.im, atoms)
end