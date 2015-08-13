export AdcgEncoder, encode

using Images

immutable AdcgEncoder{T <: ParameterizedTransform} <: Encoder
  im:: ImageParameters
  motherTransformSpace:: ParameterSpace{T}
  loss:: Loss
  bestAtomFinder:: AlignedDirectionFinder
  bestWeightsFinder:: FiniteDimConvexSolver
  localAtomImprover:: LocalImprovementFinder
end

function encode{T}(this:: AdcgEncoder{T}, image:: VectorizedImage)
  const space = this.motherTransformSpace
  const d = pixelCount(imageParameters(space))
  # Here we are using the conservative bound:
  # 
  # \sum_i w_i = \sum_i w_i ||a_i||_2
  #   <= \sum_i w_i \sqrt{d} ||a_i||_1
  #   = \sqrt{d} ||image||_1
  # 
  # Since we won't get exact reconstruction, even this bound is not always
  # accurate; we might want to fudge it upward a bit.
  const tau = sqrt(d)*norm(image, 1)
  atoms:: Vector{TransformAtom} = []
  currentEncodedImage = zeros(Float64, d)
  residual = zeros(Float64, d)
  currentLossGradient = zeros(Float64, d)
  iterationCount = 0
  #FIXME
  const imageName = "image$(rand(1:2^20))"
  imwrite(toImage(this.im, image), "$(imageName)-truth.jpg")
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
    computeResidualAndLossGradient!(imageParameters(space), image, atoms, this.loss, currentEncodedImage, residual, currentLossGradient)
    const currentLoss = loss(this.loss, residual)
    iterationCount += 1
    println("Atoms before iteration $(iterationCount): $(atoms)")
    #FIXME: Why doesn't this work on the second iteration?
    # println("Residual: $(residual)")
    imwrite(toImage(this.im, max(residual, 0)), "$(imageName)-$(iterationCount)-positive-residual.jpg")
    imwrite(toImage(this.im, abs(min(residual, 0))), "$(imageName)-$(iterationCount)-negative-residual.jpg")
    println("Loss before iteration $(iterationCount): $(currentLoss)")
    #FIXME
    if iterationCount > 20
      println("Finished encoding: Ran to max number of iterations.")
      break
    elseif currentLoss < 1e-5
      println("Finished encoding: Residual is small.")
      break
    end
    #FIXME: Should compute the Frank-Wolfe lower bound here.
    println("Finding the gradient direction...")
    @time const nextAtomParameters:: Vector{Float64} = leastAlignedDirection(this.bestAtomFinder, currentLossGradient, space)
    if analyze(makeTransform(space, nextAtomParameters), currentLossGradient) >= 0
      println("Finished encoding: Cannot find an atom to improve the residual.")
      break
    end
    push!(atoms, TransformAtom(makeTransform(space, nextAtomParameters), 0.0))
    println("Added atom at $(nextAtomParameters).")
    writePicture!([TransformAtom(makeTransform(space, nextAtomParameters), 10.0)], imageParameters(space), "$(imageName)-$(iterationCount)-direction")
    
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
      # mass sums to (at most) tau.  This means minimizing the loss over a 
      # simplex; for the l2 loss this is a nonnegative Lasso problem.
      println("Finding optimal weights...")
      @time atoms = bestWeights(this.bestWeightsFinder, atoms, tau, image, imageParameters(space), this.loss)
      writePicture!(atoms, imageParameters(space), "$(imageName)-$(iterationCount)-weights")
      println("Atoms with optimal weights: $(atoms)")
    
      # Now remove any atoms with 0 weight.
      atoms = filter(a -> a.weight > 0.0, atoms)
    
      # Now do local heuristic search to improve atom locations.
      println("Finding local improvements...")
      @time atoms = findLocalImprovements(this.localAtomImprover, atoms, tau, image, imageParameters(space), this.loss, space)
      writePicture!(atoms, imageParameters(space), "$(imageName)-$(iterationCount)-local")
      println("Atoms with local improvements: $(atoms)")
      
      #FIXME: Need a real stopping condition.
      break
    end
    #FIXME: Need a real stopping condition.
  end
  TransformedImage(imageParameters(space), atoms)
end