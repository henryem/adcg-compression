export LocalImprovementFinder, findLocalImprovements

export NullImprovementFinder
export FirstOrderImprovementFinder
export NLoptImprovementFinder

abstract LocalImprovementFinder

# Find new parameters (p_i)_{i=1}^{length(currentAtoms)} that reduce the 
# squared distance between the synthesized image
#   sum_i(synthesize(T(p_i),w_i))
# and @image.  (The weights w_i are fixed.)  Note that this problem is
# nonconvex and potentially high-dimensional, so we cannot actually solve it;
# we only try to ensure improvement of the objective.  (Even if we could solve
# the problem, it wouldn't be enough, because we also need to optimize the 
# weights anyway.)
function findLocalImprovements{T <: ParameterizedTransform}(this:: LocalImprovementFinder, currentAtoms:: Vector{TransformAtom}, tau:: Float64, image:: VectorizedImage, p:: ImageParameters, loss:: Loss, space:: ParameterSpace{T})
  raiseAbstract("findLocalImprovements", this)
end


# Does no local improvement.  For use as a baseline.
immutable NullImprovementFinder <: LocalImprovementFinder end

function findLocalImprovements{T <: ParameterizedTransform}(this:: NullImprovementFinder, currentAtoms:: Vector{TransformAtom}, tau:: Float64, image:: VectorizedImage, p:: ImageParameters, loss:: Loss, space:: ParameterSpace{T})
  currentAtoms
end


# Just follows the gradient of the loss,
#   (t1, ..., td) |-> l(\sum_{i=1}^{d} wi \phi(ti) - image) ,
# for awhile.
immutable FirstOrderImprovementFinder <: LocalImprovementFinder
  stepSize:: Float64
  numSteps:: Int64
end

function findLocalImprovements{T <: ParameterizedTransform}(this:: FirstOrderImprovementFinder, currentAtoms:: Vector{TransformAtom}, tau:: Float64, image:: VectorizedImage, p:: ImageParameters, loss:: Loss, space:: ParameterSpace{T})
  const k = length(currentAtoms)
  newAtoms = [currentAtoms]
  const n = length(image)
  residual = zeros(Float64, n)
  currentLossGradient = zeros(Float64, n)
  
  newParameters = zeros(dimension(space), length(currentAtoms))
  for step in 1:this.numSteps
    # Recompute the loss gradient.  Then the gradient of the loss with respect
    # to a parameter is just the gradient of
    #   <currentLossGradient, analyze(makeTransform(params))>
    # with respect to params, which is exactly what parameterGradient!
    # computes for us.
    computeResidualAndLossGradient!(p, image, newAtoms, loss, residual, currentLossGradient)
    for (i, a) in enumerate(currentAtoms)
      parameterGradient!(a.transform, currentLossGradient, sub(newParameters, :, i))
      scale!(-this.stepSize*a.weight, sub(newParameters, :, i))
      Base.LinAlg.axpy!(1.0, parameters(a.transform), sub(newParameters, :, i))
    end
    #TODO: May want to modify atoms in place.  For now this is probably not a
    # bottleneck.
    newAtoms = [TransformAtom(sub(newParameters, :, i), a.weight) for (i, a) in enumerate(newAtoms)]
  end
  newAtoms
end


immutable NLoptImprovementFinder <: LocalImprovementFinder
end

function findLocalImprovements{T <: ParameterizedTransform}(this:: NLoptImprovementFinder, currentAtoms:: Vector{TransformAtom}, tau:: Float64, image:: VectorizedImage, p:: ImageParameters, loss:: Loss, space:: ParameterSpace{T})
  const k = length(currentAtoms)
  const d = dimension(space)
  const n = length(image)
  const D = k*d
  const weights:: Vector{Float64} = [a.weight for a in currentAtoms]
  const initialParameters:: Matrix{Float64} = hcat([parameters(a.transform) for a in currentAtoms]...)
  const initialParametersVec:: Vector{Float64} = vec(initialParameters)
  currentEncodedImage = zeros(Float64, n)
  currentResidual = zeros(Float64, n)
  currentLossGradient = zeros(Float64, n)
  
  opt = Opt(:LD_SLSQP, D)
  # Could make these parameters.
  ftol_abs!(opt, 1e-6)
  xtol_rel!(opt, 0.0)
  maxeval!(opt, 400)
  lower_bounds!(opt, vec(repmat(Float64[bounds(space, i)[1] for i in 1:d], k)))
  upper_bounds!(opt, vec(repmat(Float64[bounds(space, i)[2] for i in 1:d], k)))
  
  # NLopt requires a function that evaluates the objective and computes its 
  # gradient with respect to the point parameter at point, placing it in 
  # gradientOutput.
  function inprodAndGradient!(point:: Vector{Float64}, gradientOutput:: Vector{Float64})
    assert(length(point) == D)
    # View as a (# params)-by-(# atoms) matrix for more convenient indexing.
    const parameters = reshape(point, d, k)
    
    #FIXME: Could avoid allocation here.  Check whether this is expensive.
    const atoms = [TransformAtom(makeTransform(space, sub(parameters, :, i)), w) for (i, w) in enumerate(weights)]
    # Recompute the loss gradient.  Then the gradient of the loss with respect
    # to a parameter is just the gradient of
    #   <currentLossGradient, analyze(makeTransform(params))>
    # with respect to params, which is exactly what parameterGradient!
    # computes for us.
    #NOTE: The loss gradient is computed even when we only need to know the
    # residual.  Shouldn't be a huge deal.
    computeResidualAndLossGradient!(p, image, atoms, l, currentEncodedImage, currentResidual, currentLossGradient)
    if length(gradientOutput) > 0
      # View as a (# params)-by-(# atoms) matrix for more convenient indexing.
      const gradients = reshape(gradientOutput, d, k)
      fill!(gradients, 0.0)
      for (atomIdx, a) in enumerate(atoms)
        parameterGradient!(makeTransform(space, sub(parameters, :, atomIdx)), currentLossGradient, sub(gradients, :, atomIdx))
        scale!(a.weight, sub(gradients, :, atomIdx))
      end
    end
    loss(l, currentResidual)
  end
  
  min_objective!(opt, inprodAndGradient!)
  (minValue, improvedParametersVec, terminationStatus) = optimize(opt, initialParametersVec)
  const improvedParameters = reshape(improvedParametersVec, d, k)
  return TransformAtom[TransformAtom(makeTransform(space, sub(improvedParameters, :, i)), w) for (i, w) in enumerate(weights)]
end



# Similar to NLoptImprovementFinder, but optimizes over weights and parameters
# simultaneously.  Since weights are constrained, we don't actually use the
# optimized weights in the solution; we just allow the solver to imagine that
# weights can be changed.  The point is that some local movements may only make
# sense if they are accompanied by weight changes.
immutable NLoptWeightedImprovementFinder <: LocalImprovementFinder
end

function findLocalImprovements{T <: ParameterizedTransform}(this:: NLoptImprovementFinder, currentAtoms:: Vector{TransformAtom}, tau:: Float64, image:: VectorizedImage, p:: ImageParameters, l:: Loss, space:: ParameterSpace{T})
  const k = length(currentAtoms)
  const d = dimension(space)+1
  const n = length(image)
  const D = k*d
  # Each column of initialParameters (and other parameter matrices) has first
  # the weight of that atom, then the dimension(space) parameters.
  const initialParameters:: Matrix{Float64} = hcat([vcat(a.weight, parameters(a.transform)) for a in currentAtoms]...)
  const initialParametersVec:: Vector{Float64} = vec(initialParameters)
  currentEncodedImage = zeros(Float64, n)
  currentResidual = zeros(Float64, n)
  currentLossGradient = zeros(Float64, n)
  
  opt = Opt(:LD_SLSQP, D)
  # Could make these parameters.
  ftol_abs!(opt, 1e-6)
  xtol_rel!(opt, 0.0)
  maxeval!(opt, 400)
  lower_bounds!(opt, vec(repmat(vcat(0, Float64[bounds(space, i)[1] for i in 1:dimension(space)]), k)))
  upper_bounds!(opt, vec(repmat(vcat(tau, Float64[bounds(space, i)[2] for i in 1:dimension(space)]), k)))
  
  # NLopt requires a function that evaluates the objective and computes its 
  # gradient with respect to the point parameter at point, placing it in 
  # gradientOutput.
  function inprodAndGradient!(point:: Vector{Float64}, gradientOutput:: Vector{Float64})
    assert(length(point) == D)
    # View as a (# params)-by-(# atoms) matrix for more convenient indexing.
    const parameters = reshape(point, d, k)
    
    #FIXME: Could avoid allocation here.  Check whether this is expensive.
    const atoms = [TransformAtom(makeTransform(space, sub(parameters, 2:d, i)), parameters[1,i]) for i in 1:k]
    # Recompute the loss gradient.  Then the gradient of the loss with respect
    # to a parameter is just the gradient of
    #   <currentLossGradient, analyze(makeTransform(params))>
    # with respect to params, which is exactly what parameterGradient!
    # computes for us.
    #NOTE: The loss gradient is computed even when we only need to know the
    # residual.  Shouldn't be a huge deal.
    computeResidualAndLossGradient!(p, image, atoms, l, currentEncodedImage, currentResidual, currentLossGradient)
    if length(gradientOutput) > 0
      # View as a (# params)-by-(# atoms) matrix for more convenient indexing.
      const gradients = reshape(gradientOutput, d, k)
      fill!(gradients, 0.0)
      for (atomIdx, a) in enumerate(atoms)
        parameterGradient!(makeTransform(space, sub(parameters, 2:d, atomIdx)), currentLossGradient, sub(gradients, 2:d, atomIdx))
        scale!(a.weight, sub(gradients, 2:d, atomIdx))
        #FIXME: Compute gradient of weight.  I think that's just makeX(atoms)^T * currentLossGradient.
      end
    end
    loss(l, currentResidual)
  end
  
  min_objective!(opt, inprodAndGradient!)
  (minValue, improvedParametersVec, terminationStatus) = optimize(opt, initialParametersVec)
  const improvedParameters = reshape(improvedParametersVec, d, k)
  return TransformAtom[TransformAtom(makeTransform(space, sub(improvedParameters, 2:d, i)), a.weight) for (i, a) in enumerate(currentAtoms)]
end