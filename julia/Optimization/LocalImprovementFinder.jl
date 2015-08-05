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
function findLocalImprovements{T <: ParameterizedTransform}(this:: LocalImprovementFinder, currentAtoms:: Vector{TransformAtom}, image:: VectorizedImage, space:: ParameterSpace{T})
  raiseAbstract("findLocalImprovements", this)
end


# Does no local improvement.  For use as a baseline.
immutable NullImprovementFinder <: LocalImprovementFinder end

function findLocalImprovements{T <: ParameterizedTransform}(this:: NullImprovementFinder, currentAtoms:: Vector{TransformAtom}, image:: VectorizedImage, space:: ParameterSpace{T})
  currentAtoms
end


# Just follows the gradient of the loss,
#   (t1, ..., td) |-> l(\sum_{i=1}^{d} wi \phi(ti) - image) ,
# for awhile.
immutable FirstOrderImprovementFinder <: LocalImprovementFinder
  stepSize:: Float64
  numSteps:: Int64
end

function findLocalImprovements{T <: ParameterizedTransform}(this:: FirstOrderImprovementFinder, currentAtoms:: Vector{TransformAtom}, image:: VectorizedImage, space:: ParameterSpace{T})
  newParameters = zeros(dimension(space), length(currentAtoms))
  for step in 1:this.numSteps
    for (i, a) in enumerate(currentAtoms)
      parameterGradient!(a.transform, image, sub(newParameters, :, i))
      scale!(-this.stepSize, sub(newParameters, :,i))
      Base.LinAlg.axpy!(1.0, parameters(a.transform), sub(newParameters, :,i))
    end
  end
  #FIXME: May want to consider modifying atoms in place.
  [TransformAtom(sub(newParameters, :, i), a.weight) for (i, a) in enumerate(currentAtoms)]
end


immutable NLoptImprovementFinder <: LocalImprovementFinder
end

function findLocalImprovements{T <: ParameterizedTransform}(this:: NLoptImprovementFinder, currentAtoms:: Vector{TransformAtom}, image:: VectorizedImage, space:: ParameterSpace{T})
  #FIXME
end