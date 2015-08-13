export FiniteDimConvexSolver, bestWeights
export FixedStep


# Finds a convex combination of a fixed set of atoms that best represents
# a given image.  For a squared loss, this is a finite-dimensional convex 
# quadratic program with linear constraints, specifically a Lasso problem.
# 
# The actual problem solved is:
#   min_x l(Tx - y) s.t. x >= 0, <x, 1> <= tau ,
# where x is our weight vector to be found, T is the synthesis matrix for our 
# transforms (i.e. for a weight vector x, Tx gives a synthesized image), y
# is the image to be matched, and the constraints ensure that the weights are
# nonnegative and sum to at most tau.
# 
# Some solvers may solve only an approximate form of this, e.g. ignoring the 
# nonnegativity constraints and imposing an l1 penalty on x instead of the
# constraint <x, 1> <= tau.
abstract FiniteDimConvexSolver

# Finds (approximately) nonnegative weights summing to at most maxWeight such
# that the squared distance between the synthesized image
#   sum_i(synthesize(T(p_i),w_i))
# and @image is (approximately) minimized.  (The parameters p_i are fixed.)
function bestWeights(this:: FiniteDimConvexSolver, currentAtoms:: Vector{TransformAtom}, maxWeight:: Float64, image:: VectorizedImage, imageParams:: ImageParameters, loss:: Loss)
  raiseAbstract("findBestWeights", this)
end


# Not a real fully-corrective step -- just takes a convex combination of the
# previous atoms (which are assumed to be the first n-1 atoms given) and
# the new atom (which is assumed to be the last atom given).  Probably
# stupid.  For use as a baseline.
immutable FixedStep <: FiniteDimConvexSolver
  gamma:: Float64
end

function bestWeights(this:: FixedStep, currentAtoms:: Vector{TransformAtom}, maxWeight:: Float64, image:: VectorizedImage, imageParams:: ImageParameters, loss:: Loss)
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