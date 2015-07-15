export LassoSolver, solveAll

# Solves a Lasso optimization problem:
#   \min_{T \in \R^{d \times n}} .5||X T - Y||_F^2 + \lambda ||T||_1 ,
# where X \in \R^{m \times d}, Y \in \R^{m \times n}, and ||T||_1 means the
# l1 norm of vec(T).
# There are existing Lasso packages that are faster in general.  However, we 
# would like to solve many problems with the same X and different y, and
# packages like glmnet are (apparently) not optimized for this.  This solver
# assumes that we have precomputed something that allows us to compute
# (X^T X + \gamma I)^{-1} b quickly for some fixed \gamma (which determines
# the step size of the algorithm: smaller \gamma gives larger steps) and for 
# _any_ b.
# We use ADMM.  See, for example:
#   https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
# The iteration is:
#   1. A <- prox_{T \mapsto .5||XT-Y||_F^2}(B - residual)
#   2. B <- prox_{T \mapsto ||T||_1}(A + residual)
#   3. residual <- residual + A - B
# Step 2 is simple soft shrinkage (with parameter lambda/gamma), and step 3 is 
# trivial.
# Step 1 involves solving a regularized least squares problem with the origin at
# (B - residual), which we can massage to an ordinary ridge regression problem
# by completing the square and ignoring constant terms:
#   T' = \argmin_T ||XT-Y||_F^2 + \gamma ||T - (B - residual)||_F^2
#      = \argmin_T ||XT||_F^2 - 2<X^T Y, T> + ||Y||_F^2 + \gamma ||T||_F^2 - 2\gamma<B - residual, T> + \gamma||B - residual||_F^2
#      = \argmin_T ||XT||_F^2 - 2<X^T Y + \gamma(B - residual), T> + \gamma||T||_F^2
# First-order conditions give us the solution:
#    (X^T X + \gamma I)T' - X^T Y - \gamma(B - residual)
#    => T' = (X^T X + \gamma I) \ (X^T Y + \gamma(B - residual))
# Note that this means we must compute X^T Y, which is slightly unfortunate.
immutable LassoSolver end

const TOLERANCE = 1e-5

function solveAll(this:: LassoSolver, XTXInv, XTY:: Matrix{Float64}, lambda:: Float64, gamma:: Float64)
  A = XTXInv \ XTY
  B = copy(A)
  residual = zeros(A)
  while norm(residual) < TOLERANCE
    #TODO: Should figure out a way to do this solve and subtract in place.
    A[:,:] = XTXInv \ (XTY + gamma*(B - residual))
    softShrinkage!(A, residual, B, lambda/gamma)
    residual[:,:] = residual + A - B
  end
  #NOTE: It is important to return B, since A is not necessarily sparse.
  B
end

# Soft shrinkage on (input + resid), placed in output.
function softShrinkage!(input:: Matrix{Float64}, resid:: Matrix{Float64}, output:: Matrix{Float64}, lambda:: Float64)
  assert(size(input) == size(output) == size(resid))
  for i in 1:length(input)
    #FIXME: @inbounds
    current = input[i] + resid[i]
    if current > lambda
      current -= lambda
    elseif current < -lambda
      current += lambda
    else
      current = 0
    end
    #FIXME: @inbounds
    output[i] = current
  end
end