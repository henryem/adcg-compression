export LassoSolver, solveAll

# Solves a Lasso optimization problem:
#   \min_{t \in \R^{d \times n}} .5||X t - Y||_2^2 + \lambda ||t||_1 ,
# where X \in \R^{m \times d}, Y \in \R^{m \times n}, and ||t||_1 means the
# l1 norm of vec(t).
# There are existing Lasso packages that are faster in general.  However, we 
# would like to solve many problems with the same X and different y, and
# packages like glmnet are (apparently) not optimized for this.  This solver
# assumes that we have precomputed something that allows us to compute
# (X^T X + \gamma I)^{-1} b quickly for some fixed \gamma (which determines
# the step size of the algorithm: smaller \gamma gives larger steps) and for 
# _any_ b.
# We use ADMM.  (See, for example, https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf .)
immutable LassoSolver end

const TOLERANCE = 1e-5

function solveAll(this:: LassoSolver, XTXInv, X:: Matrix{Float64}, Y:: Matrix{Float64}, lambda:: Float64)
  A = XTXInv \ Y
  B = A
  resid = zeros(a)
  while norm(residual) < 1e-5
    #TODO: Should figure out a way to do this solve and subtract in place.
    A[:,:] = XTXInv \ (B - resid)
    softShrinkage!(A, resid, B)
    resid[:,:] = resid + A - B
  end
end

# Soft shrinkage on (input + resid), placed in output.
function softShrinkage!(input:: Matrix{Float64}, resid:: Matrix{Float64}, output:: Matrix{Float64}, lambda:: Float64)
  assert(size(input) == size(output) == size(resid))
  for i in 1:length(input)
    @inbounds current = input[i] + resid[i]
    if current > lambda
      current -= lambda
    elseif current < -lambda
      current += lambda
    else
      current = 0
    end
    @inbounds output[i] = current
  end
end