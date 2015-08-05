export FiniteDimQpSolver, bestWeights
export AdmmLasso
export InteriorPointNonnegativeLasso
export FixedStep


using Base.LinAlg
using ArrayViews


# Finds a convex combination of a fixed set of atoms that best represents
# a given image.  For a squared loss, this is a finite-dimensional convex 
# quadratic program with linear constraints, specifically a Lasso problem.
abstract FiniteDimQpSolver

# Finds (approximately) nonnegative weights summing to at most maxWeight such
# that the squared distance between the synthesized image
#   sum_i(synthesize(T(p_i),w_i))
# and @image is (approximately) minimized.  (The parameters p_i are fixed.)
function bestWeights(this:: FiniteDimQpSolver, currentAtoms:: Vector{TransformAtom}, maxWeight:: Float64, image:: VectorizedImage, imageParams:: ImageParameters)
  raiseAbstract("findBestWeights", this)
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


# Solves a nonnegative Lasso problem using a primal-dual interior point method.
# In relatively low dimension (say, <1000 atoms), this is probably the fastest
# available way to solve the nonnegative Lasso.  If it is hard to solve a
# d-by-d linear system, prefer a first-order method instead.
# 
# The particular method is described in p. 612 of Boyd and Vandenberghe 
# (algorithm 11.2).  The search directions are described by equation 11.54.
# Specialized to a QP, they can be simplified a bit.
immutable InteriorPointNonnegativeLasso <: FiniteDimQpSolver
  mu:: Float64
  feasibilitySlack:: Float64
  objectiveSlack:: Float64
  backtrackingAlpha:: Float64
  backtrackingBeta:: Float64
  
  function InteriorPointNonnegativeLasso(mu, feasibilitySlack, objectiveSlack, backtrackingAlpha, backtrackingBeta)
    assert(mu > 1)
    assert(feasibilitySlack > 0)
    assert(objectiveSlack > 0)
    assert(backtrackingAlpha > 0)
    assert(backtrackingAlpha < 1)
    assert(backtrackingBeta > 0)
    assert(backtrackingBeta < 1)
    new(mu, feasibilitySlack, objectiveSlack, backtrackingAlpha, backtrackingBeta)
  end
end

# Default settings.
function InteriorPointNonnegativeLasso()
  InteriorPointNonnegativeLasso(10.0, 1e-8, 1e-8, 0.01, 0.5)
end

# Finds a step in direction (deltaX, deltaLambda, deltaNu) from point
# (x, lambda, nu) that satisfies several conditions, detailed below.
# It is assumed that @residual contains the current residual, before
# a step is taken.
function lineSearch{A <: AbstractVector{Float64}}(alpha:: Float64, beta:: Float64, x:: Vector{Float64}, lambda:: Vector{Float64}, nu:: Float64, deltaX:: A, deltaLambda:: A, deltaNu:: Float64, residual:: Vector{Float64})
  const n = size(x, 1)
  # We have to keep lambda all-nonnegative, so our first condition on s is
  # that this stays true.  So we upper-bound it by something just a little
  # smaller than the step size that makes some coordinate of lambda equal to
  # 0.
  const sLambdaBound = min(1.0, minimum([dl >= 0 ? 1.0 : -l / dl for (l, dl) in zip(lambda, deltaLambda)]))
  s = 0.99*sLambdaBound
  
  # Now we find a point where we have gotten a reduction of at least alpha per
  # unit of s.  We just backtrack from our maximum by powers of beta.
  const initialResidualNorm = norm(residual, 2)
  
  while true
    const deltaS = (beta-1)*s

    # Recalculate the residual efficiently, without memory allocation.  (This
    # is the true inner loop of the interior point method, so speed is most
    # critical here.)
    #residual[1:n] = C - (lambda+s*deltaLambda) + (nu+s*deltaNu)*ones(n)
    # Change = -s*deltaLambda + s*deltaNu*ones(n)
    #residual[(n+1):(2*n)] = C + (x+s*deltaX) .* (lambda+s*deltaLambda) - (1/t)*ones(n)
    # Change = (x+s*deltaX) .* (lambda+s*deltaLambda) - x .* lambda
    #        = s*deltaX .* lambda + s*deltaLambda .* x + s^2*deltaX .* deltaLambda
    #residual[2n+1] = C + <1,s*deltaX+x>
    # Change = s*<1,deltaX>
    dualResidualSquared = mapreduce(+, 1:n) do idx
      (residual[idx] - s*deltaLambda[idx] + s*deltaNu)^2
    end
    centralityResidualSquared = mapreduce(+, 1:n) do idx
      (residual[idx+n] + s*deltaX[idx]*lambda[idx] + s*x[idx]*deltaLambda[idx] + s^2*deltaX[idx]*deltaLambda[idx])^2
    end
    primalResidualSquared = (s*sum(deltaX))^2
    const residualNorm = sqrt(dualResidualSquared + centralityResidualSquared + primalResidualSquared)
    
    if residualNorm <= (1.0-alpha*s)*initialResidualNorm
      break
    else
      s *= beta
      continue
    end
  end
  
  s
end

# The vector we are trying to get to 0 in the primal-dual interior point
# method.  See comments in bestWeights.  The result is placed in @out, a
# (2n+1)-vector.
function kktResidual!(TtT:: Matrix{Float64}, Ttu:: Vector{Float64}, tau:: Float64, t:: Float64, x:: Vector{Float64}, lambda:: Vector{Float64}, nu:: Float64, out:: Vector{Float64})
  const n = size(TtT, 1)
  #TODO: Should use BLAS in-place multiply here.
  out[1:n] = TtT*x - Ttu - lambda + nu*ones(n)
  out[(n+1):(2*n)] = x .* lambda - (1/t)*ones(n)
  out[2*n+1] = sum(x) - tau
end

function bestWeights(this:: InteriorPointNonnegativeLasso, currentAtoms:: Vector{TransformAtom}, maxWeight:: Float64, image:: VectorizedImage, imageParams:: ImageParameters)
  # To shorten notation, let
  #   n = length(currentAtoms)
  #   T_i = currentAtoms[i].transform
  #   T = [T_1; T_2; ...; T_n]
  #   x_i = the weight of atom i in the solution (to be found)
  #   x = [x_1, ..., x_n]
  #   u = image
  #   tau = maxWeight
  # The problem is:
  # min_{x \in R^n} (1/2)||T x - u||_2^2 s.t. x \succeq 0, <x, 1> = tau
  # 
  # So we have a single affine constraint and n simple nonnegativity
  # inequalities, and our objective f_0 is a quadratic.  In terms of BV:
  # f_0 = (1/2)||T x - u||_2^2
  # \grad f_0 = T^T T x - T^T u
  # \grad^2 f_0 = T^T T
  # f_i = <x, -e_i>
  # \grad f_i = -e_i
  # \grad^2 f_i = 0
  # f = -x
  # D f = - eye(n)
  # A = ones(n)
  # b = tau
  # 
  # The Newton system at each iteration is therefore (taking lambda to be the
  # n-dimensional dual variable corresponding to the nonnegativity
  # constraints):
  # [T^T T,        -eye(n), ones(n);
  #  diag(lambda), diag(x), 0;
  #  ones(n)',     0,       0]
  # 
  # Though this can be solved by block inversion, I believe that would only 
  # save a constant amount of time, because one of the required inversions
  # is of the form (diag(x) + diag(lambda) (T^T T)^{-1})^{-1}, and I do not
  # know of a way to speed that up.  (It's not just a rank-one update; it's
  # easily-invertible-plus-arbitrary-diagonal, and the diagonal part and T^T T
  # are not in general similar.  So that seems hard.)  So we just perform a 
  # full solve at each iteration :-(.
  # 
  # The residual vector is (taking nu to be the scalar dual variable
  # corresponding to the single equality constraint and t to be the barrier
  # scale which we determine on each iteration):
  # [T^T T x - T^T u - lambda + nu ones(n),
  #  diag(lambda) x - (1/t) ones(n),
  #  <x, 1> - tau]
  # 
  # Whew.
  
  const T:: Matrix{Float64} = makeX(imageParams, [a.transform for a in currentAtoms])
  const TtT = T'*T
  const u = image
  const Ttu = T'*u
  const n = length(currentAtoms)
  const tau = maxWeight
  # Initialize weights uniformly.  TODO: Could be better to take existing 
  # weights and fudge them a bit so they're strictly feasible.
  x = ones(n) / tau
  # Initialize lambdas to -1/f_i(x), which is just 1/x_i
  lambda = 1.0 ./ x
  nu = 0.0 #FIXME: Not sure if this should be positive to start.
  r = zeros(2*n+1)
  
  while true
    # eta is the "surrogate duality gap", the duality gap if the current
    # iterates were primal- and dual-feasible (which is only true in the
    # limit).  It is <-f(x), lambda> = <x, lambda>.
    const eta = dot(x, lambda)
    const t = this.mu*n/eta
    kktResidual!(TtT, Ttu, tau, t, x, lambda, nu, r)
    scale!(r, -1.0)

    const primalInfeasibility = abs(r[2*n+1])
    const dualInfeasibility = norm(view(r, 1:n), 2)
    if primalInfeasibility < this.feasibilitySlack && dualInfeasibility < this.feasibilitySlack && eta < this.objectiveSlack
      break
    end
    
    #TODO: Preallocate memory for this.
    # Set up and solve the Newton system to find search directions.
    const newtonSystem = [TtT           -eye(n)      ones(n);
                          diagm(lambda) diagm(x)     zeros(n);
                          ones((1,n))   zeros((1,n)) 0]
    const searchDirections = newtonSystem \ r
    const deltaX = view(searchDirections, 1:n)
    const deltaLambda = view(searchDirections, (n+1):(2*n))
    const deltaNu = searchDirections[2*n+1]
    
    # Now we do a line search with the given directions.
    const s = lineSearch(this.backtrackingAlpha, this.backtrackingBeta, x, lambda, nu, deltaX, deltaLambda, deltaNu, r)
    
    # Take the step.
    Base.LinAlg.axpy!(s, deltaX, x)
    Base.LinAlg.axpy!(s, deltaLambda, lambda)
    nu += s*deltaNu
  end
  
  [TransformAtom(a.transform, weight) for (a, weight) in zip(currentAtoms, x)]
end
