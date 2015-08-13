export InteriorPointNonnegativeLasso
export FixedStep

using Base.LinAlg


# Solves a nonnegative Lasso problem using a primal-dual interior point method.
# In relatively low dimension (say, <1000 atoms), this is probably the fastest
# available way to solve the nonnegative Lasso.  If it is hard to solve a
# d-by-d linear system, prefer a first-order method instead.
# 
# The particular method is described in p. 612 of Boyd and Vandenberghe 
# (algorithm 11.2).  The search directions are described by equation 11.54.
# Specialized to a QP, they can be simplified a bit.  So this currently handles
# only the quadratic loss, though it could be extended to other losses,
# probably with a drop in performance.
immutable InteriorPointNonnegativeLasso <: FiniteDimConvexSolver
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

# Finds a step in direction (deltaX, deltaLambda, deltaRho) from point
# (x, lambda, rho) that satisfies several conditions, detailed below.
# It is assumed that @residual contains the current residual, before
# a step is taken.
function lineSearch{A <: AbstractVector{Float64}}(alpha:: Float64, beta:: Float64, TtT:: Matrix{Float64}, Ttu:: Vector{Float64}, tau:: Float64, t:: Float64, x:: Vector{Float64}, lambda:: Vector{Float64}, rho:: Float64, deltaX:: A, deltaLambda:: A, deltaRho:: Float64, residual:: Vector{Float64})
  const n = size(x, 1)
  # We have to keep lambda (and rho) all-nonnegative, so our first condition on 
  # s is that this stays true.  So we upper-bound it by something just a little
  # smaller than the step size that makes some coordinate of lambda (or rho)
  # equal to 0.
  const sLambdaBound = min(
    1.0,
    deltaRho >= 0 ? 1.0 : -rho / deltaRho,
    minimum([dl >= 0 ? 1.0 : -l / dl for (l, dl) in zip(lambda, deltaLambda)]))

  # BV does backtracking for primal feasibility, but for our simple constraints
  # we can avoid that.
  const sPrimalBound = min(
    1.0,
    # tau = sum(x + s*deltaX) = sum(x) + s*sum(deltaX)
    #   => s = [tau - sum(x)] / sum(deltaX)
    sum(deltaX) <= 0 ? 1.0 : (tau - sum(x)) / sum(deltaX),
    minimum([dx >= 0 ? 1.0 : -currentX / dx for (currentX, dx) in zip(x, deltaX)]))

  function recalculateResidualNorm(s:: Float64)
    # Recalculate the residual efficiently, without memory allocation.  (This
    # is the true inner loop of the interior point method, so speed is most
    # critical here.)  To do this we use the residual with no step and then
    # add an appropriate delta to get the residual after a step of length s.
    #residual[1:n] = C - T^T T(x + s*deltaX) - (lambda+s*deltaLambda) + (rho+s*deltaRho)*ones(n)
    # Change = -s*T^T T deltaX - s*deltaLambda + s*deltaRho*ones(n)
    #residual[(n+1):(2*n)] = C + (x+s*deltaX) .* (lambda+s*deltaLambda) - (1/t)*ones(n)
    # Change = (x+s*deltaX) .* (lambda+s*deltaLambda) - x .* lambda
    #        = s*deltaX .* lambda + s*deltaLambda .* x + s^2*deltaX .* deltaLambda
    #residual[2n+1] = C - (rho+s*deltaRho)*(<1,s*deltaX+x> - tau)
    #        = C - (rho+s*deltaRho)*(<1,s*deltaX> + <1,x> - tau)
    #        = C - s*rho<1,deltaX> - s^2*deltaRho*<1,deltaX> - s*deltaRho*<1,x> + s*deltaRho*tau
    # Change = -s*deltaRho*<1,x> - s*rho*<1,deltaX> - s^2*deltaRho*<1,deltaX> + s*deltaRho*tau
    
    #TODO: Currently computing T^T deltaX via n dot products.  Could use BLAS 
    # instead...
    const dualResidualSquared = mapreduce(+, 1:n) do idx
      # (Using the symmetric structure of TtT here.)
      (residual[idx] + s*dot(sub(TtT, :, idx), deltaX) - s*deltaLambda[idx] + s*deltaRho)^2
    end
    centralityResidualSquared = mapreduce(+, 1:n) do idx
      (residual[idx+n] + s*deltaX[idx]*lambda[idx] + s*x[idx]*deltaLambda[idx] + s^2*deltaX[idx]*deltaLambda[idx])^2
    end
    centralityResidualSquared += (residual[2*n+1] - s*deltaRho*sum(x) - s*rho*sum(deltaX) - s^2*deltaRho*sum(deltaX) + s*deltaRho*tau)^2
    const residualNorm = sqrt(dualResidualSquared + centralityResidualSquared)
    residualNorm
  end
  
  s = 0.99*min(sLambdaBound, sPrimalBound)
  
  # Now we find a point where we have gotten a reduction of at least alpha per
  # unit of s.  We just backtrack from our maximum by powers of beta.
  const initialResidualNorm = norm(residual, 2)
  while true
    if recalculateResidualNorm(s) <= (1.0-alpha*s)*initialResidualNorm
      break
    elseif s < sLambdaBound*1e-10
      println("Warning: Stopping line search because the search direction seems bad!")
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
function kktResidual!(TtT:: Matrix{Float64}, Ttu:: Vector{Float64}, tau:: Float64, t:: Float64, x:: Vector{Float64}, lambda:: Vector{Float64}, rho:: Float64, out:: Vector{Float64})
  const n = size(TtT, 1)
  #TODO: Should use BLAS in-place multiply here.
  out[1:n] = TtT*x - Ttu - lambda + rho*ones(n)
  out[(n+1):(2*n)] = x .* lambda - (1.0/t)*ones(n)
  out[2*n+1] = -rho*(sum(x) - tau) - (1.0/t)
end

function bestWeights(this:: InteriorPointNonnegativeLasso, currentAtoms:: Vector{TransformAtom}, maxWeight:: Float64, image:: VectorizedImage, imageParams:: ImageParameters, l:: Loss)
  assert(isa(l, L2Loss))
  
  # To shorten notation, let
  #   n = length(currentAtoms)
  #   T_i = currentAtoms[i].transform
  #   T = [T_1; T_2; ...; T_n]
  #   x_i = the weight of atom i in the solution (to be found)
  #   x = [x_1, ..., x_n]
  #   u = image
  #   tau = maxWeight
  # The problem is:
  # min_{x \in R^n} (1/2)||T x - u||_2^2 s.t. x \succeq 0, <x, 1> <= tau
  # 
  # So we have a single affine inequality and n simple nonnegativity
  # inequalities, and our objective f_0 is a quadratic.  In terms of BV:
  # f_0 = (1/2)||T x - u||_2^2
  # \grad f_0 = T^T T x - T^T u
  # \grad^2 f_0 = T^T T
  # f_i = -x_i = <x, -e_i> (i = 1, ..., n)
  # \grad f_i = -e_i
  # \grad^2 f_i = 0
  # f_{n+1} = <x, 1> - tau
  # \grad f_{n+1} = 1
  # \grad^2 f_{n+1} = 0
  # f = (-x, <x, 1> - tau)
  # D f = (- eye(n); ones(1, n))
  # 
  # The Newton system at each iteration is therefore (taking lambda to be the
  # n-dimensional dual variable corresponding to the nonnegativity
  # constraints and rho to be the 1-dimensional dual variable corresponding
  # to the sum constraint):
  # [T^T T,             -eye(n),      ones(n);
  #  diag(lambda),      diag(x),      zeros(n);
  #  -rho*ones((1,n))', zeros((1,n)), <x, 1> - tau]
  # 
  # Though this can be solved by block inversion, I believe that would only 
  # save a constant amount of time, because one of the required inversions
  # is of the form (diag(x) + diag(lambda) (T^T T)^{-1})^{-1}, and I do not
  # know of a way to speed that up.  (It's not just a rank-one update; it's
  # easily-invertible-plus-arbitrary-diagonal, and the diagonal part and T^T T
  # are not in general similar.  So that seems hard.)  However, __solving__
  # the system (not inverting the matrix) by an iterative method could take
  # advantage of this structure and make this step faster by a factor of n or
  # so.  For now we just invert the whole thing naively.
  # 
  # The residual vector is (taking t to be the barrier scale which we
  # determine on each iteration):
  # [\grad f_0 + (D f)^T (lambda, rho),
  #  -diag(lambda, rho) f - (1/t) ones(n+1)]
  # = 
  # [T^T T x - T^T u - lambda + rho ones(n),
  #  diag(lambda) x - (1/t) ones(n),
  #  -rho*(<x,1> - tau)) - (1/t)]
  # 
  # Whew.
  
  const T:: Matrix{Float64} = makeX(imageParams, [a.transform for a in currentAtoms])
  const TtT = T'*T
  const u = image
  const Ttu = T'*u
  const n = length(currentAtoms)
  const numInequalities = n+1
  const tau = maxWeight
  
  # Initialize x to almost uniform, with a little bit of randomness to avoid
  # ties.
  # TODO: Could be better to take existing weights and fudge them a bit so 
  # they're strictly feasible.
  const startingTotalWeight = .9*tau
  x = (startingTotalWeight / n) * (ones(n) + (1e-8)*rand(n))
  # Initialize lambdas to -1/f_i(x), which is just 1/x_i
  lambda = 1.0 ./ x
  # Similarly, initialize rho to -1/f_{n+1}(x) > 0
  rho = -1.0 / (sum(x) - tau)
  r = zeros(2*n+1)
  i = 0
  
  while true
    i += 1
    # eta is the "surrogate duality gap", the duality gap if the current
    # iterates were primal- and dual-feasible (which is only true in the
    # limit).  It is <-f(x), (lambda, rho)> = <x, lambda> - (<x,1> - tau) rho.
    const eta = dot(x, lambda) - (sum(x) - tau)*rho
    const t = this.mu*numInequalities/eta
    kktResidual!(TtT, Ttu, tau, t, x, lambda, rho, r)

    const dualInfeasibility = norm(sub(r, 1:n), 2)
    if dualInfeasibility < this.feasibilitySlack && eta < this.objectiveSlack
      break
    end

    # Set up and solve the Newton system to find search directions.
    #TODO: Use operator splitting or something to solve this efficiently
    # (since we can invert TtT just once and everything else is diagonal).
    const newtonSystem = [TtT              -eye(n)      ones(n);
                          diagm(lambda)    diagm(x)     zeros(n);
                          -rho*ones((1,n)) zeros((1,n)) tau - sum(x)]
    # The b vector in the solve is -r, not r.
    scale!(r, -1.0)
    const searchDirections = newtonSystem \ r
    scale!(r, -1.0)
    const deltaX = sub(searchDirections, 1:n)
    const deltaLambda = sub(searchDirections, (n+1):(2*n))
    const deltaRho = searchDirections[2*n+1]
    
    # Now we do a line search with the given directions.
    const s = lineSearch(this.backtrackingAlpha, this.backtrackingBeta, TtT, Ttu, tau, t, x, lambda, rho, deltaX, deltaLambda, deltaRho, r)
    
    # Take the step.
    Base.LinAlg.axpy!(s, deltaX, x)
    Base.LinAlg.axpy!(s, deltaLambda, lambda)
    rho += s*deltaRho
  end
  
  [TransformAtom(a.transform, weight) for (a, weight) in zip(currentAtoms, x)]
end
