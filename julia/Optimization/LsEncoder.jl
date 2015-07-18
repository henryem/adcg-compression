export LsEncoder, encode
export LassoEncoder, encode, encodeAll

using Utils, ImageUtils


# We add a bit of regularization to make the problem better-conditioned.  Or
# perhaps we have a Gaussian prior on the transform weights :-S .
const REGULARIZATION = 1e-9

# Given a fixed discrete basis of n transforms (t_i)_{i=1}^{n} and an image y
# in R^d, finds a weight vector w in R^n so that
#   ||y - \sum_{i=1}^{n} w_i t_i||_2^2,
# the pixelwise squared error, is minimized.  In principle this is a great
# idea, but for large (or infinite) transform bases it is computationally
# infeasible.  (We can write this as ||X w - y||_2^2, so that the solution
# is (X^T X)^{-1} T^T y.  So we only ever have to do O(d^2 n + d^3) work.
# For a 64x64 image, this is feasible, but for larger images it is infeasible,
# and the difficulty grows linearly with the basis size.)
immutable LsEncoder <: Encoder
  im:: ImageParameters
  transforms:: Vector{Transform}
  sparsificationStrategy:: SparsificationStrategy
  XInv:: Container{Base.LinAlg.QRCompactWY}
  
  function LsEncoder{T <: Transform}(im:: ImageParameters, transforms:: Vector{T}, sparsificationStrategy:: SparsificationStrategy)
    const XInvFunc = function()
      #TODO: Adding the regularization in this way adds O(n^2) work, which
      # could be bad.  An alternative method is to actually compute
      # (X^T X + \lambda I)^{-1} (or its Cholesky factorization).  That would
      # require multiplying by X^T at solve time and is theoretically less 
      # stable.  This way we have to augment y with 0s at solve time, which
      # is theoretically fast but, because Julia offers no block-sparse vectors,
      # is in practice slow.  If we really care about performance here, we 
      # should have different methods for when n >> d and vice-versa.
      const regularizedX = makeRegularizedX(im, transforms, REGULARIZATION)
      qrfact(regularizedX)
    end
    new(im, transforms, sparsificationStrategy, Lazy(Base.LinAlg.QRCompactWY, XInvFunc))
  end
end

function LsEncoder{T <: Transform}(im:: ImageParameters, transforms:: Vector{T})
  LsEncoder(im, transforms, NoSparsification())
end

function encode(this:: LsEncoder, image:: VectorizedImage)
  const augmentedImage = vcat(image, zeros(length(this.transforms)))
  const weights = sparsify(this.sparsificationStrategy, get(this.XInv) \ augmentedImage)
  const atoms = [TransformAtom(weights[i], this.transforms[i]) for i in find(weights)]
  TransformedImage(this.im, atoms)
end

function encodeAll(this:: LsEncoder, images:: VectorizedImages)
  const augmentedImages = vcat(images, zeros(length(this.transforms), size(images, 2)))
  const weights = get(this.XInv) \ augmentedImages
  map(1:size(images, 2)) do imageIdx
    const sparseWeights = sparsify(this.sparsificationStrategy, sub(weights, :, imageIdx))
    const atoms = [TransformAtom(this.transforms[i], sparseWeights[i]) for i in find(sparseWeights)]
    TransformedImage(this.im, atoms)
  end
end

#FIXME: Arbitrary.
const LASSO_INVERSE_STEPSIZE = 1e-4
const LASSO_L1_REGULARIZATION = 1e-7

immutable LassoEncoder <: Encoder
  im:: ImageParameters
  transforms:: Vector{Transform}
  XTXInv:: Container{Base.LinAlg.Cholesky}
  XT:: Container{Matrix{Float64}}
  
  function LassoEncoder{T <: Transform}(im:: ImageParameters, transforms:: Vector{T})
    const XTXInvFunc = function()
      const regularizedXTX = makeRegularizedXTX(im, transforms, LASSO_INVERSE_STEPSIZE)
      cholfact(regularizedXTX)
    end
    const XTFunc = function()
      #FIXME: Should use a sparse matrix here.
      makeX(im, transforms)'
    end
    new(im, transforms, Lazy(Base.LinAlg.Cholesky, XTXInvFunc), Lazy(Matrix{Float64}, XTFunc))
  end
end

function encode(this:: LassoEncoder, image:: VectorizedImage)
  encodeAll(this, reshape(image, (length(image), 1)))[1]
end

function encodeAll(this:: LassoEncoder, images:: VectorizedImages)
  const solver = AdmmLassoSolver()
  println("images:")
  println([(i, images[i]) for i in find(images)])
  #TODO: X^T Y is simply the analysis of all the images under all the
  # transforms.  Potentially it is faster to call analyze() directly, to avoid
  # constructing X^T.  But currently the rows of X^T are constructed anyway
  # when analyze() is called, and this way we get to use BLAS to apply it.
  const weights = solveAll(solver, get(this.XTXInv), get(this.XT) * images, LASSO_L1_REGULARIZATION, LASSO_INVERSE_STEPSIZE)
  println("weights:")
  println([(i, weights[i]) for i in find(weights)])
  map(1:size(images, 2)) do imageIdx
    const atoms = [TransformAtom(this.transforms[i], weights[i,imageIdx]) for i in find(sub(weights, :, imageIdx))]
    TransformedImage(this.im, atoms)
  end
end