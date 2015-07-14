export LsEncoder, encode
export LassoEncoder, encode, encodeAll

using Utils, ImageUtils


# Forms the linear operator of application of @transforms.
function makeX{T <: Transform}(im:: ImageParameters, transforms:: Vector{T})
  X = zeros(pixelCount(im), length(transforms))
  for (i, t) in enumerate(transforms)
    addSynthesized!(t, 1.0, sub(X, :, i))
  end
  X
end

# Forms the linear operator of application of @transforms, plus a padding of
# features equal to regularization*I.  The padding makes X suitable for use in, 
# for example, regularized QR.
function makeRegularizedX{T <: Transform}(im:: ImageParameters, transforms:: Vector{T}, regularization:: Float64)
  const d = pixelCount(im)
  const n = length(transforms)
  regularizedX = zeros(d + n, n)
  for (i, t) in enumerate(transforms)
    addSynthesized!(t, 1.0, sub(regularizedX, 1:d, i))
  end
  for i in 1:n
    regularizedX[d+i,i] = regularization
  end
  regularizedX
end

function makeRegularizedXTX{T <: Transform}(im:: ImageParameters, transforms:: Vector{T}, regularization:: Float64)
  const X = makeX(im, transforms)
  X' * X + regularization*eye(length(transforms))
end


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
      println("regularizedX in LsEncoder:")
      println([(i, regularizedX[i]) for i in find(regularizedX)])
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
  println("augmentedImages:")
  println([(i, augmentedImages[i]) for i in find(augmentedImages)])
  const weights = get(this.XInv) \ augmentedImages
  println("weights:")
  println([(i, weights[i]) for i in find(weights)])
  map(1:size(images, 2)) do imageIdx
    const sparseWeights = sparsify(this.sparsificationStrategy, sub(weights, :, imageIdx))
    const atoms = [TransformAtom(this.transforms[i], sparseWeights[i]) for i in find(sparseWeights)]
    TransformedImage(this.im, atoms)
  end
end

const LASSO_L2_REGULARIZATION = 1e-4

immutable LassoEncoder <: Encoder
  im:: ImageParameters
  transforms:: Vector{Transform}
  XInv:: Container{Base.LinAlg.QRCompactWY}
  
  function LassoEncoder(im:: ImageParameters, transforms:: Vector{Transform})
    #FIXME: Augment image with 0s.  And rename this.  And fix dimensions here as above.
    const XInvFunc = function()
      const regularizedX = makeRegularizedX(im, transforms, LASSO_L2_REGULARIZATION)
      qrfact(X)
    end
    new(im, transforms, Lazy(Base.LinAlg.QRCompactWY, XInvFunc))
  end
end

function encode(this:: LassoEncoder, image:: VectorizedImage)
  encodeAll(this, reshape(image, (length(image), 1)))[1]
end

function encodeAll(this:: LassoEncoder, images:: VectorizedImages)
  const solver = LassoSolver()
  # The l1 regularization term.
  const lambda = 1e-3 #FIXME: Arbitrary.
  #TODO: Expensive.  Should preallocate memory, maybe.
  const augmentedImages = vcat(images, zeros(length(this.transforms), size(images, 2)))
  const weights = solveAll(solver, get(this.XInv), augmentedImages, lambda)
  map(1:size(images, 2)) do imageIdx
    const atoms = [TransformAtom(this.transforms[i], weights[i,imageIdx]) for i in find(sub(weights, :, imageIdx))]
    TransformedImage(this.im, atoms)
  end
end