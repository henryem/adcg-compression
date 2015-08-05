export Transform, TransformAtom, analyze, synthesize, addSynthesized!
export FixedFilter
export makeX, makeRegularizedX, makeRegularizedXTX
export ParameterSpace, dimension, bounds, makeTransform, grid, uniformSample
export ParameterizedTransform, parameters, parameterGradient!

abstract Transform

immutable TransformAtom
  transform:: Transform
  weight:: Float64
end

function TransformAtom(transform:: Transform)
  TransformAtom(transform, 1.0)
end

# ImageParameters for the images produced by synthesize().
function image(this:: Transform)
  raiseAbstract("image", this)
end

# Returns the degree of matching between this transform and @image.  For
# example, <T_theta, image>.
function analyze(this:: Transform, image:: VectorizedImage)
  raiseAbstract("analyze", this)
end

# The inverse of analyze().  For example, weight*T_theta.  The result is
# added in place to @out.
function addSynthesized!(this:: Transform, weight:: Float64, out:: AbstractVector{Float64})
  raiseAbstract("addSynthesized!", this)
end

function synthesize(this:: Transform, weight:: Float64)
  synthesized = zeros(pixelCount(image(this)))
  addSynthesized!(this, weight, synthesized)
  synthesized
end

function synthesize(this:: TransformAtom)
  synthesize(this.transform, this.weight)
end

function addSynthesized!(this:: TransformAtom, out:: AbstractVector{Float64})
  addSynthesized!(this.transform, this.weight, out)
end


# A trivial filter that is parameterized by an image.
immutable FixedFilter <: Transform
  filter:: VectorizedImage
  imageParams:: ImageParameters
end

function image(this:: FixedFilter)
  this.imageParams
end

function analyze(this:: FixedFilter, image:: VectorizedImage)
  dot(this.filter, image)
end

function addSynthesized!(this:: FixedFilter, weight:: Float64, out:: AbstractVector{Float64})
  Base.LinAlg.axpy!(weight, this.filter, out)
end


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
# for example, regularized QR for solving a least-squares problem involving X.
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

# Forms the linear operator X^T X + r*I, where X is the linear operator of
# application of @transforms, and r is the regularization parameter.
function makeRegularizedXTX{T <: Transform}(im:: ImageParameters, transforms:: Vector{T}, regularization:: Float64)
  const X = makeX(im, transforms)
  X' * X + regularization*eye(length(transforms))
end


# A transform (e.g. an image filter) determined by a vector of parameters.
# Each transform type is associated with a single ParameterSpace type, say P.
# Implementations must provide a constructor accepting arguments:
#   (space:: P, parameters:: Vector{Float64})
abstract ParameterizedTransform <: Transform

function parameters(this:: ParameterizedTransform)
  raiseAbstract("parameters", this)
end

# The ParameterSpace describing this transform's parameters.
function parameterSpace(this:: ParameterizedTransform)
  raiseAbstract("parameterSpace", this)
end

# Our transform is a function of both parameters \theta and an input image
# x, and it returns a scalar which is the weight of the filter when applied to
# the image.  In math notation we can think of this as
#   T: \Theta \to \Reals^{d \times 1}.
# If \Theta is in \Reals^p, then the Jacobian J of T at a point \theta is in 
# \Reals^{p \times d}.
# Multiplying this by an image results in a vector in \Theta again.  This is
# the direction of greatest increase of analyze(image) in \Theta at some point 
# in \Theta.
# The gradient is filled into @out.
function parameterGradient!(this:: ParameterizedTransform, image:: VectorizedImage, out:: AbstractVector{Float64})
  raiseAbstract("parameterGradient!", this)
end

function Base.show(io:: IO, m:: ParameterizedTransform)
  print(io, "$(typeof(m)):$(parameters(m))")
end


# Metadata about the parameter space for one particular kind of
# ParameterizedTransform, T.  Subclasses may provide more concrete metadata.
# Subclasses should contain enough information to construct instances of T
# given a parameter vector.
# 
# For simplicity, we assume that the parameter space is a bounded, box-
# constrained subset of R^d for some finite d.
# 
# Default implementations of grid() and uniformSample() assume that the
# uniform measure on the space gives a reasonable coverage of the parameterized
# transforms.  Sometimes this is not true, in which case implementations should 
# provide their own implementations of those methods.
abstract ParameterSpace{T <: ParameterizedTransform}

# The number of parameters.  It is always true that
#   length(parameters(t)) == dimension(parameterSpace(t))
# for t:: T.
function dimension(this:: ParameterSpace)
  raiseAbstract("dimension", this)
end

# The bounds of the box constraints.  A 2-tuple (lowerBound, upperBound).
function bounds(this:: ParameterSpace, dim:: Int64)
  raiseAbstract("bounds", this)
end

function makeTransform{T <: ParameterizedTransform}(this:: ParameterSpace{T}, parameters:: Vector{Float64})
  #NOTE: T should provide a constructor like this.
  T(this, parameters)
end

# A Vector{T} containing a gridding of the parameter space.  At most 
# numGridPoints transforms are included.  (So, with a simple gridding, each
# dimension gets numGridPoints^(1/dimension(this)) different values.)
function grid{T <: ParameterizedTransform}(this::ParameterSpace{T}, numGridPoints:: Int64)
  const numGridPointsPerParam = floor(Int64, numGridPoints^(1.0/dimension(this)))
  const numGridPointsRounded = numGridPointsPerParam^dimension(this)
  const ranges = [linrange(bounds(this, i)[1], bounds(this, i)[2], numGridPointsPerParam) for i in 1:dimension(this)]
  theGrid:: Vector{ParameterizedTransform} = ParameterizedTransform[]
  # I am so sorry for the following line of code.  All it does is fill in
  # theGrid from the Cartesian product of elements of @ranges, each element
  # being a parameter vector that is then mapped to a transform.  It does so
  # rather inefficiently (in the constant-factor sense).
  cartesianmap(
    (p...) -> push!(theGrid, makeTransform(this, Float64[ranges[i][p[i]] for i in 1:dimension(this)])),
    tuple([numGridPointsPerParam for i in 1:dimension(this)]...))
  theGrid
end

# A uniform sample from the parameter space.
function uniformSample{T <: ParameterizedTransform}(this:: ParameterSpace{T})
  const params = [(bounds(this, i)[2] - bounds(this, i)[1])*rand() + bounds(this, i)[1] for i in 1:dimension(this)]
  makeTransform(this, params)
end