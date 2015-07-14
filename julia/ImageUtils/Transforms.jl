export Transform, TransformAtom, analyze, synthesize, addSynthesized!
export ParameterizedTransform, parameters, appliedJacobian, grid

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


abstract ParameterizedTransform <: Transform

function parameters(this:: ParameterizedTransform)
  raiseAbstract("parameters", this)
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
function appliedJacobian(this:: ParameterizedTransform, image:: VectorizedImage)
  raiseAbstract("appliedJacobian", this)
end

# A Vector{ParameterizedTransform} containing a gridding of the parameter
# space.  At most numGridPoints transforms are included.
function grid{P <: ParameterizedTransform}(this:: Type{P}, image:: ImageParameters, numGridPoints:: Int64)
  raiseAbstract("grid", this)
end

function Base.show(io:: IO, m:: ParameterizedTransform)
  print(io, "$(typeof(m)):$(parameters(m))")
end