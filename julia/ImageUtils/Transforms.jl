export Transform, TransformAtom, analyze, synthesize
export ParameterizedTransform, parameters, appliedJacobian
export Wavelet, apply, gradient, defaultWavelet
export SimpleWavelet
export ParameterizedPixelatedFilter, analyze, synthesize, parameters, appliedJacobian

using Cubature

abstract Transform

immutable TransformAtom
  transform:: Transform
  weight:: Float64
end

# Returns the degree of matching between this transform and @image.  For
# example, <T_theta, image>.
function analyze(this:: Transform, image:: VectorizedImage)
  raiseAbstract("analyze", this)
end

# The inverse of analyze().  For example, weight*T_theta.
function synthesize(this:: Transform, weight:: Float64)
  raiseAbstract("synthesize", this)
end

function synthesize(this:: TransformAtom)
  synthesize(this.transform, this.weight)
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
function grid{P <: ParameterizedTransform}(this:: Type{P}, numGridPoints:: Int64)
  raiseAbstract("grid", this)
end


abstract Wavelet

# p is of length 2.
function apply(this:: Wavelet, p:: Vector{Float64})
  raiseAbstract("apply", this)
end

# p is of length 2.
function gradient(this:: Wavelet, p:: Vector{Float64})
  raiseAbstract("gradient", this)
end


immutable SimpleWavelet <: Wavelet end

function apply(this:: SimpleWavelet)
  #FIXME
end

function gradient(this:: SimpleWavelet, p:: Vector{Float64})
  #FIXE
end

function defaultWavelet()
  SimpleWavelet()
end


# A filter that takes a function in continuous land, then rotates/scales/shifts
# it, then pixelates it.  This results in a linear functional on pixelated 
# images.  Applying the filter via analyze() just means applying this linear
# functional to the image.
immutable ParameterizedPixelatedFilter <: ParameterizedTransform
  d:: Wavelet
  dPrime:: Function
  pixelSize:: Float64
  imageWidthInPixels:: Int64
  xRightShift:: Float64
  yDownShift:: Float64
  angleRadians:: Float64
  parabolicScale:: Float64
  SR:: Matrix{Float64}
  f:: Vector{Float64}
  
  function ParameterizedPixelatedFilter(d:: Wavelet, pixelSize:: Float64, imageWidthInPixels:: Int64, xRightShift:: Float64, yDownShift:: Float64, angleRadians:: Float64, parabolicScale:: Float64)
    const S = [[this.parabolicScale^(1/2) 0], [0 this.parabolicScale]]
    const R = [[cos(this.angleRadians) -sin(this.angleRadians)], [sin(this.angleRadians) cos(this.angleRadians)]]
    const f = [this.xRightShift, this.yDownShift]
    new(d, pixelSize, imageWidthInPixels, xRightShift, yDownShift, angleRadians, parabolicScale, S*R, f)
  end
end

function ParameterizedPixelatedFilter(d:: Wavelet, pixelSize:: Float64, imageWidthInPixels:: Int64, parameters:: Vector{Float64})
  ParameterizedPixelatedFilter(d, pixelSize, imageWidthInPixels, parameters[1], parameters[2], parameters[3], parameters[4])
end

function transformedD(this:: ParameterizedPixelatedFilter, coords:: Vector{Float64})
  apply(this.d, this.SR * (x - this.f))
end

function pixelatedTemplate(this:: ParameterizedPixelatedFilter)
  template = zeros(this.imageWidthInPixels, this.imageWidthInPixels)
  for xPixel in 1:this.imageWidthInPixels
    for yPixel in 1:this.imageWidthInPixels
      xLeftBound = (xPixel-1)*this.pixelSize
      xRightBound = xPixel*this.pixelSize
      yTopBound = (yPixel-1)*this.pixelSize
      yBottomBound = yPixel*this.pixelSize
      template[xPixel, yPixel] = hcubature(coords -> transformedD(this, coords), [xLeftBound, yTopBound], [xRightBound, yBottomBound])
    end
  end
  template
end

function analyze(this:: ParameterizedPixelatedFilter, image:: VectorizedImage)
  dot(pixelizedTemplate(this), image)
end

function synthesize(this:: ParameterizedPixelatedFilter, weight:: Float64)
  weight*pixelizedTemplate(this)
end

function parameters(this:: ParameterizedPixelatedFilter)
  (this.xRightShift, this.yDownShift, this.angleRadians, this.parabolicScale)
end

function pixelatedJacobian(this:: ParameterizedPixelatedFilter)
  #FIXME
end

function appliedJacobian(this:: ParameterizedPixelatedFilter, image:: VectorizedImage)
  pixelatedJacobian(this) * image
end

function grid(this:: Type{ParameterizedPixelatedFilter}, numGridPoints:: Int64)
  const numGridPointsPerParam = floor(numGridPoints^(1/4))
  #FIXME
  # ParameterizedPixelatedFilter(d:: Wavelet, pixelSize:: Float64, imageWidthInPixels:: Int64, xRightShift:: Float64, yDownShift:: Float64, angleRadians:: Float64, parabolicScale:: Float64)
end