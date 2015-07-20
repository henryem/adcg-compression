export SrprsFilter, analyze, synthesize, addSynthesized!, parameters, parameterSpace, parameterGradient!
export SrprsSpace, dimension, bounds, grid, uniformSample

# Utility method to apply a shift-rotate-scale transform.
function srprsX(x:: Float64, y:: Float64, xRightShift:: Float64, yDownShift:: Float64, sint:: Float64, cost:: Float64, xScale:: Float64, yScale:: Float64)
  const xShifted = x - xRightShift
  const yShifted = y - yDownShift
  const xRotated = cost*xShifted - sint*yShifted
  xRotated / xScale
end

# Utility method to apply a shift-rotate-scale transform.
function srprsY(x:: Float64, y:: Float64, xRightShift:: Float64, yDownShift:: Float64, sint:: Float64, cost:: Float64, xScale:: Float64, yScale:: Float64)
  const xShifted = x - xRightShift
  const yShifted = y - yDownShift
  const yRotated = sint*xShifted + cost*yShifted
  yRotated / yScale
end

function srprsScale(xScale:: Float64, yScale:: Float64)
  1.0 / (xScale*yScale)
end


# A filter that takes a function in continuous land, then rotates/scales/shifts
# it, then pixelates it.  This results in a linear functional on pixelated 
# images.  Applying the filter via analyze() just means applying this linear
# functional to the image.
immutable SrprsFilter <: ParameterizedTransform
  parameterSpace #:: SrprsSpace, but Julia doesn't allow circular references...
  parameters:: Vector{Float64}
  filter:: Container{VectorizedImage}
  
  function SrprsFilter(parameterSpace, parameters:: Vector{Float64})
    const lazyFilter = Lazy(VectorizedImage, () -> pixelatedFilter(parameterSpace.image.pixelCountPerSide, parameterSpace.d, parameters))
    new(parameterSpace, parameters, lazyFilter)
  end
end

function SrprsFilter(d:: Wavelet, image:: ImageParameters, parameters:: Vector{Float64})
  SrprsFilter(SrprsSpace(d, image), parameters)
end

xRightShift(this:: SrprsFilter) = this.parameters[1]
xRightShift(p:: Vector{Float64}, :: Type{SrprsFilter}) = p[1]
yDownShift(this:: SrprsFilter) = this.parameters[2]
yDownShift(p:: Vector{Float64}, :: Type{SrprsFilter}) = p[2]
angleRadians(this:: SrprsFilter) = this.parameters[3]
angleRadians(p:: Vector{Float64}, :: Type{SrprsFilter}) = p[3]
parabolicScale(this:: SrprsFilter) = this.parameters[4]
parabolicScale(p:: Vector{Float64}, :: Type{SrprsFilter}) = p[4]

const NUM_GRID_POINTS_PER_DIM = 10

function pixelatedFilter(p:: Int64, d:: Wavelet, parameters:: Vector{Float64})
  template = zeros(p, p)
  const xR = xRightShift(parameters, SrprsFilter)
  const yD = yDownShift(parameters, SrprsFilter)
  const aR = angleRadians(parameters, SrprsFilter)
  const s = parabolicScale(parameters, SrprsFilter)
  griddedPixelGridCubature!(toTwoDFunction(d), xR, yD, aR, sqrt(s), s, NUM_GRID_POINTS_PER_DIM, template)  
  f = vec(template)
  f /= norm(f, 2)
  f
end

function parameters(this:: SrprsFilter)
  this.parameters
end

function parameterSpace(this:: SrprsFilter)
  this.parameterSpace:: SrprsSpace
end

function image(this:: SrprsFilter)
  this.parameterSpace.image
end

function analyze(this:: SrprsFilter, image:: VectorizedImage)
  dot(get(this.filter), image)
end

#FIXME: Not sure if accepting AbstractVector causes a performance hit here.
function addSynthesized!(this:: SrprsFilter, weight:: Float64, out:: AbstractVector{Float64})
  Base.LinAlg.axpy!(weight, get(this.filter), out)
end

# The gradient of the scalar function p -> <synthesize(SrprsFilter(p)), im>,
# evaluated at parameters(this).  A |parameters(this)|-vector.
#NOTE: This is currently optimized for being called once.  Analyze(), which is
# analogous to this method, is optimized (via caching) for being called many 
# times.  This is convenient for current uses of this class, but totally
# arbitrary and confusing from an interface-design perspective.  Should
# consider options for refactoring.
function parameterGradient!(this:: SrprsFilter, im:: VectorizedImage, out:: AbstractVector{Float64})
  # const gradientFunc = SinglePointSrprsGradient(this, im)
  # const w = imageWidth(image(this))
  # const numGridPointsPerDim = w * NUM_GRID_POINTS_PER_DIM
  # vectorGridCubature!(gradientFunc, 0.0, float(w), 0.0, float(w), numGridPointsPerDim, out)
  
  # Using a numerical gradient for now.  Note that we should _definitely_
  # use a streaming analyze() for this, if we're going to stick with numerical
  # gradients.
  numericalGradient!(p -> analyze(makeTransform(parameterSpace(this), p), im), parameters(this), 1e-11, out)
end


# The canonical parameter space for an SrprsFilter.
# 
# "Shift Rotate PaRabolic Scale" parameters -- an x right shift, a y down 
# shift, an angle in radians, and a parabolic scale (i.e. the x direction is 
# scaled by scale^(1/2) and y direction is scaled by scale).
# 
# All parameters have box constraints: Shifts are in [0, imageWidth(this.im)];
# angles are in [0, pi] (which works better than [0, 2pi] as long as the mother
# wavelet is antisymmetrical); scales are in [MIN_SCALE, 2*imageWidth(this.im)]
# (where MIN_SCALE is some constant smaller than 1, below which size we don't
# expect to see any resolvable atoms in the world).  (One reason to impose a 
# minimum scale: when we use gridding to _approximate_ the integral of a 
# wavelet over a pixel, atoms with very small scale may be integrated 
# inaccurately.)
# 
# Mnemonic: Surprise, parameters!
immutable SrprsSpace <: ParameterSpace{SrprsFilter}
  d:: Wavelet
  image:: ImageParameters
end

function SrprsSpace(image:: ImageParameters)
  SrprsSpace(defaultWavelet(image), image)
end

# Parameters are [xRightShift, yDownShift, angleRadians, parabolicScale].
const NUM_SRPRS_PARAMETERS = 4

function dimension(this:: SrprsSpace)
  NUM_SRPRS_PARAMETERS
end

const MIN_SRPRS_SCALE = 1e-1

function bounds(this:: SrprsSpace, dim:: Int64)
  if dim == 1
    (0, imageWidth(this.image))
  elseif dim == 2
    (0, imageWidth(this.image))
  elseif dim == 3
    (0, pi)
  elseif dim == 4
    const w = imageWidth(this.image)
    (min(MIN_SRPRS_SCALE, w), 2*w)
  else
    error("Dim $(dim) out of bounds for $(this).")
  end
end


# The full parameter gradient of a filter is the gradient of the function
#   p -> <synthesize(SrprsFilter(p)), image> .
# This can be estimated as a sum of products of pixel values with the values
# of synthesize() at single points.  Here we compute the gradient of one such 
# summand.  We can then average these over many points to estimate the full 
# gradient.
immutable SinglePointSrprsGradient <: VectorTwoDFunction
  filter:: SrprsFilter
  image:: VectorizedImage
end

function Utils.outputDimension(this:: SinglePointSrprsGradient)
  4
end

function continuousToPixel(continuousImageCoordinate:: Float64)
  if continuousImageCoordinate == 0.0
    # In this common special case, we still want to round up, since it is at 
    # the edge of pixel 1.
    1
  else
    ceil(Int64, continuousImageCoordinate)
  end
end

using Debug

function Utils.addApplied!(this:: SinglePointSrprsGradient, x:: Float64, y:: Float64, output:: Vector{Float64})
  const d = this.filter.parameterSpace.d
  const p = parameters(this.filter)
  const w = imageWidth(image(this.filter))
  
  const pixelX:: Int64 = continuousToPixel(x)
  const pixelY:: Int64 = continuousToPixel(y)
  const flatPixelIdx = pixelX+w*(pixelY-1)
  const imageValue = this.image[flatPixelIdx]
  
  const xR = xRightShift(this.filter)
  const yD = yDownShift(this.filter)
  const t = angleRadians(this.filter)
  const sinT = sin(t)
  const cosT = cos(t)
  const s = parabolicScale(this.filter)
  const sInv = 1.0/s
  const sqrtS = sqrt(s)
  const sqrtSInv = 1.0 / sqrtS
  const totalScale = srprsScale(sqrtS, s)

  #FIXME: It seems that the gradient is incorrectly large when we start
  # at exactly the optimum.  Could be due to the gridding strategy.
  const transformedX = srprsX(x, y, xR, yD, sinT, cosT, sqrtS, s)
  const transformedY = srprsY(x, y, xR, yD, sinT, cosT, sqrtS, s)
  const waveletValue = totalScale*apply(d, transformedX, transformedY)
  const waveletGradX = totalScale*gradientX(d, transformedX, transformedY)
  const waveletGradY = totalScale*gradientY(d, transformedX, transformedY)
  
  # Note: Some of this could be described a bit more concisely using matrices,
  # but for a 2-dimensional problem, linear algebra is much less efficient
  # than manual multiplication, and we do care about efficiency in this code.
  # d/dxRightShift = <w'_t(x,y), S_t R_t (-1, 0)>
  const ddxRightShift = (waveletGradX*sqrtSInv*(-cosT) + waveletGradY*sInv*(-sinT))
  # d/dyDownShift = <w'_t(x,y), S_t R_t (0, -1)>
  const ddyDownShift = (waveletGradX*sqrtSInv*sinT + waveletGradY*sInv*(-cosT))
  # d/dangleRadians = <w'_t(x,y), S_t d/dangleRadians(R_t) (x - xRightShift, y - yDownShift)
  const ddangleRadians = (waveletGradX*sqrtSInv*(-sinT*(x-xR) - cosT*(y-yD)) + waveletGradY*sInv*(cosT*(x-xR) - sinT*(y-yD)))
  # d/dparabolicScale has an extra term introduced by the product rule, since
  # it hits the whole wavelet:
  # d/dparabolicScale = -3/2 s^{5/2} w_t(x,y) + <w'_t(x,y), d/dparabolicScale(S_t) R_t (x - xRightShift, y - yDownShift)>
  const ddparabolicScale = (-3/2)*s^(-5/2)*waveletValue + (waveletGradX*(-1/2)*s^(-3/2)*(cosT*(x-xR) - sinT*(y-yD)) + waveletGradX*(-1)*s^(-2)*(sinT*(x-xR) + cosT*(y-yD)))
  
  output[1] += imageValue*ddxRightShift
  output[2] += imageValue*ddyDownShift
  output[3] += imageValue*ddangleRadians
  output[4] += imageValue*ddparabolicScale
end