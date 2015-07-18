export SrprsSpace, dimension, bounds, grid, uniformSample
export SrprsFilter, analyze, synthesize, addSynthesized!, parameters, parameterSpace, appliedJacobian


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

# The Jacobian of the function p -> synthesize(SrprsFilter(p)).
# A |parameters(this)|-by-pixelCount(this.image) matrix.
function pixelatedJacobian(this:: SrprsFilter)
  #FIXME
end

# The gradient of the function p -> <synthesize(SrprsFilter(p)), image>.  A |parameters(this)|-vector.
function appliedJacobian(this:: SrprsFilter, image:: VectorizedImage)
  pixelatedJacobian(this) * image
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
