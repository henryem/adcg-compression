export ParameterizedPixelatedFilter, analyze, synthesize, addSynthesized!, parameters, appliedJacobian, grid

using Cubature

# A filter that takes a function in continuous land, then rotates/scales/shifts
# it, then pixelates it.  This results in a linear functional on pixelated 
# images.  Applying the filter via analyze() just means applying this linear
# functional to the image.
immutable ParameterizedPixelatedFilter <: ParameterizedTransform
  d:: Wavelet
  image:: ImageParameters
  xRightShift:: Float64
  yDownShift:: Float64
  angleRadians:: Float64
  parabolicScale:: Float64
  filter:: Container{VectorizedImage}
  
  function ParameterizedPixelatedFilter(d:: Wavelet, image:: ImageParameters, xRightShift:: Float64, yDownShift:: Float64, angleRadians:: Float64, parabolicScale:: Float64)
    const lazyFilter = Lazy(VectorizedImage, () -> pixelatedFilter(d, image, xRightShift, yDownShift, angleRadians, parabolicScale))
    new(d, image, xRightShift, yDownShift, angleRadians, parabolicScale, lazyFilter)
  end
end

function ParameterizedPixelatedFilter(d:: Wavelet, image:: ImageParameters, parameters:: Vector{Float64})
  ParameterizedPixelatedFilter(d, image, parameters[1], parameters[2], parameters[3], parameters[4])
end

const NUM_GRID_POINTS_PER_DIM = 10

function pixelatedFilter(d:: Wavelet, image:: ImageParameters, xRightShift:: Float64, yDownShift:: Float64, angleRadians:: Float64, parabolicScale:: Float64)
  template = zeros(image.pixelCountPerSide, image.pixelCountPerSide)
  griddedPixelGridCubature!(toTwoDFunction(d), xRightShift, yDownShift, angleRadians, sqrt(parabolicScale), parabolicScale, NUM_GRID_POINTS_PER_DIM, template)  
  f = vec(template)
  f /= norm(f, 2)
  f
end

#FIXME: Old code for pixelatedFilter() using Cubature package:
# const RELATIVE_TOLERANCE = 1e-4
# for xPixel in 1:this.image.pixelCountPerSide
#   for yPixel in 1:this.image.pixelCountPerSide
#     const xLeftBound = xPixel-1
#     const xRightBound = xPixel
#     const yTopBound = yPixel-1
#     const yBottomBound = yPixel
#     const cubaturePixelEstimate = hcubature([xLeftBound, yTopBound], [xRightBound, yBottomBound], reltol = RELATIVE_TOLERANCE) do coords
#       transformedD(this, coords)
#     end
#     template[xPixel, yPixel] = cubaturePixelEstimate[1]
#   end
# end

function image(this:: ParameterizedPixelatedFilter)
  this.image
end

function analyze(this:: ParameterizedPixelatedFilter, image:: VectorizedImage)
  dot(get(this.filter), image)
end

#FIXME: Not sure if accepting AbstractVector causes a performance hit here.
function addSynthesized!(this:: ParameterizedPixelatedFilter, weight:: Float64, out:: AbstractVector{Float64})
  Base.LinAlg.axpy!(weight, get(this.filter), out)
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

const MIN_SCALE = 1e-2

function grid(this:: Type{ParameterizedPixelatedFilter}, image:: ImageParameters, numGridPoints:: Int64)
  const wavelet = defaultWavelet(image)
  const numGridPointsPerParam = int(floor(numGridPoints^(1/4)))
  const w = imageWidth(image)
  vec([ParameterizedPixelatedFilter(wavelet, image, xRightShift, yDownShift, angleRadians, parabolicScale)
    for xRightShift in linrange(0, w, numGridPointsPerParam),
        yDownShift in linrange(0, w, numGridPointsPerParam),
        angleRadians in linrange(0, pi, numGridPointsPerParam),
        parabolicScale in linrange(min(MIN_SCALE, w), w, numGridPointsPerParam)])
end