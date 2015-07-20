export Wavelet, apply, toTwoDFunction, gradient, gradient!, gradientX, gradientY
export defaultWavelet
export GaussianWavelet
export TwoDGaussianFunction

using Utils

typealias TwoDPoint Vector{Float64}

# Should be centered at (0,0).  (That's a little weird, because an image has
# its top-left corner there.  But downstream users of the wavelet are
# responsible for shifting appropriately.)
abstract Wavelet


function apply(this:: Wavelet, p:: TwoDPoint)
  raiseAbstract("apply", this)
end

function apply(this:: Wavelet, x:: Float64, y:: Float64)
  raiseAbstract("apply", this)
end

# TwoDFunction form of the wavelet.
function toTwoDFunction(this:: Wavelet)
  raiseAbstract("toIntegrable", this)
end

function gradient(this:: Wavelet, x:: Float64, y:: Float64)
  out = zeros(2)
  gradient!(this, x, y, out)
  out
end

function gradient!(this:: Wavelet, x:: Float64, y:: Float64, gradientOutput:: TwoDPoint)
  gradientOutput[1] = gradientX(this, x, y)
  gradientOutput[2] = gradientY(this, x, y)
end

function gradientX(this:: Wavelet, x:: Float64, y:: Float64)
  raiseAbstract("gradientX", this)
end

function gradientY(this:: Wavelet, x:: Float64, y:: Float64)
  raiseAbstract("gradientY", this)
end


function defaultWavelet(image:: ImageParameters)
  #FIXME: Probably not a great wavelet.
  GaussianWavelet(image)
end


immutable GaussianWavelet <: Wavelet
  image:: ImageParameters
end

function apply(this:: GaussianWavelet, p:: TwoDPoint)
  (1/(2*pi)^(1/2)) * e^(-.5*(p[1]^2 + p[2]^2))
end

function apply(this:: GaussianWavelet, x:: Float64, y:: Float64)
  (1/(2*pi)^(1/2)) * e^(-.5*(x^2 + y^2))
end

function toTwoDFunction(this:: GaussianWavelet)
  TwoDGaussianFunction()
end

function gradientX(this:: GaussianWavelet, x:: Float64, y:: Float64)
  -x*apply(this, x, y)
end

function gradientY(this:: GaussianWavelet, x:: Float64, y:: Float64)
  -y*apply(this, x, y)
end

function gradient!(this:: GaussianWavelet, x:: Float64, y:: Float64, gradientOutput:: TwoDPoint)
  const c = apply(this, x, y)
  gradientOutput[1] = -c*x
  gradientOutput[2] = -c*y
end


immutable TwoDGaussianFunction <: TwoDFunction end

function Utils.apply(this:: TwoDGaussianFunction, x:: Float64, y:: Float64)
  (1/(2*pi)^(1/2)) * e^(-.5*(x^2 + y^2))
end