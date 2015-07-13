export Wavelet, apply, gradient, defaultWavelet
export SimpleWavelet
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
  raiseAbstract("gradient", this)
end


immutable SimpleWavelet <: Wavelet
  image:: ImageParameters
end

function apply(this:: SimpleWavelet, p:: TwoDPoint)
  #FIXME: A Gaussian.  Probably not a great wavelet.
  (1/(2*pi)^(1/2)) * e^(-.5*(p[1]^2 + p[2]^2))
end

function apply(this:: SimpleWavelet, x:: Float64, y:: Float64)
  #FIXME: A Gaussian.  Probably not a great wavelet.
  (1/(2*pi)^(1/2)) * e^(-.5*(x^2 + y^2))
end

function toTwoDFunction(this:: SimpleWavelet)
  TwoDGaussianFunction()
end

function gradient(this:: SimpleWavelet, x:: Float64, y:: Float64)
  #FIXME
end

function defaultWavelet(image:: ImageParameters)
  SimpleWavelet(image)
end


immutable TwoDGaussianFunction <: TwoDFunction end

function Utils.apply(this:: TwoDGaussianFunction, x:: Float64, y:: Float64, totalScale:: Float64)
  (1/(2*pi)^(1/2)) * e^(-.5*(x^2 + y^2)) / totalScale
end