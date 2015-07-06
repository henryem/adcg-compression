export Transform, analyze, synthesize

abstract Transform

# Returns a single Float64.
function analyze(this:: Transform, image:: VectorizedImage)
  raiseAbstract("analyze", this)
end

# The inverse of analyze().
function synthesize(this:: Transform, weight:: Float64)
  raiseAbstract("synthesize", this)
end


abstract ParameterizedTransform <: Transform

function parameters(this:: ParameterizedTransform)
  raiseAbstract("parameters", this)
end


#NOTE: Should allow different image sizes, in which case that should be a
# parameter here.
immutable ParameterizedPixelatedFilter <: ParameterizedTransform
  # A function from continuous image coordinates to a real number weight.
  d:: Function
  pixelSize:: Float64
  imageWidthInPixels:: Int64
  xRightShift:: Float64
  yDownShift:: Float64
  angleRadians:: Float64
  parabolicScale:: Float64
end

function pixelatedTemplate(this:: ParameterizedFilter)
  template = zeros(this.imageWidthInPixels, this.imageWidthInPixels)
  for xPixel in 1:this.imageWidthInPixels
    for yPixel in 1:this.imageWidthInPixels
      xLeftBound = (xPixel-1)*this.pixelSize
      xRightBound = xPixel*this.pixelSize
      yTopBound = (yPixel-1)*this.pixelSize
      yBottomBound = yPixel*this.pixelSize
      
    end
  end
end

function analyze(this:: ParameterizedFilter, image:: VectorizedImage)
  dot(pixelizedTemplate(this), image)
end

function synthesize(this:: ParameterizedFilter, value:: Float64)
  value*pixelizedTemplate(this)
end