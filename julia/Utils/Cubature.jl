export TwoDFunction, apply
export VectorTwoDFunction, outputDimension, apply!

export griddedPixelGridCubature!, gridCubature, vectorGridCubature!

# Some methods for fast cubature (numerical integration).  The package
# Cubature provides a more general-purpose toolkit than this, but unfortunately 
# it seems to involve unnecessary memory allocations that ruin performance.
# Here we specialize to 2 dimensions and avoid memory allocation.

# A function f: R^2 -> R.
abstract TwoDFunction

function apply(this:: TwoDFunction, x:: Float64, y:: Float64)
  raiseAbstract("apply", this)
end


# A function f: R^2 -> R^d, where d is finite.
abstract VectorTwoDFunction

function outputDimension(this:: VectorTwoDFunction)
  raiseAbstract("dimension", this)
end

function addApplied!(this:: VectorTwoDFunction, x:: Float64, y:: Float64, out:: Vector{Float64})
  raiseAbstract("addApplied!", this)
end


#FIXME: Document.  Computes box integrals over a shifted, rotated, scaled grid.
# Each box integral is itself computed by simple gridding.
function griddedPixelGridCubature!(f:: TwoDFunction, xRightShift:: Float64, yDownShift:: Float64, angleRadians:: Float64, xScale:: Float64, yScale:: Float64, numGridPointsPerPixelDim:: Int64, result:: Array{Float64,2})
  for yPixel = 1:size(result, 2)
    for xPixel = 1:size(result, 1)
      const topLeftX = xPixel - xRightShift - 1.0
      const topLeftY = yPixel - yDownShift - 1.0
      const sint = sin(angleRadians)
      const cost = cos(angleRadians)
      @inbounds result[xPixel, yPixel] = gridCubature(f, topLeftX, 1.0, topLeftY, 1.0, sint, cost, xScale, yScale, numGridPointsPerPixelDim)
    end
  end
end

# Computes a single integral over a shifted, rotated, scaled box, using naive 
# gridding.
function gridCubature(f:: TwoDFunction, topLeftX:: Float64, xWidth:: Float64, topLeftY:: Float64, yWidth:: Float64, sint:: Float64, cost:: Float64, xScale:: Float64, yScale:: Float64, numGridPointsPerPixelDim:: Int64)
  result = 0.0
  for y = linrange(topLeftY, topLeftY+yWidth, numGridPointsPerPixelDim)
    for x = linrange(topLeftX, topLeftX+xWidth, numGridPointsPerPixelDim)
      xRotated = cost*x - sint*y
      yRotated = sint*x + cost*y
      xRotated /= xScale
      yRotated /= yScale
      result += apply(f, xRotated, yRotated)
    end
  end
  const totalScale = xScale * yScale * numGridPointsPerPixelDim^2 / (xWidth * yWidth)
  result / totalScale
end

# As gridCubature, but for vector-valued functions.  It is assumed that
# @results is all-zero.
function vectorGridCubature!(fs:: VectorTwoDFunction, topLeftX:: Float64, xWidth:: Float64, topLeftY:: Float64, yWidth:: Float64, numGridPointsPerPixelDim:: Int64, results:: Vector{Float64})
  for y = linrange(topLeftY, topLeftY+yWidth, numGridPointsPerPixelDim)
    for x = linrange(topLeftX, topLeftX+xWidth, numGridPointsPerPixelDim)
      addApplied!(fs, x, y, results)
    end
  end
  const totalScale = numGridPointsPerPixelDim^2 / (xWidth * yWidth)
  scale!(results, 1/totalScale)
end
