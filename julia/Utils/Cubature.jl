export TwoDFunction, apply

export griddedPixelGridCubature!

# Some methods for fast cubature (numerical integration).  The package
# Cubature provides a more general-purpose than this, but unfortunately it
# seems to involve unnecessary memory allocations that ruin performance.
# Here we specialize to 2 dimensions and avoid memory allocation.

abstract TwoDFunction

#FIXME: Document
function apply(this:: TwoDFunction, x:: Float64, y:: Float64)
  raiseAbstract("apply", this)
end


#FIXME: Document.  Computes box integrals over a shifted, rotated, scaled grid.
# Each box integral is itself computed by simple gridding.
function griddedPixelGridCubature!(f:: TwoDFunction, xRightShift:: Float64, yDownShift:: Float64, angleRadians:: Float64, xScale:: Float64, yScale:: Float64, numGridPointsPerPixel:: Int64, result:: Array{Float64,2})
  for yPixel = 1:size(result, 2)
    for xPixel = 1:size(result, 1)
      const topLeftX = xPixel - xRightShift - 1.0
      const topLeftY = yPixel - yDownShift - 1.0
      const sint = sin(angleRadians)
      const cost = cos(angleRadians)
      @inbounds result[xPixel, yPixel] = griddedCubature(f, topLeftX, topLeftY, sint, cost, xScale, yScale, numGridPointsPerPixel)
    end
  end
end

function griddedCubature(f:: TwoDFunction, topLeftX:: Float64, topLeftY:: Float64, sint:: Float64, cost:: Float64, xScale:: Float64, yScale:: Float64, numGridPointsPerPixel:: Int64)
  result = 0.0
  totalScale = xScale*yScale
  for y = linrange(topLeftY, topLeftY+1.0, numGridPointsPerPixel)
    for x = linrange(topLeftX, topLeftX+1.0, numGridPointsPerPixel)
      xRotated = cost*x - sint*y
      yRotated = sint*x + cost*y
      xRotated /= xScale
      yRotated /= yScale
      result += apply(f, xRotated, yRotated) / totalScale
    end
  end
  result / numGridPointsPerPixel^2
end

