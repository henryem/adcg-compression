export numericalGradient!

# Compute approximate gradients numerically.  Useful for checking explicit
# formulae.
function numericalGradient!(f:: Function, point:: AbstractVector{Float64}, stepSize:: Float64, out:: AbstractVector{Float64})
  const d = length(point)
  @inbounds for dim = 1:d
    const original = point[dim]
    point[dim] = original + stepSize
    const forwardValue = f(point)
    point[dim] = original - stepSize
    const backwardValue = f(point)
    out[dim] = (forwardValue - backwardValue) / (2*stepSize)
    point[dim] = original
  end
end