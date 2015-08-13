export Loss, loss, lossGradient, lossGradient!
export L2Loss

# A loss function defined on the space of measurement residuals -- e.g.
# Phi mu - y.
abstract Loss

function loss(this:: Loss, residual:: Vector{Float64})
  raiseAbstract("loss", this)
end

# The gradient of the loss with respect to its input, (\nabla l)(r).
function lossGradient(this:: Loss, residual:: Vector{Float64})
  out = zeros(residual)
  lossGradient!(this, residual, out)
  out
end

# As lossGradient(), but in place.
function lossGradient!(this:: Loss, residual:: Vector{Float64}, out:: Vector{Float64})
  raiseAbstract("gradient!", this)
end


immutable L2Loss <: Loss end

function loss(this:: L2Loss, residual:: Vector{Float64})
  .5*norm(residual, 2)^2
end

function lossGradient!(this:: L2Loss, residual:: Vector{Float64}, out:: Vector{Float64})
  copy!(out, residual)
end