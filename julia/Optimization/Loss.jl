export Loss, loss, lossGradient, lossGradient!
export L2Loss

abstract Loss

function loss(this:: Loss, residual:: Vector{Float64})
  raiseAbstract("loss", this)
end

function lossGradient(this:: Loss, residual:: Vector{Float64})
  out = zeros(residual)
  gradient!(this, residual, out)
  out
end

function lossGradient!(this:: Loss, residual:: Vector{Float64}, out:: Vector{Float64})
  raiseAbstract("gradient!", this)
end


immutable L2Loss <: Loss end

function loss(this:: L2Loss, residual:: Vector{Float64})
  .5*norm(residual, 2)^2
end

function lossGradient!(this:: Loss, residual:: Vector{Float64}, out:: Vector{Float64})
  copy!(out, residual)
end