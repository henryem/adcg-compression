export SparsificationStrategy, sparsify
export NoSparsification, HardThresholding, FixedSparsity

# A strategy for taking a dense weight vector and producing a sparse one.
# Note that more sophisticated optimization-based strategies cannot use these
# schemes.
abstract SparsificationStrategy

function sparsify(this:: SparsificationStrategy, vector:: AbstractVector{Float64})
  raiseAbstract("sparsify", this)
end


immutable NoSparsification <: SparsificationStrategy end

function sparsify(this:: NoSparsification, vector:: AbstractVector{Float64})
  vector
end


immutable HardThresholding <: SparsificationStrategy
  threshold:: Float64
end

function sparsify(this:: HardThresholding, vector:: AbstractVector{Float64})
  sparsevec(map(e -> e > this.threshold ? e : 0.0, vector))
end


immutable FixedSparsity <: SparsificationStrategy
  proportionNonzeros:: Float64
end

function sparsify(this:: FixedSparsity, vector:: AbstractVector{Float64})
  const threshold = quantile(vector, 1-this.proportionNonzeros)
  sparsevec(map(e -> e > threshold ? e : 0.0, vector))
end