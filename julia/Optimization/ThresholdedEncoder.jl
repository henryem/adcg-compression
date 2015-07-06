export ThresholdedEncoder, encode
export SparsificationStrategy, sparsify
export HardThresholding, FixedSparsity

using Images


abstract SparsificationStrategy

function sparsify(this:: SparsificationStrategy, vector:: Vector{Float64})
  raiseAbstract("sparsify", this)
end


immutable HardThresholding <: ThresholdingStrategy
  threshold:: Float64
end

function sparsify(this:: HardThresholding, vector:: Vector{Float64})
  sparsevec(map(e -> e > this.threshold ? e : 0.0, vector))
end


immutable FixedSparsity <: ThresholdingStrategy
  proportionNonzeros:: Float64
end

function sparsify(this:: FixedSparsity, vector:: Vector{Float64})
  const threshold = quantile(vector, 1-this.proportionNonzeros)
  sparsevec(map(e -> e > threshold ? e : 0.0, vector))
end


# Given a fixed discrete basis of transforms, applies all of the transforms,
# then hard-thresholds to generate sparsity.
immutable ThresholdedEncoder <: Encoder
  transforms:: Vector{ImageTransform}
  threshold:: ThresholdingStrategy
end

function encode(this:: ThresholdedEncoder, image:: VectorizedImage)
  const sparseEncoding = sparsify(this.threshold, map(t -> transform(t, image), this.transforms))
  #FIXME: Hopefully this does not copy the array.  Not sure about the semantics
  # of immutables.
  TransformedImage(this.transforms, sparseEncoding)
end