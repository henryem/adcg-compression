export SparseEncoder, encode
export SparsificationStrategy, sparsify
export HardThresholding, FixedSparsity

using ImageUtils

abstract SparsificationStrategy

function sparsify(this:: SparsificationStrategy, vector:: Vector{Float64})
  raiseAbstract("sparsify", this)
end


immutable HardThresholding <: SparsificationStrategy
  threshold:: Float64
end

function sparsify(this:: HardThresholding, vector:: Vector{Float64})
  sparsevec(map(e -> e > this.threshold ? e : 0.0, vector))
end


immutable FixedSparsity <: SparsificationStrategy
  proportionNonzeros:: Float64
end

function sparsify(this:: FixedSparsity, vector:: Vector{Float64})
  const threshold = quantile(vector, 1-this.proportionNonzeros)
  sparsevec(map(e -> e > threshold ? e : 0.0, vector))
end


# Given a fixed discrete basis of transforms, applies all of the transforms,
# then sparsifies them in some simple one-pass way.
immutable SparseEncoder <: Encoder
  im:: ImageParameters
  transforms:: Vector{Transform}
  threshold:: SparsificationStrategy
end

function encode(this:: SparseEncoder, image:: VectorizedImage)
  const sparseEncoding = sparsify(this.threshold, map(t -> analyze(t, image), this.transforms))
  const nonzeroAtoms = map(nonzeroIdx -> TransformAtom(this.transforms[nonzeroIdx], sparseEncoding[nonzeroIdx]), find(sparseEncoding))
  TransformedImage(this.im, nonzeroAtoms)
end