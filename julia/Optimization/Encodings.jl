export EncodedImage, encoded, decoded
export TransformedImage, encoded, decoded

using Utils

abstract EncodedImage

function encoded(this:: EncodedImage)
  raiseAbstract("encoded", this)
end

function decoded(this:: EncodedImage)
  raiseAbstract("decoded", this)
end


immutable TransformedImage
  transforms:: Vector{Transform}
  transformValues:: Vector{Float64}
end

function encoded(this:: TransformedImage)
  this.transformValues
end

function decoded(this:: TransformedImage)
 [synthesize(t, v) for (t, v) in zip(transforms, transformValues)]
end