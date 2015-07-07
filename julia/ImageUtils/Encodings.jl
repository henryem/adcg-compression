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


immutable TransformedImage <: EncodedImage
  transformAtoms:: Vector{TransformAtom}
end

function encoded(this:: TransformedImage)
  map(a -> a.weight, this.transformAtoms)
end

function decoded(this:: TransformedImage)
  toImage(mapreduce(synthesize, +, this.transformAtoms))
end