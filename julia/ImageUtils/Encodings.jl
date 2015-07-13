export EncodedImage, encoded, decoded
export TransformedImage, encoded, decoded

using Utils

abstract EncodedImage

# Some representation of the image in encoded form.  For example, a list of
# matching filters and their weights.  The format is not defined, so this
# method's output is really suitable only for human consumption.
function encoded(this:: EncodedImage)
  raiseAbstract("encoded", this)
end

function decoded(this:: EncodedImage)
  raiseAbstract("decoded", this)
end


immutable TransformedImage <: EncodedImage
  im:: ImageParameters
  transformAtoms:: Vector{TransformAtom}
end

function TransformedImage(im:: ImageParameters, transformAtom:: TransformAtom)
  TransformedImage(im, [transformAtom])
end

function encoded(this:: TransformedImage)
  map(a -> a.weight, this.transformAtoms)
end

function decoded(this:: TransformedImage)
  #TODO: Use addSynthesized! instead.
  toImage(this.im, mapreduce(synthesize, +, this.transformAtoms))
end