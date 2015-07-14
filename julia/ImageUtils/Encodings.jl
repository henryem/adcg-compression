export EncodedImage, imageParameters, encoded, decodeAll, decoded, decodeInto!, decodedToImage
export TransformedImage

using Utils

abstract EncodedImage

function imageParameters(this:: EncodedImage)
  raiseAbstract("imageParameters", this)
end

# Some representation of the image in encoded form.  For example, a list of
# matching filters and their weights.  The format is not defined, so this
# method's output is really suitable only for human consumption.
function encoded(this:: EncodedImage)
  raiseAbstract("encoded", this)
end

# More convenient and efficient version of hcat(map(decoded, these)...).
function decodeAll{E <: EncodedImage}(these:: Vector{E}, im:: ImageParameters)
  decodedVectors = zeros(pixelCount(im), length(these))
  for (i, t) in enumerate(these)
    decodeInto!(t, sub(decodedVectors, :, i))
  end
  decodedVectors
end

function decoded(this:: EncodedImage)
  decodedVector = zeros(pixelCount(imageParameters(this)))
  decodeInto!(this, decodedVector)
  decodedVector
end

# As decoded(), but decodes into an existing output @out.  @out is assumed
# empty (all zeros) and of the right size.
#FIXME: We only really need to support Array and SubArray here.  AbstractVector
# is of course technically the right type to accept, but it may cause a
# performance hit.
function decodeInto!(this:: EncodedImage, out:: AbstractVector{Float64})
  raiseAbstract("decodeInto!", this)
end

function decodedToImage(this:: EncodedImage)
  toImage(imageParameters(this), decoded(this))
end


immutable TransformedImage <: EncodedImage
  im:: ImageParameters
  transformAtoms:: Vector{TransformAtom}
end

function TransformedImage(im:: ImageParameters, transformAtom:: TransformAtom)
  TransformedImage(im, [transformAtom])
end

function imageParameters(this:: TransformedImage)
  this.im
end

function encoded(this:: TransformedImage)
  map(a -> a.weight, this.transformAtoms)
end

function decodeInto!(this:: TransformedImage, out:: AbstractVector{Float64})
  for a in this.transformAtoms
    addSynthesized!(a, out)
  end
end