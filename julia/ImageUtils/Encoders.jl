export Encoder, encode, encodeAll

abstract Encoder

# Returns an EncodedImage, which includes its own method for decoding into
# a regular Image.
function encode(this:: Encoder, image:: VectorizedImage)
  raiseAbstract("encode", this)
end

function encodeAll(this:: Encoder, images:: Vector{VectorizedImage})
  map(i -> encode(this, i), images)
end