using Images


abstract Encoder

# Returns an EncodedImage, which includes its own method for decoding into
# a regular Image.
function encode(this:: Encoder, image:: VectorizedImage)
  raiseAbstract("encode", this)
end