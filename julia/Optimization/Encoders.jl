export Encoder, encode, encodeAll

using ImageUtils

abstract Encoder

# Returns an EncodedImage, which includes its own method for decoding into
# a regular Image.
function encode(this:: Encoder, image:: VectorizedImage)
  raiseAbstract("encode", this)
end

# Encode many images at once.  Provided for convenience and, occasionally, for
# efficiencies inside the encoder.  Encoders should override this when they
# can provide a more efficient implementation.
# 
# Returns a Vector{EncodedImage}.
function encodeAll(this:: Encoder, images:: VectorizedImages)
  vec(mapslices(im -> encode(this, im), images, [1]))
end