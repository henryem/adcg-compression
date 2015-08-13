export Encoder, encode, encodeAll
export computeResidualAndLossGradient!

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


# Utility methods for encoders:

# Computes the gradient of @loss with respect to its input, applied to the
# residual obtained when we synthesize @atoms and subtract the target image
# @image.  Also places the current residual (which, for squared loss, will
# be the same as the gradient...) in @residual.
function computeResidualAndLossGradient!(p:: ImageParameters, image:: VectorizedImage, atoms:: Vector{TransformAtom}, loss:: Loss, encodedImage:: VectorizedImage, residual:: VectorizedImage, out:: VectorizedImage)
  const imageEncoded = TransformedImage(p, atoms)
  fill!(encodedImage, 0.0)
  decodeInto!(imageEncoded, encodedImage)
  #FIXME: This will allocate memory.
  residual[:] = encodedImage - image
  lossGradient!(loss, residual, out)
end