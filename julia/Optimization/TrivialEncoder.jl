export TrivialEncoder, encode

using ImageUtils

# Given a fixed discrete basis of transforms, applies all of the transforms.
immutable TrivialEncoder <: Encoder
  transforms:: Vector{ImageTransform}
end

# Returns an EncodedImage, which includes its own method for decoding into
# a regular Image.
function encode(this:: Encoder, image:: VectorizedImage)
  const encodings = map(t -> TransformAtom(t, transform(t, image)), this.transforms)
  TransformedImage(encodings)
end