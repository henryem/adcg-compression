export TrivialEncoder, encode

using ImageUtils

# Given a fixed discrete basis of transforms, applies all of the transforms.
immutable TrivialEncoder <: Encoder
  im:: ImageParameters
  transforms:: Vector{Transform}
end

# Returns an EncodedImage, which includes its own method for decoding into
# a regular Image.
function encode(this:: TrivialEncoder, image:: VectorizedImage)
  const encodings = map(this.transforms) do t
    const a = TransformAtom(t, analyze(t, image))
    a
  end
  TransformedImage(this.im, encodings)
end