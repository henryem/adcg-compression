export TrivialEncoder, encode
export SparsifyingEncoder, encode

using ImageUtils

# Given a fixed discrete basis of transforms, applies all of the transforms.
immutable TrivialEncoder <: Encoder
  im:: ImageParameters
  transforms:: Vector{Transform}
end

# Returns an EncodedImage, which includes its own method for decoding into
# a regular Image.
function encode(this:: TrivialEncoder, image:: VectorizedImage)
  println("Encoding an image of norm $(norm(image))...")
  const encodings = map(this.transforms) do t
    const a = TransformAtom(t, analyze(t, image))
    if a.weight > 1e-9
      println("Large weight for atom $(a) (norm $(norm(synthesize(a))))")
    end
    a
  end
  TransformedImage(this.im, encodings)
end


# Given a fixed discrete basis of transforms, applies all of the transforms,
# then sparsifies them in some simple one-pass way.
immutable SparsifyingEncoder <: Encoder
  im:: ImageParameters
  transforms:: Vector{Transform}
  threshold:: SparsificationStrategy
end

function encode(this:: SparsifyingEncoder, image:: VectorizedImage)
  const sparseEncoding = sparsify(this.threshold, map(t -> analyze(t, image), this.transforms))
  const nonzeroAtoms = map(nonzeroIdx -> TransformAtom(this.transforms[nonzeroIdx], sparseEncoding[nonzeroIdx]), find(sparseEncoding))
  TransformedImage(this.im, nonzeroAtoms)
end