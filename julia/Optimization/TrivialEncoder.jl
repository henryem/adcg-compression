export TrivialEncoder, encode

using Images

# Given a fixed discrete basis of transforms, applies all of the transforms.
immutable TrivialEncoder <: Encoder
  transforms:: Vector{ImageTransform}
end