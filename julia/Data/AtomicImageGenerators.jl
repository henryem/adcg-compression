export AtomicImageGenerator, generate, singleImage, singleAtomImages
export RandomAtomicImageGenerator, generate

using Images
using ImageUtils

immutable AtomicImageGenerator <: DataGenerator{Image}
  imageInAtoms:: Vector{EncodedImage}
end

function generate(this:: AtomicImageGenerator)
  Image[decodedToImage(im) for im in this.imageInAtoms]
end

function singleImage(imageInAtoms:: EncodedImage)
  AtomicImageGenerator([imageInAtoms])
end

function singleAtomImages{T <: Transform}(transforms:: Vector{T})
  AtomicImageGenerator([TransformedImage(TransformAtom(t)) for t in transforms])
end

function singleAtomImages{T <: Transform}(transforms:: Vector{T}, weight:: Float64)
  AtomicImageGenerator([TransformedImage(TransformAtom(t, weight)) for t in transforms])
end

#FIXME: For now hard-coded to use a particular parameterized class of atoms.
# Could generalize if the filter had a reasonableRandomSample() method.
immutable RandomAtomicImageGenerator <: DataGenerator{Image}
  numImages:: Int64
  numAtoms:: Int64
  meanAtomWeight:: Float64
  parameterSpace:: ParameterSpace
end

function generateRandomAtom(this:: RandomAtomicImageGenerator)
  const weight = this.meanAtomWeight*2.0*rand()
  TransformAtom(uniformSample(this.parameterSpace), weight)
end

function generate(this:: RandomAtomicImageGenerator)
  map(1:this.numImages) do i
    const atoms = [generateRandomAtom(this) for a in 1:this.numAtoms]
    decodedToImage(TransformedImage(this.image, atoms))
  end
end