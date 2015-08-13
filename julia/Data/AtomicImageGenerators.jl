export AtomicImageGenerator, generate, singleImage, singleAtomImages
export RandomAtomicImageGenerator, generate

using Images
using ImageUtils

immutable AtomicImageGenerator <: DataGenerator{Image}
  imagesInAtoms:: Vector{EncodedImage}
end

function generate(this:: AtomicImageGenerator)
  Image[decodedToImage(im) for im in this.imagesInAtoms]
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


# Samples from the natural parameter range on @parameterSpace.  There are
# @numImages images, and each has @numAtoms, each with approximately 
# @meanAtomWeight weight.
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
    decodedToImage(TransformedImage(imageParameters(this.parameterSpace), atoms))
  end
end