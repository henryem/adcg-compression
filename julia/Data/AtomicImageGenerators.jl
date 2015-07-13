export AtomicImageGenerator, generate, singleImage, singleAtomImages
export RandomAtomicImageGenerator, generate

using Images
using ImageUtils

immutable AtomicImageGenerator <: DataGenerator{Image}
  imageInAtoms:: Vector{EncodedImage}
end

function generate(this:: AtomicImageGenerator)
  Image[decoded(im) for im in this.imageInAtoms]
end

function singleImage(imageInAtoms:: EncodedImage)
  AtomicImageGenerator([imageInAtoms])
end

function singleAtomImages{T <: Transform}(im:: ImageParameters, transforms:: Vector{T})
  AtomicImageGenerator([TransformedImage(im, TransformAtom(t)) for t in transforms])
end

#FIXME: For now hard-coded to use a particular parameterized class of atoms.
# Could generalize if the filter had a reasonableRandomSample() method.
immutable RandomAtomicImageGenerator <: DataGenerator{Image}
  numImages:: Int64
  numAtoms:: Int64
  meanAtomWeight:: Float64
  meanAtomScale:: Float64
  image:: ImageParameters
end

function generateRandomAtom(this:: RandomAtomicImageGenerator)
  const weight = this.meanAtomWeight*2.0*rand()
  const len = imageWidth(this.image)
  const xRightShift = rand()*len
  const yDownShift = rand()*len
  const angleRadians = pi*rand()
  const parabolicScale = .1 + this.meanAtomScale*2.0*rand()
  const transform = ParameterizedPixelatedFilter(defaultWavelet(this.image), this.image, xRightShift, yDownShift, angleRadians, parabolicScale)
  TransformAtom(transform, weight)
end

function generate(this:: RandomAtomicImageGenerator)
  map(1:this.numImages) do i
    const atoms = [generateRandomAtom(this) for a in 1:this.numAtoms]
    decoded(TransformedImage(this.image, atoms))
  end
end