export AtomicImageGenerator, generate
export RandomAtomicImageGenerator, generate

using Images
using ImageUtils

immutable AtomicImageGenerator <: DataGenerator{Image}
  imageInAtoms:: TransformedImage
end

function generate(this:: AtomicImageGenerator)
  decoded(this.imageInAtoms)
end

#FIXME: For now hard-coded to use a particular parameterized class of atoms.
# Could generalize if the filter had a reasonableRandomSample() method.
immutable RandomAtomicImageGenerator <: DataGenerator{Image}
  numAtoms:: Int64
  pixelSize:: Float64
  imageSize:: Int64
end

function generateRandomAtom()
  const weight = rand()
  const length = this.pixelSize*this.imageSize
  const center = length / 2
  const xRightShift = center + (rand() - .5)*length
  const yDownShift = center + (rand() - .5)*length
  const angleRadians = pi*rand()
  #FIXME: This one is kinda arbitrary.
  const parabolicScale = length*abs(randn()) / 5
  const transform = ParameterizedPixelatedFilter(defaultWavelet(), pixelSize, imageSize, xRightShift, yDownShift, angleRadians, parabolicScale)
  TransformAtom(transform, weight)
end

function generate(this:: RandomAtomicImageGenerator)
  const atoms = [generateRandomAtom() for i in 1:this.numAtoms]
  decoded(TransformedImage(atoms))
end