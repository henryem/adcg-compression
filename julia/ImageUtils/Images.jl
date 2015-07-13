export VectorizedImage
export toVectorizedImage
export ImageParameters, imageWidth, pixelCount, toImage

using Images # (The 3rd-party package)

typealias VectorizedImage Vector{Float64}

function toVectorizedImage(im:: Image)
  vec(float(data(convert(Image{Gray}, im))))
end

immutable ImageParameters
  pixelCountPerSide:: Int64
end

function ImageParameters(im:: Image)
  #FIXME: No idea if this works.
  assert(size(data(im), 1) == size(data(im), 2))
  ImageParameters(size(data(im), 1))
end

function imageWidth(this:: ImageParameters)
  this.pixelCountPerSide
end

function pixelCount(this:: ImageParameters)
  this.pixelCountPerSide * this.pixelCountPerSide
end

function toImage(this:: ImageParameters, im:: VectorizedImage)
  grayim(reshape(im, (this.pixelCountPerSide, this.pixelCountPerSide)))
end