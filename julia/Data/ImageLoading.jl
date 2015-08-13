export TestImageLoader, generate
export FileImageLoader, generate

using Images, TestImages

function toStandardFormat(im:: Image, format:: ImageParameters)
  standardizedImage = convert(Image{Gray}, im)
  const xSize, ySize = size(data(im))[1:2]
  
  if xSize > format.pixelCountPerSide
    standardizedImage = subim(1:format.pixelCountPerSide, 1:ySize)
  elseif xSize < format.pixelCountPerSide
    const diff = format.pixelCountPerSide - xSize
    const leftPad = ceil(Int64, diff / 2)
    const rightPad = floor(Int64, diff / 2)
    standardizedImage = grayim(padarray(data(standardizedImage), (leftPad, 0), (rightPad, 0), "value", 0.0))
  end
  
  if ySize > format.pixelCountPerSide
    standardizedImage = subim(1:format.pixelCountPerSide, 1:format.pixelCountPerSide)
  elseif ySize < format.pixelCountPerSide
    const diff = format.pixelCountPerSide - ySize
    const topPad = int(ceil(diff / 2))
    const downPad = int(floor(diff / 2))
    standardizedImage = grayim(padarray(data(standardizedImage), (0, topPad), (0, downPad), "value", 0.0))
  end
  
  standardizedImage
end


immutable TestImageLoader <: DataGenerator{Image}
  filenames:: Vector{String}
  imageFormat:: ImageParameters
end

function generate(this:: TestImageLoader)
  Image[toStandardFormat(testimage(f), this.imageFormat) for f in this.filenames]
end


immutable FileImageLoader <: DataGenerator{Image}
  filenames:: Vector{String}
  imageFormat:: ImageParameters
end

function generate(this:: FileImageLoader)
  Image[toStandardFormat(imread(f), this.imageFormat) for f in this.filenames]
end