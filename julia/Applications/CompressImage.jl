using ArgParse, Images
using Utils, Data, ImageUtils, Optimization

# Example:
#   ./run Applications/CompressImage.jl -e 'grid(ParameterizedPixelatedEncoder,10^4)' -d 'RandomAtomicImageGenerator(10,1.0,100)'
#FIXME: How to pass image dimensions to grid()?

function parseArgs()
  s = ArgParseSettings()
  
  @add_arg_table s begin
    "--encoder", "-e"
      help = "the Encoder to use, a Julia string (without spaces)"
    "--data-generator", "-d"
      help = "the DataGenerator to use, a Julia string"
    "--data-seed"
      help = "the random seed for generating the data"
      arg_type = Int
    "--encoder-seed"
      help = "the random seed for the encoder"
      arg_type = Int
  end

  return parse_args(s)
end

function run()
  const args = parseArgs()
  if args["data-seed"] != nothing srand(args["data-seed"]) end
  const dataGenerator = eval(parse(args["data-generator"]))
  const data = generate(dataGenerator)
  const vectorizedImages = map(toVectorizedImage, data)
  if args["encoder-seed"] != nothing srand(args["encoder-seed"]) end
  const encoder = eval(parse(args["encoder"]))
  const encodedImages = encodeAll(encoder, vectorizedImages)
  const decodedImages = map(decoded, encodedImages)
  const viewableImages = map(toImage, decodedImages)
  
  #FIXME: Display viewableImages or compute distances on decodedImages or something.
  imwrite(viewableImages[1], "compressed.jpg")
  imwrite(data[1], "original.jpg")
end

run()
