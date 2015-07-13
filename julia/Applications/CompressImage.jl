using ArgParse, Images
using Utils, Data, ImageUtils, Optimization

# Example:
#   ./run Applications/CompressImage.jl -e 'TrivialEncoder(ImageParameters(512),grid(ParameterizedPixelatedFilter,ImageParameters(512),10^4))' -d 'RandomAtomicImageGenerator(1,3,ImageParameters(512))'

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
    "--output-dir", "-o"
      help = "the directory in which to place any output (created if necessary)"
  end

  return parse_args(s)
end

function makeOutputDir!(dirname:: String)
  if dirname != nothing try mkdir(dirname) end end
  if dirname != nothing "$(dirname)" else "." end
end

function run()
  const args = parseArgs()
  if args["data-seed"] != nothing srand(args["data-seed"]) end
  const dataGenerator = eval(parse(args["data-generator"]))
  const data = generate(dataGenerator)
  const outDir = makeOutputDir!(args["output-dir"])
  # println([[e.transformAtoms[1].transform.xRightShift,
  #   e.transformAtoms[1].transform.yDownShift,
  #   e.transformAtoms[1].transform.angleRadians,
  #   e.transformAtoms[1].transform.parabolicScale] for e in dataGenerator.imageInAtoms])
  for i in 1:length(data)
    imwrite(data[i], "$(outDir)/original_$(i).jpg")
  end
  const vectorizedImages = map(toVectorizedImage, data)
  
  if args["encoder"] != nothing
    if args["encoder-seed"] != nothing srand(args["encoder-seed"]) end
    const encoder = eval(parse(args["encoder"]))
    const encodedImages = map(img -> encode(encoder, img), vectorizedImages)
    const decodedImages = map(decoded, encodedImages)

    # println(float(decodedImages[1].data))
  
    for i in 1:length(decodedImages)
      imwrite(decodedImages[i], "$(outDir)/compressed_$(i).jpg")
    end
  end
end

run()
