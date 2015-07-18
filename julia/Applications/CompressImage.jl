using ArgParse, Images
using Utils, Data, ImageUtils, Optimization

# Example:
#   ./run Applications/CompressImage.jl -e 'TrivialEncoder(ImageParameters(512),grid(SrprsSpace(ImageParameters(512)),10^4))' -d 'RandomAtomicImageGenerator(1,3,ImageParameters(512))'

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

function compareImages(imA:: AbstractVector{Float64}, imB:: AbstractVector{Float64})
  const normA = norm(imA)
  const normB = norm(imB)
  println("||imA|| = $(normA), ||imB|| = $(normB), ||imA||/||imB|| = $(normA / normB)")
  println("||imA||_inf = $(norm(imA, Inf)), ||imB||_inf = $(norm(imB, Inf)), ||imA||_inf/||imB||_inf = $(norm(imA, Inf) / norm(imB, Inf))")
  println("Unnormalized distance: $(norm(imA - imB))")
  println("Normalized distance:: $(norm(imA/normA - imB/normB))")
  println("Approximate l0 distance: $(length(filter(x->abs(x)>1e-9,imA-imB))) out of $(length(imA))")
  println("Approximate l0 distance on normalized vectors: $(length(filter(x->abs(x)>1e-9,imA/normA-imB/normB))) out of $(length(imA))")
end

function compareAtoms(atomA:: TransformAtom, atomB:: TransformAtom)
  xRightShiftDiff = atomA.transform.xRightShift - atomB.transform.xRightShift
  yDownShiftDiff = atomA.transform.yDownShift - atomB.transform.yDownShift
  angleRadiansDiff = atomA.transform.angleRadians - atomB.transform.angleRadians
  parabolicScaleDiff = atomA.transform.parabolicScale - atomB.transform.parabolicScale
  weightDiff = atomA.weight - atomB.weight
  println("Atomic representation diffs: ")
  println(xRightShiftDiff, yDownShiftDiff, angleRadiansDiff, parabolicScaleDiff, weightDiff)
end

function run()
  const args = parseArgs()
  if args["data-seed"] != nothing srand(args["data-seed"]) end
  const dataGenerator = eval(parse(args["data-generator"]))
  const data = generate(dataGenerator)
  const n = length(data)
  const outDir = makeOutputDir!(args["output-dir"])
  # println([[e.transformAtoms[1].transform.xRightShift,
  #   e.transformAtoms[1].transform.yDownShift,
  #   e.transformAtoms[1].transform.angleRadians,
  #   e.transformAtoms[1].transform.parabolicScale] for e in dataGenerator.imageInAtoms])
  for i in 1:n
    imwrite(data[i], "$(outDir)/$(i)_original.jpg")
  end
  const vectorizedImages = toVectorizedImages(data)
  
  if args["encoder"] != nothing && n > 0
    if args["encoder-seed"] != nothing srand(args["encoder-seed"]) end
    const encoder = eval(parse(args["encoder"]))
    const im = ImageParameters(data[1])
    const encodedImages = encodeAll(encoder, vectorizedImages)
    const vectorizedDecodedImages = decodeAll(encodedImages, im)
  
    for i in 1:n
      println("Analyzing compression result for image $(i)...")
      compareImages(sub(vectorizedImages, :, i), sub(vectorizedDecodedImages, :, i))
      #FIXME: Specialized for the case where the ith data point is generated
      # from the ith transform and there is no sparsity.
      # compareAtoms(dataGenerator.imageInAtoms[i].transformAtoms[1], encodedImages[i].transformAtoms[i])
      
      decodedImage = toImage(ImageParameters(data[i]), sub(vectorizedDecodedImages, :, i))
      imwrite(decodedImage, "$(outDir)/$(i)_compressed.jpg")
    end
  else
    println("No encoding to be done.")
  end
end

run()
