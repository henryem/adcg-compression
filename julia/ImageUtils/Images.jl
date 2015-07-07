export VectorizedImage, TwoDImage, SparseImage
export toVectorizedImage

using Image # (The 3rd-party package)

typealias VectorizedImage Vector{Float64}
typealias TwoDImage Matrix{Float64}
typealias SparseImage SparseMatrix{Float64}

function toVectorizedImage(im:: Image)
  #FIXME: Probably not this simple.
  data(im)
end