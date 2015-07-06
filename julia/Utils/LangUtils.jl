export raiseAbstract

function raiseAbstract{T}(methodName:: String, this:: T)
  error("$(methodName) not implemented for $(typeof(this))")
end