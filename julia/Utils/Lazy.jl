export Container, get
export Lazy, get

abstract Container{T}

function get{T}(this:: Container{T})
  raiseAbstract("get", this)
end

type Lazy{T} <: Container{T}
  isCalculated:: Bool
  valueType:: Type{T}
  calculator:: Function
  value:: T
  
  function Lazy(valueType, calculator)
    new(false, valueType, calculator)
  end
end

function Lazy{T}(valueType:: Type{T}, calculator:: Function)
  Lazy{T}(valueType, calculator)
end

function get{T}(this:: Lazy{T})
  if !this.isCalculated
    this.value = this.calculator()
    this.isCalculated = true
  end
  this.value
end