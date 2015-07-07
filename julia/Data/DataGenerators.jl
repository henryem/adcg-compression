export DataGenerator, generate

abstract DataGenerator{D}

function generate{D}(this:: DataGenerator{D})
  raiseAbstract("generate", this)
end