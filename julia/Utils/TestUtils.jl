# A little framework for writing unit tests.  Example usage:
# 
#   module MyClassTest
#   @testMethod "Tests whether 1+1==2" begin
#     @assertEq(1+1, 2)
#   end
# 
#   @testMethod "Tests whether 1+1==3" begin
#     @assertEq(1+1, 3)
#   end
#   end
# 
# The main advantage over just writing a series of @test statements is that
# individual test methods can fail without stopping other tests.  A real
# testing framework would add a count of the number and names of failed tests,
# but for now this is pretty threadbare.
# 
# In addition to @testMethod, several assertion methods are provided that are
# a little better than the builtin @test and @test_throws.  Both macros (which
# generate friendly failure messages automatically) and functions (for which
# you should provide friendly failure messages yourself) are provided.
# 
# Note: @testMethod is probably a little fragile because I don't fully
# understand how scoping interacts with macros in Julia.

using Base.Test

export @testMethod, @profileMethod, assertThat, assertTrue, @assertTrue, assertEq, @assertEq, assertThrows, @assertThrows, assertThrowsThis, @assertThrowsThis, assertApproxEq, @assertApproxEq, approximatelyEquals

# An ugly hack to display exceptions, including full stack traces, anywhere.
function printException(err:: Exception, backtrace)
  Base.showerror(STDOUT, err)
  Base.show_backtrace(STDOUT, backtrace)
  println()
end

function printException(err:: Base.Test.Error, backtrace)
  printException(err.err, backtrace)
end

# A fallback for Base.showerror on generic exception types.  For some bizarre
# reason the exception messages are not organized as methods of exceptions, and
# the closest we get is a series of print functions called Base.showerror().
# But Base.showerror() is not defined for all exceptions, so we need to define
# a fallback here.
function Base.showerror(io:: IO, e:: Exception)
  @printf("Exception: %s", string(e))
end

macro testMethod(testName, testExpr)
  quote
    try
      $(esc(testExpr))
    catch err
      @printf("The test \"%s\" failed with the following message:\n", $(testName))
      printException(err, catch_backtrace())
    end
  end
end

function assertThat(assertedTrueFunc:: Function, failureMessage:: String)
  Base.Test.do_test(assertedTrueFunc, failureMessage)
end

function assertTrue(assertedTrueValue:: Bool, assertedTruth:: String)
  assertThat(() -> assertedTrueValue, "Expected that $(assertedTruth), but found otherwise!")
end

macro assertTrue(expr)
  quote
    assertTrue($(esc(expr)), $(string(expr)))
  end
end

# Assert equality, with an arbitrary comparison function.
function assertEq(actualValue:: Any, expectedValue:: Any, comparator:: Function, valueName:: String, expectedValueName:: String)
  assertTrue(comparator(actualValue, expectedValue), "$(valueName) equals $(expectedValueName) (actual value was $(actualValue) but $(expectedValue) was expected)")
end

function assertEq(actualValue:: Any, expectedValue:: Any, valueName:: String, expectedValueName:: String)
  assertEq(actualValue, expectedValue, (x, y) -> x == y, valueName, expectedValueName)
end

macro assertEq(expr1, expr2)
  quote
    assertEq($(esc(expr1)), $(esc(expr2)), $(string(expr1)), $(string(expr2)))
  end
end

macro assertEq(expr1, expr2, comparatorExpr)
  quote
    assertEq($(esc(expr1)), $(esc(expr2)), $(esc(comparatorExpr)), $(string(expr1)), $(string(expr2)))
  end
end

function assertThrows(func, funcName)
  threwException = true
  try
    func()
    threwException = false
  end
  assertTrue(threwException, "evaluating \"$(funcName)\" throws an exception")
end

# Note: The exception thrown by @func is compared to @expectedException by
# stringifying them.  This is horrible, but since Julia apparently has no
# canonical way to compare arbitrary objects, this seems to be the best we
# can do.
function assertThrowsThis(func, expectedException, funcName)
  threwException = true
  try
    func()
    threwException = false
  catch err
    assertTrue(string(err) == string(expectedException), "$(funcName) throws exception $(string(expectedException))")
  end
  assertTrue(threwException, "evaluating \"$(funcName)\" throws an exception")
end

macro assertThrows(expr)
  quote
    assertThrows(() -> $(esc(expr)), $(string(expr)))
  end
end

macro assertThrowsThis(expr, expectedExceptionExpr)
  quote
    assertThrowsThis(() -> $(esc(expr)), $(esc(expectedExceptionExpr)), $(string(expr)))
  end
end

# Assert @value1 approximately equals @value2 in max-norm distance, up to @epsilon error.
function approximatelyEquals(value1:: Array, value2:: Array, epsilon:: Float64)
  maximum(abs(vec(value1) - vec(value2))) < epsilon
end

function approximatelyEquals(value1:: Float64, value2:: Float64, epsilon:: Float64)
  abs(value1 - value2) < epsilon
end

function assertApproxEq(value, expectedValue, valueName:: String, expectedValueName:: String, epsilon:: Float64)
  assertEq(value, expectedValue, (x, y) -> approximatelyEquals(x, y, epsilon), valueName, expectedValueName)
end

macro assertApproxEq(expr1, expr2, epsilonExpr)
  quote
    assertApproxEq($(esc(expr1)), $(esc(expr2)), $(string(expr1)), $(string(expr2)), $(esc(epsilonExpr)))
  end
end