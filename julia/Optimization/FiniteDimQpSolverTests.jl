using Utils, ImageUtils, Optimization

function testSimpleProblem(solver:: FiniteDimQpSolver)
  const im = ImageParameters(2)
  const filters = [TransformAtom(FixedFilter([1.0, 0.0, 0.0, 0.0], im)), TransformAtom(FixedFilter([0.0, 1.0, 0.0, 0.0], im))]
  const atoms = bestWeights(solver, filters, 3.0, [2.0, 1.0, 0.0, 0.0], im)
  const tolerance = 1e-5
  @assertApproxEq([a.weight for a in atoms], [2.0, 1.0], tolerance)
end

@testMethod "default InteriorPointNonnegativeLasso on a simple problem" begin
  testSimpleProblem(InteriorPointNonnegativeLasso())
end