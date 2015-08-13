using Utils, ImageUtils, Optimization

function testSimpleProblem(solver:: FiniteDimConvexSolver)
  const im = ImageParameters(2)
  const filters = [TransformAtom(FixedFilter([1.0, 0.0, 0.0, 0.0], im)), TransformAtom(FixedFilter([0.0, 1.0, 0.0, 0.0], im))]
  const atoms = bestWeights(solver, filters, 5.0, [2.0, 1.0, 0.0, 0.0], im, L2Loss())
  const tolerance = 1e-4
  @assertApproxEq([a.weight for a in atoms], [2.0, 1.0], tolerance)
end

@testMethod "default InteriorPointNonnegativeLasso on a simple problem" begin
  testSimpleProblem(InteriorPointNonnegativeLasso())
end

function testBoundaryProblem(solver:: FiniteDimConvexSolver)
  const im = ImageParameters(2)
  const filters = [TransformAtom(FixedFilter([1.0, 0.0, 0.0, 0.0], im)), TransformAtom(FixedFilter([0.0, 1.0, 0.0, 0.0], im))]
  const atoms = bestWeights(solver, filters, 3.0, [2.0, 1.0, 0.0, 0.0], im, L2Loss())
  const tolerance = 1e-4
  @assertApproxEq([a.weight for a in atoms], [2.0, 1.0], tolerance)
end

@testMethod "default InteriorPointNonnegativeLasso on a simple problem where the optimum lies on the boundary" begin
  testBoundaryProblem(InteriorPointNonnegativeLasso())
end

function testConstrainedProblem(solver:: FiniteDimConvexSolver)
  const im = ImageParameters(2)
  const filters = [TransformAtom(FixedFilter([1.0, 0.0, 0.0, 0.0], im)), TransformAtom(FixedFilter([0.0, 1.0, 0.0, 0.0], im))]
  const atoms = bestWeights(solver, filters, 1.5, [2.0, 1.0, 0.0, 0.0], im, L2Loss())
  const tolerance = 1e-4
  @assertApproxEq([a.weight for a in atoms], [1.25, 0.25], tolerance)
end

@testMethod "default InteriorPointNonnegativeLasso on a simple problem where the unconstrained optimum lies outside the feasible set" begin
  testConstrainedProblem(InteriorPointNonnegativeLasso())
end

function testVeryConstrainedProblem(solver:: FiniteDimConvexSolver)
  const im = ImageParameters(2)
  const filters = [TransformAtom(FixedFilter([1.0, 0.0, 0.0, 0.0], im)), TransformAtom(FixedFilter([0.0, 1.0, 0.0, 0.0], im))]
  const atoms = bestWeights(solver, filters, 0.5, [2.0, 1.0, 0.0, 0.0], im, L2Loss())
  const tolerance = 1e-4
  @assertApproxEq([a.weight for a in atoms], [0.5, 0.0], tolerance)
end

@testMethod "default InteriorPointNonnegativeLasso on a simple problem where the unconstrained optimum lies outside the feasible set" begin
  testVeryConstrainedProblem(InteriorPointNonnegativeLasso())
end