# Compute Contract

## Scope

This page is the kernel compute contract router. Detailed authority is split by
responsibility so planning, collective primitives, lowering, backend handoff,
and CPU SIMD support do not share one large owner.

Public authority:

- `/kernel/include/kernel/program/compute/*`

Verification authority:

- `/kernel/tests/contract/run/compute.cpp`
- `/kernel/tests/contract/run/model.cpp`
- `/kernel/tests/contract/program/compute.cpp`
- `/kernel/tests/contract/program/compute/`
- `/tools/check/run`

## Verification Shape

`kernel.compute` owns the small scheduling, CPU lowering, and artifact
validation contract used by the development smoke loop.
`kernel.compute.model` owner contains DSL, graph, fusion, backend lowering,
collective primitive, metadata, planning, and resident-run contracts and stays
in `tools/check/run`. The two executables do not call each other or repeat semantic
owners. Kernel Program and Compute verification share one allocation
instrumentation object target, so the split does not create a second allocator
implementation or duplicate its compilation.

Compute contract tests follow the same owner split as the public contract:
thin routers call focused owners for DSL identity, fixed-op construction,
escape safety, fusion planning, fused IR build, backend lowering, runtime
binding validation, and CPU SIMD artifact validation. Shared lowering fixtures
are split under `/kernel/tests/contract/program/compute/lowering/` by
canonical byte helpers, forged IR builders, fixed nonlinear helpers, text
helpers, and focused operation-builder owners under `lowering/ops/`
(`basic`, `expanded`, `fixed/{scalar,bit,arithmetic}`, and name-stress builders).
DSL fixed-op contracts are split under `dsl/ops/fixed/` by scalar and predicate
coverage, bit and shift coverage, and nonlinear division, reciprocal, and root
coverage. Fixed saturating and multiply arithmetic contracts are split under
`fixed/arithmetic/` by DSL admission, backend lowering, malformed IR rejection,
and fused ternary operand remapping.
Core DSL wrappers for construction, unary ops, arithmetic, constants, primitive
bounds, select, and comparisons are owner-local under `dsl/functions/core/`;
core bounds split order and interval owners; select branches and comparisons use focused value helpers. Range bounds and predicates,
such as unordered interval clamping and inclusive checks,
live under `dsl/functions/range/`. The compact facade mirrors those owners and only forwards. Composite DSL helper coverage includes owner-local hash, noise, vector, stats,
aggregate/difference, ratio/standardization, standardized moments,
bounded approximation, fixed deterministic transcendental formulas,
complex scalar-pair formulas, signal-window formulas, linear,
matrix, affine, mix, polynomial evaluation/derivative, interpolation, mask,
tolerance, robust piecewise penalty, projection, reflection, and cross contract files under
the same `dsl/ops/` route. Map-local composite helpers must keep domain-free
math names and remain
pure formulas over the current lane; cross-lane range, ordering, movement, or
collective state belongs in graph-level primitives instead.
Focused tests include their smallest semantic leaf headers, so changing one
primitive cannot invalidate unrelated contract objects through an aggregate
include.
Symbolic-expression physical ownership and its template-to-compiled boundary
are owned by [IR Lowering](./compute/ir/lowering.md). The product `dsl.hpp`
facade contains no implementation body; consumers import the exact
declaration-owning leaf.
Fusion fixtures follow the same shape under `fusion/local/`: policy hashes,
graph fixtures, oversized binding fixtures, and fused op builders have separate
owners. Fusion planning contracts are split by semantic outcome under
`fusion/plan/`: successful plan shape, CPU-visible boundary decisions, and
fail-closed rejection/default reasons. Ownership remains split so added tests
stay with their semantic DSL, fusion, or backend lowering owner.
Fused-IR construction follows the same rule: `fusion/build.cpp` is only the
runner, `fusion/build/model.hpp` is the sole two-node graph and carrier fixture
authority, and the `carrier`, `pair`, `chain`, `output`, `shift`, and
`capacity` leaves own their distinct semantic outcomes.

Numeric algebra verification is routed by `numeric/algebra.cpp`.
`numeric/algebra/model.hpp` owns the fixed formats and identity-axis oracle
once, while the `matrix`, `factor`, `solve`, `spectrum`, `scratch`, `accuracy`,
and `identity` leaves own their respective plan, reference, initialized-state,
bit-exact, and policy contracts. Each leaf is a test translation-unit boundary,
and the runner executes the canonical semantic order.

Compute planning tests are owner-local as well: `plan.cpp` is only the runner,
while `plan/` owns fail-closed planner admission, deterministic identity,
shape/byte guard semantics, and dispatch-only projection parity.
Sort, histogram, and stencil primitive tests follow the same rule: each root
file is only the runner, while the matching folder owns planner rejection,
descriptor
identity, plan shape, and CPU reference semantics.
Graph rejection tests are split by failure domain: `graph/rejection.cpp` is
only the runner, while `graph/rejection/` owns primitive descriptor, node,
buffer, and policy/scalar fail-closed vocabulary.
DSL and backend-lowering rejection follow the same physical rule:
`dsl/reject.cpp` and `backend/lowering/reject.cpp` are runners only.
Their matching `reject/` folders own binding, storage, numeric-domain,
carrier, malformed-shape, and canonical-mask contracts. The shared
`compute/reject/model` owner provides the one header-only body fixture and one
compiled CPU/Metal/Vulkan accept-or-reject oracle; suite-local `model` owners
keep invalid-byte builders and operation-shape helpers with the suite that
consumes them. Every leaf constructs fresh state, so runner order is not a
semantic input.

## Detail Owners

- [Planning and graph identity](./compute/planning.md)
- [Collective primitive contracts](./compute/primitives.md)
- [IR and lowering](./compute/ir/lowering.md)
- [Backend handoff and run ordering](./compute/handoff.md)
- [CPU SIMD caps, artifact, and validation](./compute/cpu/simd.md)
- [Partition primitive details](./compute/partition.md)
- [Segmented scan](./compute/segmented/scan.md) and [segmented reduce](./compute/segmented/reduce.md)

## Boundary

Kernel compute remains deterministic authority only. It consumes declared maps,
IR, caps, plans, binding facts, and CPU SIMD caps; it never observes node, OS,
Metal, Vulkan, driver, PMU, clock, or benchmark state while planning.
Node owns
adapter observation and runtime evidence in
[`/node/docs/contracts/accel.md`](../../../node/docs/contracts/accel.md).
