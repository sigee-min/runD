# Skeleton Contract

## Scope

Kernel skeleton execution provides deterministic rank-generic index traversal
and serial ordered fold over caller-owned views. Scheduled execution reuses the
Kernel Program and Workspace contracts.

Authority:

- `/kernel/include/kernel/program/skeleton.hpp`
- `/kernel/include/kernel/program/executor.hpp`
- `/kernel/include/kernel/program/report.hpp`
- `/kernel/tests/contract/program/skeleton/`

## Model

`Space<Rank>` and `Index<Rank>` use `u64` extents and row-major
coordinates. Rank is positive and compile-time fixed. A space is admitted only
when its row-major product is representable in `u64`.

`View<T, Rank>` is non-owning. Construction validates data presence, shape,
stride, and addressability. Traversal preserves row-major identity and never
changes the caller's storage ownership.

## Execution

`each(space, callback)` is serial traversal. `each(seq(), ...)` names the
same policy explicitly.

`executor(workspace, backend, workers, alignment, tile_policy)` binds an
explicit WorkerBackend and prepares a reusable scheduled traversal.
`exec.prepare(space)` compiles the schedule and returns `PreparedEach`;
`prepared.run(callback)` validates generation and schedule identity before
dispatch. `each(exec, space, callback)` is the one-shot form.

`each(par(...), space, callback)` acquires an explicit `ParallelRuntime`
from the scoped provider, then follows the same executor path. It never falls
back to serial execution when provider acquisition or backend validation fails.

Scheduled packet identity is `u32`. A space whose linear unit count exceeds
that range fails before schedule construction. Physical tiling may refine
dispatch granularity but cannot change logical index order or coverage.

## Callback Boundary

A direct callback is a concrete, non-virtual, non-erased callable invoked with
one `Index<Rank>`. Function pointers and `std::function` are outside the
direct callback contract. Callback failure is recorded through the prepared
execution failure signal and returned with stable first-failure authority.

Partition callbacks receive one contiguous `Partition`. Indirect partition
identity is reserved for schedule placement and is not accepted by direct
row-major traversal.

## Alignment

`align(units)` requires a positive power-of-two boundary. Scheduled
partitions must start and end on that boundary except for the final logical
end. Alignment validation occurs before callback execution.

## Cold And Hot Paths

Preparation owns validation, workspace reset, reservation, schedule/program
compile, fold-graph compile, and vector growth. A successful no-growth proof
freezes the capacities required by repeated runs. The hot run validates the
prepared generation and dispatches through the frozen backend without
recompiling the program.

## Fold

`fold(space, accumulator, callback)` is serial row-major reduction.
Scheduled reduction order is owned by the fixed fold graph, never by worker
completion order. Strict floating-point policy follows the Reduction contract.

## Evidence

`SkeletonResult` reports admission, visited count, boundary validation, and
physical tile evidence. `KernelExecutionReport` projects backend dispatch and
worker timing evidence from the associated Workspace.

Verification covers shape, callback admission, views, alignment, scheduled
capacity, provider acquisition, physical tiling, and fixed-order fold under
`/kernel/tests/contract/program/skeleton/`.
