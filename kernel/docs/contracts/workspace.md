# Workspace Dispatch Contract

## Scope

`Workspace` is caller-owned reusable storage for Kernel schedule, fold,
program, placement, dispatch telemetry, and worker statistics. It is the sole
capacity owner for `KernelProgram` preparation and execution.

Authority:

- `/kernel/include/kernel/schedule/workspace.hpp`
- `/kernel/include/kernel/dispatch/kernel.hpp`
- `/kernel/include/kernel/dispatch/orchestrator.hpp`
- `/kernel/src/schedule/workspace/`
- `/kernel/src/dispatch/`
- `/kernel/tests/contract/workspace/`
- `/kernel/tests/contract/dispatch/`

## Storage

Workspace retains:

- schedule partitions
- packet work units and deterministic ordered packet indices
- packet-to-partition and placement scratch
- partition loads, counts, offsets, and write offsets
- fold slots, fold graph nodes, and reduction edges
- one `KernelProgram` and generation
- worker partition and timing vectors
- telemetry and the last stable reason

`ResetWorkspace` clears live values and program identity without replacing
owning vectors. Reserved capacity remains available for warm preparation.

## Reservation

`WorkspaceReservation` states every required vector capacity.
`ReserveWorkspace` grows those vectors during the cold path.
`WorkspaceCapacity` reports actual retained capacity, and
`WorkspaceSatisfiesReservation` proves component-wise sufficiency.

`ScheduleWorkspaceReservation` and
`KernelProgramWorkspaceReservation` derive requirements from the same
projection and fold-graph laws used by compilation. `KernelProgramCapacityProof`
records required and available capacity plus the minimum checked margin.

A no-growth compile or run is admitted only when the proof is checked and
satisfied. Capacity failure is precise and cannot be replaced by an empty
program or an allocation attempt on the hot path.

### Compute tile run boundary

Compute tile preparation uses a normal owning Workspace once to compile the
immutable `ComputeTileRunPlan`. A bound tile run does not clone that Workspace.
Its `ComputeTileRunStorage` contains a mutable Workspace control whose
`KernelProgram` views point into the pinned immutable plan, while the four
worker-stat arrays arrive as explicit typed spans in
`ComputeTileRunStorageView` and are passed as explicit dispatch sinks.

This is a narrower execution boundary than general Workspace preparation:
schedule, packet, placement, and fold arrays are immutable plan data during a
tile run, so the mutable run owner must not reserve a second copy merely to
satisfy vector-capacity checks. `ComputeTileRunPlan::bind` checks the failure,
worker-tile, and all worker-stat span capacities component by component before
publishing the view. The storage phase prevents binding while workers or an
unconsumed async result can observe the current pointers.

## Schedule Compilation

Schedule compilation preserves packet identity and deterministic placement.
Uniform, contiguous-balanced, capacity-weighted, and weighted-stable placement
may change partition shape but not logical packet order. Ordered packet views
are explicit and remain owned by Workspace.

## Dispatch

`RunPreparedProgram` validates Workspace, program generation, backend
capabilities, dispatch callback, partition count, no-growth proof, and optional
worker-stat sinks. It then executes the frozen `ScheduleView` through
`WorkerBackend`.

`WorkerBackend` exposes worker count, affinity policy, capability evidence,
static partition execution, and optional asynchronous submission. Capability
fields are accepted only when their corresponding function pointers exist.
Nested, width-mismatched, non-static, or insufficient no-allocation backends
fail before callback execution.

Worker completion order is non-semantic. Schedule packet identity and fold
graph order remain the only ordering authority.

## Telemetry

Telemetry distinguishes declared capability, checked proof, derived schedule
shape, and measured runtime data. Worker participation and timing fields are
reported only when their sinks were supplied and populated. Missing evidence is
not inferred from topology or backend names.

## Verification

Focused Workspace contracts prove reservation, warm reset, placement,
no-growth, packet identity, and capacity telemetry. Dispatch contracts prove
backend validation, exact partition coverage, failure signaling, worker stats,
and orchestrator handoff. Run those routes before `tools/check/run`.
