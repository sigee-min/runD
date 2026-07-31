# Program Contract

## Scope

Kernel Program compiles a deterministic schedule, fixed-order fold graph,
physical tile plan, backend contract, placement metadata, telemetry schema, and
no-growth capacity proof into one `KernelProgram` stored in caller-owned
`Workspace`.

Public authority:

- `/kernel/include/kernel/program/build.hpp`
- `/kernel/include/kernel/program/model.hpp`
- `/kernel/include/kernel/program/request.hpp`
- `/kernel/include/kernel/program/executor.hpp`
- `/kernel/include/kernel/program/phase.hpp`
- `/kernel/include/kernel/program/report.hpp`
- `/kernel/include/kernel/program/compute/tile/run.hpp`
- `/kernel/include/kernel/schedule/workspace.hpp`

Implementation authority:

- `/kernel/src/program/`
- `/kernel/src/dispatch/orchestrator/run.cpp`
- `/kernel/src/schedule/workspace/`

## Compile Request

`KernelProgramCompileRequest` owns the schedule request, worker backend,
physical tile policy, fold operation, strict floating-point policy, worker-stat
choice, and no-allocation requirement. Compile derives backend capabilities
from the supplied backend and never discovers a runtime or device.

A successful compile publishes:

- one generation-bound `ScheduleView`
- one `KernelProgramTilePlan`
- one fixed-order `FoldGraphView`
- one `KernelProgramDispatchContract`
- one checked `KernelProgramCapacityProof`
- placement metadata and telemetry schema
- the backend capabilities used by validation

A failed compile stores one precise reason and does not publish an executable
program.

## Schedule And Capacity

Program packet identity is `u32`. Schedule projection, workspace reservation,
and compilation must agree on partition count, packet scratch, fold graph, and
worker-stat capacities. A no-growth request succeeds only when
`ReserveWorkspace` has made every required capacity resident and the final
proof is satisfied.

`ResetWorkspace` clears live state while retaining vector capacity. Repeated
preparation may rebuild values but must not allocate after the checked
reservation boundary.

## Physical Tile Plan

`KernelProgramPhysicalTilePolicy` may split scheduled partitions into stable
physical tiles. Tile selection is derived only from packet count, execution
width, alignment, and the declared policy. It preserves packet identity and
uses static round-robin tile assignment. Telemetry reports whether physical
tiling ran and the exact tile units, count, and assignment.

## Tile Phase Description

`TilePhaseDescription` is the value contract used by Node graph compilation
to describe phase identity, tile count, deterministic ascending order, and
scratch/output/queue/task capacity requirements.

`ValidateTilePhaseDescription`, `TilePhaseRequiredCapacity`,
`AdmitTilePhase`, and `TilePhaseTileAt` are pure admission and indexing
helpers. They reject invalid identifiers, zero tile counts, non-power-of-two
alignment, arithmetic overflow, insufficient capacity, and out-of-range tile
indices with stable reasons.

## Execution

`executor(workspace, backend, workers, alignment, tile_policy)` validates one
explicit execution context. `Executor::prepare(space)` resets and reserves
the workspace, compiles a `KernelProgram`, and returns a generation-bound
`PreparedEach`. `PreparedEach::run` calls `RunPreparedProgram` through
the supplied backend.

`par(...)` obtains the same explicit executor inputs from the installed
`ParallelRuntimeProvider`. Provider scope and lifetime are owned by
`ScopedParallelRuntimeProvider`; missing or invalid provider state fails
closed.

`ComputeTileExecutor` is the buffer-oriented CPU Compute worker executor. It
reuses `CompileKernelProgram`, the physical tile plan, and
`RunPreparedProgram`. Preparation reserves all workspace, failure-slot, and
worker accounting storage. `run`, `run_with`, and prepared run instances
use the frozen plan without selecting another scheduler.

The implementation is owned by the focused leaves under
`src/program/compute/tile/`. `prepare`, sync `run`, async submit/finish,
callback work, and result projection have one compiled owner each. Sync and
async execution both reset one `Context` through
`Begin(...)` and both close through `Project(...)`; neither path carries a
second failure scan, worker-count formula, tail formula, or result vocabulary.

## Monitoring

`execution_report(workspace)` and `execution_report(executor)` project the
last Kernel execution without allocation or backend work. Reports expose
packet, partition, worker, tile, timing, and physical-tile evidence. A report
is observed only after program activity or a recorded execution reason.

## Verification

Focused contracts live under:

- `/kernel/tests/contract/program/`
- `/kernel/tests/contract/program/compute/tile.cpp`
- `/kernel/tests/contract/workspace/`
- `/kernel/tests/contract/dispatch/`

Use the narrow Kernel contract routes first, then `tools/check/run`. Release
surface changes additionally require `tools/release/run` and the external
package consumer.
