# Kernel Contracts

Stable `/kernel` behavior lives here. Architecture routing stays in
[Platform](../architecture/platform.md).

## Contents

| Page | Owns |
| --- | --- |
| [Checked](./checked.md) | Exact unsigned 64-bit add, subtract, multiply, overflow query, output publication, and ceiling arithmetic. |
| [Program](./program.md) | Kernel program compile request, schedule inputs, compile output, tile plan, tile phase description/admission, executor preparation, and execution evidence. |
| [Accel](./compute.md) | Kernel `ComputeIR`, `ComputeMap`, `ComputePlan`, deterministic lowering, backend handoff, frozen CPU SIMD caps, and Compute telemetry truth fields. |
| [Skeleton](./skeleton.md) | Public skeleton execution model, callback boundary, index space, views, partition alignment, and evidence. |
| [Workspace](./workspace.md) | Workspace memory ownership, dispatch handoff, worker backend truth, and release accounting. |
| [Reduction](./reduction.md) | Fold slots, fold graph, fold value domain, and strict floating-point reduction behavior. |

## Rule

Contract pages describe kernel behavior only and must name the public surface
and verification surface that prove them.
