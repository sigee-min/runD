# Node Contracts

Stable `/node` behavior lives here. Kernel-owned behavior stays in
[`/kernel/docs`](../../../kernel/docs/README.md).

## Contents

| Page | Owns |
| --- | --- |
| [Accel](./accel.md) | Node-owned Accel backend observation/selection, direct SDK facade routing, runtime resources, resident support, frozen kernel caps handoff, fail-closed adapter reasons, checked CPU SIMD backend execution, and diagnostic fake backend boundary. |
| [Build Graph](./build/graph.md) | Production OBJECT ownership, component SCC condensation, focused link profiles, and installed archive invariants. |
| [Counter Arithmetic](./counter.md) | Domain-neutral unsigned evidence-counter saturation and its single internal implementation authority. |
| [Coroutine Task](./coroutine.md) | Native `Task<T>` frame, suspension, completion, and cancellation contracts. |
| [Compute Batch](./compute/batch.md) | Bounded one-submit accelerator batching, atomic Job admission, ordered completion, and shared evidence ownership. |
| [Compute Pipeline](./compute/pipeline.md) | Declaration-ordered dependent Program execution over resident Buffers, one prepared cross-Program plan, one nonempty GPU submit, typed claims and poison, explicit readback, and exact evidence laws. |
| [Compute Memory](./compute/memory.md) | Program/Job retained owner accounting, compact CPU runtime graph/SIMD-plan ownership, authenticated accelerator-token metadata, and exact allocation-free snapshot invariants. |
| [Accel Compact](./accel/compact.md) | Stable native compaction, direct rank law, bounded capacity status, reuse, and Vulkan `8 * ceil(N / 256) + 4` scratch. |
| [Accel Gather](./accel/gather.md) | Node-owned CPU/Metal/Vulkan execution for kernel-planned deterministic gather graph steps. |
| [Accel Reduce](./accel/reduce.md) | Node-owned CPU/Metal/Vulkan execution for kernel-planned deterministic reduce graph steps. |
| [Accel Scan](./accel/scan.md) | Observable prefix contract, deterministic native hierarchy, exact overflow, and Vulkan block/prefix/offset execution. |
| [Accel Segmented Scan](./accel/segmented/scan.md) | Cross-block carry, canonical overflow, cancellation, and bounded Vulkan dispatch. |
| [Accel Scatter](./accel/scatter.md) | Node-owned CPU/Metal/Vulkan execution for kernel-planned deterministic limited scatter graph steps. |
| [Accel Stencil](./accel/stencil.md) | Node-owned CPU/Metal/Vulkan execution for kernel-planned deterministic stencil graph steps. |
| [Host](./host.md) | Deterministic host API boundaries for random, timer, IO, env, input, and thread. |
| [Network](./net.md) | Byte-level `rund::net` surface, canonical address and socket identity, scheduler readiness, replay, limits, and runtime/domain semantic cut. |
| [Session](./runtime.md) | Session configuration, backend, lifecycle, actual-state snapshots, scope/result UX, trace, discovery, Compute integration, and replay evidence. |
| [Replay](./replay.md) | One-Session canonical input, record/replay/scenario scopes, checkpoint schema and lineage, persistence, and cost laws. |
| [Scheduler](./scheduler/README.md) | `task::spawn/join/scope/yield/sleep/channel/io`, task-worker lanes, deterministic scheduler evidence, and blocking primitive semantics. |
| [Storage](./storage.md) | Domain-neutral hierarchical allocated capacity, reservation, physical/allocated usage, refund, and report laws. |
| [Telemetry](./telemetry.md) | Session Compute/Replay event shape, Basic/Detail cost boundary, callback ordering, and `telemetry:detail` parity. |
| [Topology](./topology.md) | Node topology and resource evidence before projection into kernel inputs. |

Session-owned prepared-memory evidence is documented by
[Session](./runtime.md); [Scheduler](./scheduler/README.md) owns its recording
and non-interference boundary.

## Rule

Contract pages describe node behavior only, name their public and verification
surfaces, and link to kernel docs for kernel-owned semantics.
