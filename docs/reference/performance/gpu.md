# GPU Workload Sizing

This page owns the user-facing interpretation of runD Compute performance.
The measurement schema, admission rules, and frozen observations remain owned
by [Method](./method.md) and [`baseline.tsv`](./baseline.tsv).

## Element Count Is Not Work

One million elements is a large data set, but it is not necessarily a large
GPU workload. The relevant quantity is the useful work performed per byte
moved and per submission:

```text
T_cpu = N C_cpu

T_gpu = T_submit + T_sync + T_transfer + N C_gpu
```

`N` is the element count, `C_cpu` and `C_gpu` are the per-element execution
costs on the selected paths, and the other terms are fixed or transfer costs.
Offload can win only when `C_cpu > C_gpu` and

```text
N > (T_submit + T_sync + T_transfer) / (C_cpu - C_gpu).
```

Resident execution removes warm-path allocation, upload, and download, but it
does not remove submission, synchronization, dispatch, or the memory traffic
inside the kernel.

Arithmetic intensity makes the same boundary visible:

```text
I = useful operations / bytes moved

throughput <= min(compute throughput, I * memory bandwidth).
```

A light element-wise map over 32-bit values may read and write about eight
bytes per element while performing only a few integer operations. Its element
count can be high while its arithmetic intensity remains low, so memory
traffic and fixed launch costs can dominate.

## Structural Lower Bounds

Several GPU paths admit exact source- and dispatch-shape proofs independent of
wall time. The owning backend contracts remain the authority; this page states
only the performance interpretation.

For canonical Vulkan Map width `W = 256`, an active domain of `N` elements
publishes exactly

```text
G_map(N) = N == 0 ? 0 : 1 + floor((N - 1) / 256)
```

workgroups. Every element is owned by one invocation and a workgroup exposes at
most 256 invocations, so `ceil(N / 256)` is the minimum legal group count for
that fixed shader width. Both control and body source recipes consume
`kVulkanMapWidth` directly.

Metal Map classifies a binding as word-addressable exactly when

```text
A(offset, stride) := offset mod 4 == 0 and stride mod 4 == 0.
```

Every Map access has an integer logical index `k`: direct access uses `b + g`
for exact window begin `b` and lane `g`, uniform access uses zero, and indexed
access uses its checked runtime index. Its byte address is
`offset + k*stride`, which is four-byte aligned whenever `A` holds. The
specialized source therefore replaces four byte accesses with one `uint`
access for a 32-bit value and eight byte accesses with two adjacent `uint`
accesses for a 64-bit value. Misaligned bindings retain the bytewise source.
These are generated-source memory-operation counts, not claims about device
transactions, cache lines, or measured latency.

The Range planner derives a physical workgroup width `W` from `{64, 128, 256}`
and a shared capacity `C` from the selected device's workgroup and shared-memory
limits. The exact selection policy and pipeline-identity contract are owned by
the [Accel Range contract](../../../node/docs/contracts/accel/range-aggregate.md).
For a selected shared shape and an admitted `1 <= r <= C`, let a physical group
cover `A` active outputs beginning at `B` in a domain of `N` elements and end at
`E = B + A`. The direct candidate reads

```text
D = A(2r + 1)
```

input elements, while the shared path reads exactly the distinct expanded
input union

```text
H = A + min(r, B) + min(r, N - E)
  = |[max(0, B-r), min(N, E+r))|.
```

Any correct single-group evaluation needs every element in that union, so `H`
is the global-input-read lower bound for the group. The boundary loader reaches
it by fanning lane-private loaded endpoint values into clamped halo slots. For
every selected shared candidate, `H < D`, so its generated shader executes the
shared path directly without a runtime traffic branch. A selected direct-only
candidate declares no shared array and executes no workgroup barrier. These are
exact source-level input-read bounds, not claims about physical memory
transactions, occupancy, or measured latency.

## Vulkan Executable Construction

Vulkan executable acquisition compiles each cache miss synchronously under the
Device adapter. The SPIR-V shader module is a construction-scoped owner:
`vkCreateComputePipelines` uses its handle, and runD destroys the module
before publishing the cached executable. For `P` distinct executable misses
with module footprints `S[p]`, the live module model is

```text
module_handles <= 1
module_live_bytes <= max(S[p])
module_live_bytes = 0 between acquisitions and during warm execution
```

The executable cache retains the pipeline, layouts, descriptors, exact
artifact identity, and source needed for collision-safe reuse. Its capacity is
therefore independent of shader-module handle lifetime. This is a native
resource-lifetime bound, not a throughput measurement; compile latency and
pipeline-native memory remain device and driver observations.

## Product Evidence Boundary

The checked Compute profile is generated by the installed public SDK over
`map -> window/pool/rolling -> filter -> reduce` Pipelines. It records the
planner-selected candidate and physical dispatch count rather than substituting
a forced shader variant as the representative performance path.

Cold `first_result_us` includes Flow authoring, compilation, upload, Pipeline
preparation, execution, and one terminal read. Warm p50 and p95 contain only
sixty consecutive executions after one fixed sixty-run conditioning block on
the already prepared Pipeline; one terminal read follows the samples. Bounded
rolling reuses one capacity at active counts
`0`, `257`, `131072`, and `262144` with a poisoned inactive tail. CPU, Metal,
and Vulkan results, graph hashes, and output hashes must match the independent
oracle before timing is admitted.

The profile retains submissions and dispatches as separate fields. A Pipeline
may execute several Range stages in one queue submission, and a terminal
Vulkan transfer may add one submission after execution. Peak retained memory,
logical and backing scratch, transfer bytes, compile/cache evidence, and warm
allocation counters remain exact semantic evidence. The only checked timing
values are first-result latency and warm p50/p95 for each declared shape.

The resulting CPU/Metal/Vulkan comparisons are crossover samples over those
counts and window shapes. They do not establish a device-independent winner,
physical transaction count, register allocation, occupancy, spill behavior, or
universal wall-clock optimum. Current values and their host/driver provenance
live only in the admitted baseline and evidence packets owned by the
[Performance Method](./method.md).

## Execution Shape

Each product surface has the following graph-invariant cost boundary:

| Workload | Product surface | Repeated cost absent from the boundary |
| --- | --- | --- |
| One result needed on the host | `collect()` | None; this is the complete convenience boundary. |
| One Program, changing input | `Program::run()` | Repeated graph construction and compilation. |
| Repeated device-resident state | `Program::resident()` | Warm allocation and host transfer. |
| Several dependent stages | `Pipeline` | Intermediate host materialization and extra submission boundaries. |
| Many independent jobs | `Batch` | Repeated submission and synchronization boundaries. |

For `K` element-wise stages over `N` elements of width `E`, fusion can remove
up to

```text
2 (K - 1) N E
```

bytes of intermediate read/write traffic. This is a memory-traffic model. A
wall-clock improvement still requires a checked measurement because register
pressure, occupancy, barriers, and backend compilation can change the realized
cost.

## Determinism Is Fixed

Performance transformations may change physical placement and command shape,
but they must preserve:

- canonical graph and numeric-policy identity;
- stable ordering and bounded logical counts;
- Fixed width, rounding, overflow, and explicit quantization;
- output bits and output hash;
- the single CPU, Metal, and Vulkan lowering contract.

Batching does not reorder jobs. Fusion does not reassociate a reduction.
Resident execution does not retain hidden result state. A measurement is
admissible only after the matching graph identity and output have been
verified.

## Measure the Boundary You Ship

Use the installed Release SDK and measure the same surface used by the
application:

```sh
tools/measure/compute/run --resident metal
tools/measure/compute/run --resident vulkan
tools/measure/compute/run --pipeline metal
tools/measure/compute/run --pipeline vulkan
```

Interpret the result with [Method](./method.md). A passing upper bound is
regression evidence, not a speedup claim, and a different device or driver
requires its own admitted host profile.
