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

## Admitted Observation

The checked M4 Pro profile freezes the following resident medians for one
light `N = 1,048,576` workload:

| Path | Median |
| --- | ---: |
| CPU | 86.542 us |
| Metal | 477.500 us |
| Vulkan through MoltenVK | 561.000 us |

These values prove a regression boundary for that exact host, workload,
source identity, output, and driver path. They do not establish a portable
CPU-versus-GPU ranking. In particular, the Vulkan row exercises runD's Vulkan
lowering through MoltenVK on Apple hardware; it is not native Vulkan
throughput evidence.

The same profile also shows why submission amortization matters. For 64
independent jobs of 64 elements, one batch reduced the observed wall median
from 7,347.062 us to 227.645 us on Metal and from 9,136.105 us to 583.146 us
on the Vulkan/MoltenVK path. Those observations are 32.274 and 15.667 times
the respective serial rates for this exact workload. They are evidence for
the structural benefit of one submission boundary, not universal speedup
claims.

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
