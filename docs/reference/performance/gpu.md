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

## Admitted Observations

The checked M4 Pro profile contains measurements for three different cost
boundaries. They must not be read as one portable CPU-versus-GPU ranking.

### Resident Creation and Input Upload

The resident-setup timer encloses only `Program::resident(input)`. Program
compilation happens before the timer. `Job::run()` and the validation read
happen after it. The resulting medians therefore measure resident job
creation and the path's initial input copy or upload, not kernel execution:

| Path | `Program::resident(input)` median, `N = 1,048,576` |
| --- | ---: |
| CPU | 86.542 us |
| Metal | 477.500 us |
| Vulkan/MoltenVK | 561.000 us |

These values are setup-regression evidence for the exact source, input,
host, and driver path. They do not show CPU or GPU compute throughput.

### Warm Resident Execution

The workload timer creates the resident job and completes one validated
warmup before sampling. Each sample then times only `Job::run()`. After the
samples, the suite validates the result and requires its graph and output
hashes to match the CPU reference:

| Workload, `N = 262,144` | CPU | Metal | Vulkan/MoltenVK |
| --- | ---: | ---: | ---: |
| Dense expensive map | 324.959 us | 123.666 us | 190.542 us |
| Dense sort | 4,494.875 us | 342.458 us | 1,566.584 us |

For this exact profile, CPU median divided by path median was 2.628 for Metal
and 1.705 for Vulkan/MoltenVK on the dense expensive map. The corresponding
dense-sort ratios were 13.125 and 2.869. These are warm resident
`Job::run()` wall-time ratios for the named workload and host, not end-to-end
speedups or promises for another device, driver, data shape, or backend.

### Submission Amortization

For 64 independent jobs of 64 elements, one batch reduced the observed wall
median from 7,347.062 us to 227.645 us on Metal and from 9,136.105 us to
583.146 us on the Vulkan/MoltenVK path. Serial median divided by Batch median
is 32.274 and 15.667, respectively, for this exact workload. These values are
evidence for the structural benefit of one submission boundary, not
universal speedup claims. The serial wall path also includes per-job stats
observation and host-loop work, so these ratios must not be presented as pure
kernel or pure submission speedups.

### Recurrence Retention Hard Cut

The focused current-source M4 Pro recurrence measurement uses 4,096 `I32`
elements, 256 exact `x + 1` transitions, and twelve balanced samples. It keeps
typed observation outside timing and independently pairs serial/terminal and
terminal/history routes.

| Metal recurrence route | Submits / dispatches | Median |
| --- | ---: | ---: |
| 256 separately submitted Program runs | 256 / 256 | 29,502.375 us |
| Terminal-only `write_final` | 1 / 1 | 107.792 us |
| Paired terminal comparator for history | 1 / 1 | 101.749 us |
| Lossless iteration-major `write_each` | 1 / 1 | 139.688 us |

For this workload, serial median divided by terminal-only median is 273.697.
History median divided by its independently paired terminal median is 1.373.
Both fused routes report one logical step, the authored 255 barriers, a fully
verified prefix, exact output/history contents, and zero warm compilation,
allocation, descriptor, upload, download, and round-trip counters. The history
cost is mandatory device storage of all 256 states; it does not restore 256
host submissions or descriptor walks. These are focused observations for this
source, workload, and device rather than a portable speedup guarantee.

### Nested Aggregate Hard Cut

The focused current-source M4 Pro window-repeat measurement fixes
`Max = 516096`, `Tile = 1024`, `K = 504`, and `N = 64`, warms both routes, and
uses twelve balanced samples. It records the architectural progression for the
same nested result:

| Metal nested path | Physical shape | Median |
| --- | --- | ---: |
| Canonical transducer before the aggregate cut | 5,042 dispatches, 1,014 controls | 13,104.355 us |
| Ordered one-threadgroup aggregate | 1 dispatch, 1 control | 5,220.500 us |
| Parallel tile aggregate plus ordered finalize | 2 dispatches, 1 control | 369.959 us |

The shipped two-command path is 35.421 times faster than the canonical
transducer observation and 14.111 times faster than the intermediate ordered
aggregate for this exact workload. It retains one submission, exact result
parity, and zero warm allocation, upload, download, rebinding, and fallback
evidence. The speedup comes from removing the expanded occurrence/control
stream, evaluating independent exact-wide tile sums in parallel, reusing two
plan-owned Seed ranges as an `8K` low/status SoA, and keeping a single
authored-order finalize/commit authority. These are focused current-source
development observations, not installed baseline rows or a portable guarantee
for another recurrence, GPU, compiler, or driver.

### Input-Sealed Repetition Throughput Hard Cut

The focused M4 Pro measurement uses the same exact aggregate workload and
`sealed_repetitions<256>()`. A separate ordinary Pipeline is actually run 256
times inside each repeated timed interval. Three independent twelve-sample
AB/BA comparisons isolate serial versus single ordinary nested, single ordinary
nested versus sealed, and repeated ordinary nested versus sealed. Each ratio is
computed only from its own alternating pair. An untimed AB/BA preconditioning
pair precedes each phase and ends on its first recorded route, so the long
serial path cannot be an immediate predecessor in either sealed comparison.
The report retains the six paired physical medians, the
single-execution latency ratio, `T_sealed / 256`, and the directly measured
`T_256_ordinary / T_sealed` together. The latter is input-sealed repetition
throughput and must never be relabeled as single-tick latency.

An admissible row has one submission and two physical dispatches on the single
ordinary and sealed routes, 256 submissions and 512 dispatches in the measured
ordinary repeated interval, identical result bits, one sealed publication, 255
coalesced repetitions, and zero warm allocation, upload, download, rebinding,
and fallback evidence. The result applies only to the cold-proved input-sealed
contract: caller-owned read and write footprints are exactly disjoint, no
transactional state exists, and no intermediate tick is observed. Changing
active queues, recurrent state, per-tick checkpoints, or host inputs are outside
the result. No sealed-repetition number is promoted to the checked release
baseline by this contract change; focused raw evidence remains a development
observation until the release measurement route admits it.

All Vulkan observations above exercise runD's Vulkan lowering through
MoltenVK on Apple hardware. They are not native Vulkan throughput evidence.

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
