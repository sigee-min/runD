# Accel Histogram Contract

Node owns resident backend execution for kernel-planned `Histogram` graph
steps. Kernel-owned descriptor, hash, pure planning, CPU reference, count-width
law, and stable reason vocabulary stay in
[`/kernel/docs/contracts/compute.md`](../../../../kernel/docs/contracts/compute.md).

## Authority

Public support surface:

- `/node/include/node/accel/context.hpp`

Implementation authority:

- `/node/src/accel/histogram.hpp`
- `/node/src/accel/histogram/model.hpp` as the sole host parameter ABI owner
- `/node/src/accel/histogram/shape.{hpp,cpp}` as the sole resident-shape owner
- `/node/src/accel/primitive/shape.hpp`
- `/node/src/accel/cpu/histogram.cpp`
- `/node/src/accel/metal/histogram*`
- `/node/src/accel/vulkan/histogram*`
- `/node/src/accel/graph.cpp`
- `/node/src/accel/graph/collective/{bindings.hpp,defaults.cpp,desc.cpp,kind.cpp}`
- `/node/src/accel/kernel/bindings/histogram.cpp`
- `/node/src/accel/kernel/plan/{compute,count,step}.cpp`
- `/node/src/accel/kernel/backend/run.cpp` for the canonical bound-step view

Verification authority:

- `/node/tests/contract/accel/kernel/histogram.cpp`
- `/node/tests/contract/accel/kernel/histogram/match.cpp`
- `/node/tests/contract/accel/kernel/histogram/local.hpp`
- `/node/tests/contract/accel/kernel/primitive/local.hpp`

## Contract

`CompileAccelKernel(context, graph)` admits `NodeKind::Histogram`
only when the node carries `AccelGraphNode::histogram`,
`PlanHistogram(histogram).ok`, a primitive hash equal to
`HashHistogram(histogram)`, default Sort descriptor, default
non-histogram collective descriptors, matching `element_count`, and exactly two
bindings in role order: `(read bin_indices, write counts)`.

Histogram execution supports u32 bin indices and u32 counts. The
input buffer count must exactly match `HistogramDesc::element_count`, and
the output buffer count must exactly match `HistogramDesc::bin_count`.
Backend execution additionally rejects descriptor/plan pairs whose two-pass
clear-and-count shape cannot fit the backend ABI.

The semantic result is the same as the kernel CPU reference: every output count
is cleared, then each input bin index increments exactly one output bin. Bin
index formulas are not owned by Histogram; callers compute bin indices with
map-local formulas or other graph steps before this primitive. Backend timing,
command scheduling, pipeline cache state, thread arrival order, or atomic
operation interleaving is not result authority.

CPU returns the kernel reference result and invalid-bin reason directly. Metal
and Vulkan use private status storage and encode clear before count in command
order. Status storage is a fail-closed diagnostic channel only; it is not
semantic output and not an accumulator visible to callers. On invalid bin index
or malformed shape, execution returns an exact histogram reason and callers must
not consume the output count buffer as a semantic result.

`RunAccelKernel(...)` reports histogram pass count from the frozen kernel plan.
Resident Histogram runs do not stage host input and do not download output
implicitly; only explicit upload/download calls affect user-facing transfer
byte counters.

## Metal count ownership

Cold preparation selects the count executable using the one
`MetalHistogramLocal(bin_count)` predicate in `metal/histogram/local.hpp`.
At most 256 bins use a threadgroup-local histogram; larger shapes use direct
device counters with uniform-SIMD-cohort aggregation. The source recipe and
named count-pipeline identities distinguish these modes. Both consume the
same parameter/binding ABI and retain exactly the existing clear/count passes.
The clear pipeline is shared rather than recompiled for a second count mode.
The generic source is preprocessed without the local array or its barriers.

For a local histogram, preparation/encoding admits

```
G = min(ceil(N / 256), 1024)
```

full physical threadgroups. Each thread walks `gid + k * threads_per_grid`;
the native grid-size builtin is the stride authority. These disjoint walks
cover every input once, with adjacent threads reading adjacent elements.
Every group initializes the active bins of its declared 256-entry U32 atomic
array, synchronizes,
counts only valid bin indices, synchronizes again, then adds each nonzero
local bin count to its device counter once. Padding threads participate in both
barriers. Invalid bins publish the same error status while the complete group
continues through the barriers.

For each bin, the device-atomic count is at most `G`, independent of how many
inputs collide there. Across all bins it is at most `G * B`; input reads and
local atomic increments remain `N`. The 1 KiB declared threadgroup array is a
temporary locality tradeoff, not additional retained device scratch. Histogram
admission bounds `N <= UINT32_MAX`, so every local count and every intermediate
nonnegative device count is at most `N`; grouping cannot introduce overflow.

For a large histogram, a live SIMD cohort whose valid indices all name the
same bin issues one atomic add of its active-lane count. A broadcast from the first active lane and SIMD vote
prove equality; prefix/sum operations elect one active lane and count the
cohort, including partial tails and lanes surviving invalid-bin rejection.
Mixed cohorts keep one device increment per valid input. This preserves exact
integer counts without assuming a fixed hardware SIMD width, allocating a
global partial matrix, adding passes, or changing failure reasons.

Tests cover the 256/257-bin boundary, one-bin contention, mixed/uniform SIMD
cohorts, the capped-grid stride boundary, partial tails, invalid bins at both
ends, and repeated clear/count runs with all bins checked against a host oracle.
The source and atomic bounds above are algorithm evidence; measured latency
and host-specific limitations belong to `/docs/reference/performance`.
