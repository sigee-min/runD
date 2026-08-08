# Accel Stencil Contract

Node owns resident backend execution for kernel-planned `Stencil` graph
steps. Kernel-owned descriptor, hash, pure planning, CPU reference, boundary
policy, and stable reason vocabulary stay in
[`/kernel/docs/contracts/compute.md`](../../../../kernel/docs/contracts/compute.md).

## Authority

Public support surface:

- `/node/include/node/accel/context.hpp`

Implementation authority:

- `/node/src/accel/stencil.hpp`
- `/node/src/accel/stencil/model.hpp` as the sole host parameter ABI owner
- `/node/src/accel/stencil/shape.{hpp,cpp}` as the sole resident-shape owner
- `/node/src/accel/primitive/shape.hpp`
- `/node/src/accel/cpu/stencil.cpp`
- `/node/src/accel/metal/stencil*`
- `/node/src/accel/vulkan/stencil*`
- `/node/src/accel/collective*`
- `/node/src/accel/graph.cpp`
- `/node/src/accel/graph/collective/{bindings.hpp,defaults.cpp,desc.cpp,kind.cpp}`
- `/node/src/accel/kernel/bindings/stencil.cpp`
- `/node/src/accel/kernel/plan/{compute,count,step}.cpp`
- `/node/src/accel/kernel/backend/run.cpp` for the canonical bound-step view

Verification authority:

- `/node/tests/contract/accel/kernel/stencil.cpp`
- `/node/tests/contract/accel/kernel/stencil/match/`
- `/node/tests/contract/accel/kernel/stencil/local.hpp`
- `/node/tests/contract/accel/kernel/primitive/local.hpp`

## Contract

`CompileAccelKernel(context, graph)` admits `NodeKind::Stencil`
only when the node carries `AccelGraphNode::stencil`,
`PlanStencil(stencil).ok`, a primitive hash equal to
`HashStencil(stencil)`, default Sort descriptor, default non-stencil
collective descriptors, matching `element_count`, and exactly two bindings in
role order: `(read input, write output)`.

Stencil execution supports `Sum`, `Min`, and `Max` with clamp
boundary over u32 or u64 elements and radius values in `[1, element_count]`:

```text
left(i, d)  = i < d ? 0 : i - d
right(i, d) = i + d >= element_count ? element_count - 1 : i + d
output[i] = input[i] + sum(input[left(i, d)] + input[right(i, d)])
            for d in 1..radius
min/max output[i] = extremum over the same clamped window
```

Sum arithmetic wraps modulo the selected element width for every domain;
min/max compare the declared signed or unsigned domain interpretation. There
is no reduction, atomic accumulation,
schedule-dependent write, or floating-point authority.
Every semantic output element has one writer and reads only the frozen input
resident buffer, so CPU, Metal, and Vulkan may choose different physical lane
grouping while preserving the same output bits.

The admitted input and output resident spans must not overlap. They may share
one physical owner only when their exact byte spans are disjoint. This keeps
the input frozen for the complete dispatch and prevents CPU update order or GPU
inter-thread races from becoming Stencil semantics.

## Resident GPU Shape

`/node/src/accel/stencil/shape.hpp` is the single physical-shape authority for
both resident GPU backends. It fixes workgroups at `W = 256` lanes, a shared
halo radius cap `R = 8`, and shared storage for `W + 2R = 272` elements. Metal
dispatches complete threadgroups, including the final partial semantic group;
Vulkan already has that dispatch shape. A lane outside the semantic tail may
return only after every lane in its group has crossed the shared-memory
barrier.

For element count `N`, a group beginning at `B` with `A` active semantic lanes
ends at `E = B + A`. The direct kernel performs

```text
D(A, r) = A(2r + 1)
```

global input-element reads. For a shared-path candidate, define the distinct
left and right halo inputs as

```text
L = min(r, B)
Q = min(r, N - E)
```

The GPU kernels load every element in the expanded input union exactly once:

```text
H(N, B, A, r) = A + L + Q
               = |[max(0, B-r), min(N, E+r))|
```

Equivalently, with clamped-halo fractions `bL = 1 - L/r` and
`bR = 1 - Q/r`, this is `H = A + r(2 - bL - bR)`. An interior full group has
`bL=bR=0`; a fully clamped first or last side has the corresponding value one.
The center lane already loading the first or last endpoint fans that register
out to every clamped shared-halo slot. If only part of the following halo is
resident, the lane loading its final distinct endpoint fans that register out
to the remaining clamped slots. Clamp duplicates therefore cause no additional
global read.

For every admitted `1 <= r <= R`, `H < D`: when `A >= 2`,
`H <= A + 2r < A + 2Ar = D`; when `A = 1`, the group is a domain boundary, so
at least one halo is clamped and `L + Q < 2r`. The generated kernel therefore
selects the shared path directly from `r <= R`; it carries no per-group traffic
arithmetic or redundant `H < D` branch. An interior full group changes radius
one from `768` global reads to `258`, and radius eight from `4352` to `272`; a
256-element domain needs only its 256 distinct inputs for either boundary.
This is an exact structural input-read lower bound, not a hardware transaction
or wall-time claim. Radius values above eight retain the direct semantic
fallback in the same compiled function; no asymptotic optimality claim is made
for that fallback.

Center and distinct halo loads are contiguous. Each lane keeps its accumulator
private, shared-memory reads preserve the original increasing-distance order,
sum still wraps at the declared width, and min/max still compare the declared
signed or unsigned interpretation. Every lane reaches the barrier
before an inactive tail lane may return.

The maximum declared shared allocation is

```text
u32: 272 * 4 = 1088 bytes
u64: 272 * 8 = 2176 bytes
```

Vulkan pipeline acquisition verifies the selected device's reported
`maxComputeSharedMemorySize`, 256-invocation capacity, and first-dimension
workgroup capacity against that shape; 2176 bytes is also below the Vulkan
Core 16 KiB minimum. Metal pipeline acquisition verifies 256-thread capacity
and rejects a compiler-reported static threadgroup allocation above the
selected device's `maxThreadgroupMemoryLength`. These checks are pipeline
admission, not an assumed device schedule.

Both shaders derive the logical index from `group_base + local_lane`. Metal
keeps that index 64-bit and rejects `G = ceil(N/W) > UINT32_MAX` before its
threadgroup identifier narrows. Vulkan's storage-buffer index is 32-bit, so
pipeline acquisition, resource preparation, and command encoding all require
`N <= UINT32_MAX`; only then may the overflow-safe 64-bit group computation
narrow to the proven index. Vulkan additionally rejects
`G > maxComputeWorkGroupCount[0]`. Stencil count and radius remain runtime
parameters. Vulkan's pseudo artifact identity normalizes both fields to zero,
so equal operation, element width, signed-extrema mode, and source text reuse
one compiled pipeline across count, shared-path radius, and direct-fallback
radius changes.

Input and output buffers must exactly match the planned element width and
`element_count`. Resident Stencil runs do not stage host input and do not
download output implicitly; only explicit upload/download calls affect
user-facing transfer byte counters. `RunAccelKernel(...)` reports Stencil pass
count from the frozen kernel plan.
