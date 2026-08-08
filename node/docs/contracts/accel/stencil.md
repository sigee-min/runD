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
both resident GPU backends. For each admitted descriptor and device it selects
one width from

```text
W in {64, 128, 256}
```

and ranks the admitted shared-halo and direct-only shapes. A shared shape has
capacity `C > 0`; a direct-only shape has `C = 0`. Every admitted width always
contributes its direct shape, even when its shared shape is also usable. The
first backend-supported `(W, C)` in that finite ranking is frozen in the
resident resource; it does not change the Kernel descriptor, `HashStencil`,
plan, or CPU reference semantics.

Shared capacity is derived only from integer device limits. Let `e` be the
element width in bytes, `L` the backend's shared-memory limit in bytes, and
`q = 4` the fixed shared-memory reserve. Define

```text
S(L, e, q) = floor(floor(L / q) / e)
C(W, e, L, q) = min(W, floor((S(L, e, q) - W) / 2))
                 = min(W, floor((floor(L / q / e) - W) / 2))
```

The capacity expression is evaluated only when `q > 0`, `e in {4, 8}`, `W`
fits the device's workgroup limits, and `S > W`; otherwise the capacity is
zero. Equivalently, the guarded numerator is
`floor(L / q / e) - W`. The `q = 4` reserve is a deterministic resource budget
used by shape selection. It is not evidence that the driver will place four
workgroups concurrently, and it makes no physical occupancy guarantee.

The capacity is clamped to `C <= W`, which is the current loader invariant:
the lanes of one workgroup can cover every distinct input in either halo. A
shared candidate is usable only when `radius <= C`. A direct-only candidate
has `C = 0`, declares no shared array, executes no shared load, and contains no
workgroup barrier.

Metal dispatches complete threadgroups, including the final partial semantic
group; Vulkan uses the same complete-group shape. On a shared path, a lane
outside the semantic tail may return only after every lane in its group has
crossed the shared-memory barrier.

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

The shared loader reads every element in the expanded input union exactly
once:

```text
H(N, B, A, r) = A + L + Q
               = |[max(0, B-r), min(N, E+r))|
```

Equivalently, with clamped-halo fractions `bL = 1 - L/r` and
`bR = 1 - Q/r`, this is `H = A + r(2 - bL - bR)`. An interior full group has
`bL=bR=0`; a fully clamped first or last side has the corresponding value one.
The center lane already loading the first or last endpoint fans that
lane-private loaded value out to every clamped shared-halo slot. If only part
of the following halo is resident, the lane loading its final distinct
endpoint fans that lane-private loaded value out to the remaining clamped
slots. Clamp duplicates therefore cause no additional global read. This is a
source-level value-lifetime statement, not a claim about physical register
allocation.

For every selected shared shape with `1 <= r <= C`, `H < D`: when `A >= 2`,
`H <= A + 2r < A + 2Ar = D`; when `A = 1`, the group is a domain boundary, so
at least one halo is clamped and `L + Q < 2r`. The generated shared variant
therefore needs no per-group traffic comparison or redundant `H < D` branch.
This is an exact structural input-read lower bound for the selected shared
shape, not a hardware transaction or wall-time claim.

For a complete dispatch, let

```text
G(W, N) = ceil(N / W)
T(W, N) = N - (G - 1)W
```

where `T` is the active-lane count in the final group. The selector computes
the exact shared-path input-read total as

```text
R_shared = N                                      when G = 1
R_shared = N + (2G - 3)r + min(r, T)             when G > 1
```

and the direct path has `R_direct = N(2r + 1)`. Every usable shared candidate
has strictly fewer exact input reads than a direct candidate, so the selector
uses the following timing-free lexicographic integer order:

1. shared candidate before every direct candidate;
2. fewer exact shared input reads;
3. fewer launched lanes, `GW`;
4. fewer groups, `G`;
5. fewer declared shared bytes, `(W + 2C)e` for shared and zero for direct;
6. smaller width as the final deterministic tie-break.

The result is exact for this finite candidate set and declared resource model.
It is not a global algorithmic optimum, a wall-time ranking, or a prediction
of compiler register allocation, driver scheduling, cache transactions, or
physical occupancy.

Metal applies that same complete integer order to actual pipeline support. For
each candidate in order, a cached or newly compiled pipeline must belong to
the selected device, report at least `W` executable threads, and carry the
exact preparation-mode plus `(W, C)` pipeline label. A shared shape must report
at least `(W + 2C)e` static threadgroup bytes while satisfying the
overflow-safe check

```text
q * staticThreadgroupMemoryLength <= maxThreadgroupMemoryLength.
```

A direct pipeline must report zero static threadgroup bytes. Metal advances to
the next ranked candidate, including the direct candidates, only when an
otherwise valid compiled pipeline explicitly reports insufficient thread
capacity or an over-budget static allocation. Source construction, allocation,
library/function/pipeline creation, cache publication, device ownership,
identity, and under-reported static-allocation failures abort preparation; they
are not laundered into a lower-ranked shape. No branch consults compile
duration or execution timing. The first passing pipeline and its shape are
frozen together. Thus a device may advertise width 256 while a particular
compiled pipeline admits only width 128; that deterministically selects the
first width-128-or-smaller candidate in the common ranking instead of rejecting
an otherwise executable Stencil.

Candidate compilation uses explicit source-library and named-pipeline cache
transactions. A newly compiled library for a candidate rejected by compiled
resource assessment is not published. Once a valid PSO exists, source
publication returns exactly one of `Inserted`, `Existing`, or `Failed`; a
failed insertion leaves every prior LRU entry intact, while `Existing` returns
the canonical cached owner. Every constructed library and PSO contributes
exactly once to its compile counter and latency accumulator regardless of
publication disposition. If source publication succeeds but named-pipeline
publication fails, the source remains an adapter-cache entry. A retry must hit
that exact source owner and may rebuild only the missing PSO. Adapter-cache
residency is not manifest ownership: only a fully prepared candidate
contributes the manifest's one source dependency and one pipeline stage.
Rejected compile attempts remain sequential cold transients within the same
maximum-source envelope.

A Program-template immutable Stencil still owns one bare pipeline pointer.
Reuse therefore scans the same ranked list and accepts the one candidate whose
preparation-mode label, device, and reported limits match that pointer; it
neither guesses a shape nor compiles a replacement behind the immutable owner.

Center and distinct halo loads are contiguous. Each lane keeps its accumulator
private, shared-memory reads preserve the original increasing-distance order,
sum still wraps at the declared width, and min/max still compare the declared
signed or unsigned interpretation. Every lane reaches the barrier
before an inactive tail lane may return.

Vulkan shape admission uses the selected device's reported
`maxComputeSharedMemorySize`, invocation capacity, first-dimension workgroup
capacity, and first-dimension group-count limit. Metal uses the device's
threadgroup-width and `maxThreadgroupMemoryLength` limits, then applies the
compiled-pipeline fallback above. These checks are resource and pipeline
admission, not an assumed device schedule.

Both shaders derive the logical index from `group_base + local_lane`. Metal
keeps that index 64-bit and rejects `G = ceil(N/W) > UINT32_MAX` before its
threadgroup identifier narrows. Vulkan's storage-buffer index is 32-bit, so
pipeline acquisition, resource preparation, and command encoding all require
`N <= UINT32_MAX`; only then may the overflow-safe 64-bit group computation
narrow to the proven index. Vulkan additionally rejects
`G > maxComputeWorkGroupCount[0]`.

The selected width and capacity are stored in the resident resource and are
part of Metal and Vulkan source/pipeline identity. On the same backend, count
or radius changes may reuse a pipeline only when selection produces the same
`(W, C)` and the operation, element width, and signed-extrema variant also
match. A shape change is a different physical pipeline even when Kernel
semantics are otherwise equal.

Shared and direct variants preserve the same per-lane update order and must
produce exact output parity for every admitted operation and integer domain.
Metal and Vulkan outputs must each match the unchanged CPU reference and each
other bit for bit. Backend or variant selection cannot become a new semantic
hash authority.

The resident-backend maximum-shared contract enumerates the actual device
selection at `N = 515` from radius 256 down through radius 1. Because
`515 mod W = 3` for every legal `W`, every selected shape crosses workgroups
and has a three-lane tail. The first shared result must report `r = C`; the
test then prepares and executes that exact descriptor, requires the resident
resource to carry the same shared `(W, C)`, and compares every output with the
CPU reference. Its Program-template immutable retry must borrow the same
pipeline and shape without a compile or cache lookup. If every valid selection
in the enumeration is direct, the contract records a distinct verified
no-shared-capability disposition. A direct resource is never counted as
maximum-shared execution evidence.

Input and output buffers must exactly match the planned element width and
`element_count`. Resident Stencil runs do not stage host input and do not
download output implicitly; only explicit upload/download calls affect
user-facing transfer byte counters. `RunAccelKernel(...)` reports Stencil pass
count from the frozen kernel plan.
