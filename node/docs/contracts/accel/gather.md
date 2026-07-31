# Accel Gather Contract

Node owns resident backend execution for kernel-planned `ComputeGather` graph
steps. Kernel-owned descriptor, hash, pure planning, CPU reference, and stable
reason vocabulary stay in
[`/kernel/docs/contracts/compute.md`](../../../../kernel/docs/contracts/compute.md).

## Authority

Public support surface:

- `/node/include/node/accel/context.hpp`

Implementation authority:

- `/node/src/accel/gather.hpp`
- `/node/src/accel/gather/model.hpp` as the sole host parameter ABI owner
- `/node/src/accel/gather/shape.{hpp,cpp}` as the sole resident-shape owner
- `/node/src/accel/primitive/shape.hpp`
- `/node/src/accel/cpu/gather.cpp`
- `/node/src/accel/metal/gather*`
- `/node/src/accel/vulkan/gather*`
- `/node/src/accel/collective*`
- `/node/src/accel/graph.cpp`
- `/node/src/accel/kernel.cpp`
- `/node/src/accel/kernel/bindings/gather.cpp`
- `/node/src/accel/kernel/plan/{compute,count,step}.cpp`
- `/node/src/accel/kernel/backend/run.cpp` for the canonical bound-step view

Verification authority:

- `/node/tests/contract/accel/kernel/gather.cpp`
- `/node/tests/contract/accel/kernel/gather/match/`
- `/node/tests/contract/accel/kernel/gather/reject/`
- `/node/tests/contract/accel/kernel/gather/local.hpp`
- `/node/tests/contract/accel/kernel/primitive/local.hpp`

## Contract

`CompileAccelKernel(context, graph)` admits `NodeKind::Gather` only when
the node carries `AccelGraphNode::gather`, `PlanGather(gather).ok`, a
primitive hash equal to `HashGather(gather)`, default Sort descriptor,
default `AccelGraphNode::scan`, matching `element_count`, and exactly three
bindings in role order: `(read values, read indices, write output)`.

Values and output are `u32` or `u64` according to the kernel descriptor.
Indices are u32. The source buffer count must cover `source_count`, while the
index and output buffer counts must exactly match `element_count`. Backend
execution additionally rejects descriptor/plan pairs whose dispatch ABI cannot
represent the planned element count.

Native Gather execution is two deterministic passes: a failure-atomic control
preflight followed by the indirect payload pass
`output[i] = values[indices[i]]`. Duplicate indices are allowed because every
payload lane writes exactly one distinct output slot. No backend timing,
pipeline state, command scheduling, or cache state is semantic authority.

CPU returns the kernel reference reason directly. Metal and Vulkan launch one
256-lane workgroup for the control preflight. Lane `t` scans ordinals
`t, t + 256, ...`; the workgroup then performs an eight-stage shared-memory
minimum reduction over each lane's first invalid ordinal. Thus the critical
path is `O(ceil(N/256) + 8)`, total validation work remains `O(N)`, and the
implementation adds no dispatch, scratch allocation, payload copy, or command
submission relative to the existing two-pass plan. The kernel plan caps
`element_count` at `UINT32_MAX`; Vulkan therefore uses a U32 loop ordinal
with an explicit final-stride termination check so the increment cannot wrap.

The private status pair publishes a reason and exact first invalid ordinal.
Logical-count overflow takes precedence and skips index reads; otherwise the
shared minimum makes first-invalid selection independent of lane scheduling.
Only lane zero clears and publishes status and indirect arguments. Zero
logical count publishes a zero-width payload dispatch. If any index is greater
than or equal to `source_count`, both payload dispatch width and output
mutation remain zero, and callers must not consume the output buffer as a
semantic result.

`RunAccelKernel(...)` reports gather pass count from the frozen kernel plan.
Resident gather runs do not stage host input and do not download output
implicitly; only explicit upload/download calls affect user-facing transfer
byte counters.
