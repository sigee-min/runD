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

- `/node/tests/contract/compute/pipeline/view/gather.hpp` covers the one/two-pass
  boundary, warm reuse, and exact earliest invalid ordinal through the public
  Pipeline, including G=257 with the first invalid index in partial slot 256.
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

Native Gather execution consumes the Kernel indexed preflight plan followed
by `output[i] = values[indices[i]]`. Duplicate indices are allowed because
payload lanes write distinct destinations. Capacity up to 65,536 retains one
control dispatch; larger inputs use G chunk-validation groups followed by one
terminal control group. The same immutable pipeline and descriptor set serve
both shapes, with the frozen group count in the existing 24-byte parameter
ABI. API buffer barriers order partial minima before terminal publication.

Kernel owns the group bound, pass count, partial status tail and stride model
in [Indexed Native Preflight](../../../../kernel/docs/contracts/compute/primitives.md#indexed-native-preflight).
Metal and Vulkan encode exactly that plan; neither owns a second threshold or
scratch formula. CPU returns the reference reason directly. No backend timing,
cache state or workgroup arrival order defines semantic output or first error.

The private status pair publishes a reason and exact first invalid ordinal.
Logical-count overflow takes precedence and skips index reads; otherwise the
shared minimum makes first-invalid selection independent of lane scheduling.
Only lane zero of the terminal control dispatch publishes status and indirect
arguments. Zero logical count publishes a zero-width payload dispatch. If any index is greater
than or equal to `source_count`, both payload dispatch width and output
mutation remain zero, and callers must not consume the output buffer as a
semantic result.

`RunAccelKernel(...)` reports gather pass count from the frozen kernel plan.
Resident gather runs do not stage host input and do not download output
implicitly; only explicit upload/download calls affect user-facing transfer
byte counters.
