# Accel Scatter Contract

Node owns resident backend execution for kernel-planned `ComputeScatter` graph
steps. Kernel-owned descriptor, hash, pure planning, CPU reference, duplicate
write policy, and stable reason vocabulary stay in
[`/kernel/docs/contracts/compute.md`](../../../../kernel/docs/contracts/compute.md).

## Authority

Public support surface:

- `/node/include/node/accel/context.hpp`

Implementation authority:

- `/node/src/accel/scatter.hpp`
- `/node/src/accel/scatter/model.hpp` as the sole host parameter ABI owner
- `/node/src/accel/scatter/shape.{hpp,cpp}` as the sole resident-shape owner
- `/node/src/accel/primitive/shape.hpp`
- `/node/src/accel/cpu/scatter.cpp`
- `/node/src/accel/metal/scatter/` with Scatter Reduce ownership split under
  `reduce/{model,source,pipeline,prepare,encode,finish}`
- `/node/src/accel/vulkan/scatter/` with the matching Scatter Reduce ownership
  split under `reduce/{model,source,pipeline,prepare,encode,finish}`
- `/node/src/accel/collective*`
- `/node/src/accel/graph.cpp`
- `/node/src/accel/graph/collective/{bindings.hpp,defaults.cpp,desc.cpp,kind.cpp}`
- `/node/src/accel/kernel/bindings/scatter.cpp`
- `/node/src/accel/kernel/plan/{compute,count,step}.cpp`
- `/node/src/accel/kernel/backend/run.cpp` for the canonical bound-step view

Verification authority:

- `/node/tests/contract/accel/kernel/scatter.cpp`
- `/node/tests/contract/accel/kernel/scatter/match/`
- `/node/tests/contract/accel/kernel/scatter/reject/`
- `/node/tests/contract/accel/kernel/scatter/local.hpp`
- `/node/tests/contract/accel/kernel/primitive/local.hpp`

## Contract

`CompileAccelKernel(context, graph)` admits `NodeKind::Scatter` only when
the node carries a `AccelGraphNode::scatter` descriptor matching the resident
buffer shape, `PlanScatter(scatter).ok`, a primitive hash equal to
`HashScatter(scatter)`, default Sort descriptor, default scan/gather/reduce
descriptors, matching `element_count`, and exactly three bindings in role
order: `(read values, read indices, write output)`.

Values and output are `u32` or `u64` according to the kernel descriptor.
Indices are always u32. The values and index buffer counts must exactly match
`element_count`; the output buffer count must exactly match `output_count`.
`PlanScatter` rejects counts the shared diagnostic owner encoding or status
table cannot represent, so backends keep no second count admission boundary.
`PlanScatter` is also the only CPU duplicate-table capacity authority. Its
`scratch_slots` is the least power of two at least `2*element_count`; Node
accepts the complete canonical plan, sizes prepared scratch from it, and
passes it unchanged to execution. CPU runtime code does not carry a second
capacity formula.

Scatter execution is limited and deterministic:
`output[indices[i]] = values[i]`. Because Scatter is a partial writer, its
compiled run binding carries the canonical first-write reset for the complete
authored output View. Unreferenced output slots are therefore all-byte zero on
every cold or warm logical invocation; prior resident contents are not an
input. A strided View resets only its addressed lanes, not the borrowed Buffer
owner's intervening elements. The scatter kernel does not own a second clear:
the common prepared-run reset span executes immediately before the dispatch.
Duplicate target indices are forbidden and fail closed with
`compute_scatter_duplicate_index`; out-of-range indices fail closed with
`compute_scatter_index_out_of_range`. Callers may only consume output after an
`ok` evidence row.

CPU returns the kernel reference reason directly. Metal and Vulkan use private
status storage to prove failure without making arrival order semantic
authority. The status table records the earliest failing input index through
atomic minimum, while target slots record the minimum writer index for duplicate
detection. Those atomics guard rejection only; they do not combine semantic
output values, implement last-writer-wins behavior, or authorize a
schedule-dependent result. Successful rows have unique target indices, so every
semantic output write has exactly one writer. On Metal and Vulkan only the
first physical claimant writes a destination; later claimants update the
canonical owner and failure evidence but never write output. A rejected payload may
therefore reflect physical arrival and remains explicitly unconsumable, while
the execution itself has no duplicate-address storage race and the reported
failure is schedule-independent.

Resident scatter runs do not stage host input and do not download output
implicitly; only explicit upload/download calls affect user-facing transfer
byte counters.

Scatter Reduce native execution consumes one shared parameter ABI, one Kernel
parallel-fold predicate, the plan's `status_bytes`, `indirect_bytes`, and
`pass_count`, and one status decoder. Metal and Vulkan own only API-specific
allocation, binding, encoding, and completion transport; they do not repeat
numeric-order policy, workspace byte constants, dispatch count, or semantic
status meanings. Each backend keeps those API responsibilities in separate
translation units: `source` emits the shader, `pipeline` owns compilation and
cache identity, `prepare` owns resource lifetime and binding, `encode` owns the
ordered control-initialize-fold submission, and `finish` owns status,
telemetry, and dispatch accounting. The backend-local `model` is the sole
prepared-resource layout authority.
