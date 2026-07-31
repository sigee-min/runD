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

Unsigned sum arithmetic wraps at the selected element width; min/max compare
declared unsigned values. There is no reduction, atomic accumulation,
schedule-dependent write, or floating-point authority.
Every semantic output element has one writer and reads only the frozen input
resident buffer, so CPU, Metal, and Vulkan may choose different physical lane
grouping while preserving the same output bits.

Input and output buffers must exactly match the planned element width and
`element_count`. Resident Stencil runs do not stage host input and do not
download output implicitly; only explicit upload/download calls affect
user-facing transfer byte counters. `RunAccelKernel(...)` reports Stencil pass
count from the frozen kernel plan.
