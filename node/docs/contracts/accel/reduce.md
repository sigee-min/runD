# Accel Reduce Contract

Node owns resident backend execution for kernel-planned `ComputeReduce` graph
steps. Kernel-owned descriptor, hash, pure planning, CPU reference, overflow
law, deterministic plan shape, and stable reason vocabulary stay in
[`/kernel/docs/contracts/compute.md`](../../../../kernel/docs/contracts/compute.md).

## Authority

Public support surface:

- `/node/include/node/accel/context.hpp`

Implementation authority:

- `/node/src/accel/reduce/{metal,vulkan}.hpp` for backend-specific execution
  entrypoints
- `/node/src/accel/reduce/pass.hpp` for the shared GPU pass ABI only
- `/node/src/accel/reduce/shape.{hpp,cpp}` as the sole resident-shape owner
- `/kernel/include/kernel/program/compute/reduce/wide.hpp` for the one shared
  exact pair/wide source arithmetic used by Metal and Vulkan
- `/node/src/accel/primitive/shape.hpp`
- `/node/src/accel/cpu/reduce.cpp`
- `/node/src/accel/metal/reduce*`
- `/node/src/accel/vulkan/reduce*`
- `/node/src/accel/collective*`
- `/node/src/accel/graph.cpp`
- `/node/src/accel/graph/collective/{bindings.hpp,defaults.cpp,desc.cpp,kind.cpp}`
- `/node/src/accel/kernel/bindings/reduce.cpp`
- `/node/src/accel/kernel/plan/{compute,count,step}.cpp`
- `/node/src/accel/kernel/backend/run.cpp` for the canonical bound-step view

Verification authority:

- `/node/tests/contract/accel/kernel/reduce.cpp`
- `/node/tests/contract/accel/kernel/reduce/match/`
- `/node/tests/contract/accel/kernel/reduce/reject/`
- `/node/tests/contract/accel/kernel/reduce/local.hpp`
- `/node/tests/contract/accel/kernel/primitive/local.hpp`
- `/node/tests/contract/accel/kernel/fusion/`

## Contract

`CompileAccelKernel(context, graph)` admits `NodeKind::Reduce` only when
the node carries `AccelGraphNode::reduce`, `PlanReduce(reduce).ok`, a
primitive hash equal to `HashReduce(reduce)`, default Sort descriptor,
default `AccelGraphNode::scan` and `AccelGraphNode::gather`, matching
`element_count`, and exactly two bindings in role order:
`(read input, write output)`.

Reduce execution supports deterministic Sum, CountNonzero, Min, and
Max over declared 32-bit and 64-bit stored domains. The descriptor element
selects storage width; the planned Compute domain selects signed, unsigned, or
`Fixed<I,F>` interpretation. The input buffer count must exactly match
`ReduceDesc::element_count`, and the output buffer must contain at least one
element of the declared width. Backend execution additionally rejects
descriptor/plan pairs whose pass/block/grid shape cannot fit the backend ABI.

The semantic result follows the canonical ascending-input reduction law and
the declared numeric policy. `ReduceOp::Sum` fails closed with
`compute_reduce_sum_overflow` when the exact result cannot fit the declared
output domain. `ReduceOp::CountNonzero` maps each active input to `1` or `0` and
fails closed with `compute_reduce_count_overflow` if the exact count cannot fit
the output domain. Min and Max compare the declared domain without arithmetic
overflow.

Backend timing, command scheduling, pipeline cache state, thread arrival
order, and atomics are not result authority. These work and width bounds are
algorithm evidence, not a latency, throughput, or backend-ranking claim;
performance requires a current installed-Release measurement.

CPU returns the kernel reference result and overflow reason directly. Metal
and Vulkan execute `Sum` and `CountNonzero` as a fixed bounded hierarchy. The
first pass runs

```text
G = min(128, ceil(capacity / (8 * block_size)))
```

workgroups and writes one two-word `{lo, hi}` partial per group. A second fixed
tree combines those partials when `G > 1`; `G == 1` completes in the first
pass. Each first-pass thread uses a grid-stride walk, so the group cap bounds
scratch and dispatch shape without changing input coverage. There is no
cross-workgroup polling, lookback, atomic accumulation, schedule-dependent
winner, or runtime fallback.

The hierarchy keeps exact 128-bit partials without paying a 128-bit carry chain
for every U32 element. Its local U32 authority is `RundPair {lo32, hi32}` with
explicit carry. Sum consumes at most 256 inputs per narrow chunk. An unsigned
chunk is at most `256(2^32 - 1) < 2^40`; a signed chunk lies in
`[-256*2^31, 256(2^31 - 1)]`, which is contained in signed 40 bits. The pair
therefore represents every chunk exactly as two's-complement or unsigned
64-bit bits. It is then sign- or zero-extended once into
`RundWide {lo64, hi64}`. Threadgroup trees, cross-group partials, and the final
fit check use that exact two-64-bit representation. U64 Sum enters the same
wide representation per element. `CountNonzero` accumulates its local count in
the same two-32-bit pair, whose complete admitted result is at most
`2^64 - 1`, then zero-extends into the two-64-bit hierarchy.

Both source backends append the same kernel-owned pair/wide arithmetic body;
Metal and Vulkan do not maintain parallel carry, sign-extension, or final-fit
formulas. U32 local accumulation performs no per-element wide carry work while
preserving the exact mathematical sum, terminal overflow reason, and backend
parity.

The two-word operation is addition modulo `2^128`, which is associative. For
at most `2^64 - 1` stored inputs, an unsigned u64 sum is less than `2^128`, a
signed u64 sum has magnitude less than `2^127`, and a nonzero count is at most
`2^64 - 1`. The final two-word value therefore represents the exact
mathematical result; only the final pass checks whether it fits the declared
32- or 64-bit signed, unsigned, or fixed stored domain. Intermediate-prefix
overflow is deliberately not error authority, preserving exact cancellation.

Metal and Vulkan use private intermediate/status storage. Status storage is a
fail-closed diagnostic channel only; it is not semantic output and not an
accumulator visible to callers. On overflow or malformed shape, execution
returns an exact reduce reason and callers must not consume the output buffer
as a semantic result.

`RunAccelKernel(...)` reports physical reduce dispatch count from the frozen
kernel plan: one for a single-group exact reduction, two for the wide
hierarchy, and the fixed planned tree count for Min/Max.
Resident reduce runs do not stage host input and do not download output
implicitly; only explicit upload/download calls affect user-facing transfer
byte counters.

`compute.backend` covers Sum parity for `i32`, `u32`, `i64`, `u64`,
`Fixed<1,31>`, and `Fixed<1,63>` on CPU, Metal, and Vulkan.
`compute.flow-numeric-modes` covers `Fixed<16,16>` and `Fixed<20,44>`,
storage-width cancellation, exact overflow reasons, and 65,536-element scale
rows through Standalone execution. Runtime receives an opaque prepared Job
state and therefore keeps no second numeric-domain matrix. `accel.kernel-core`
directly covers 32- and 64-bit CountNonzero plus Min and Max at an admitted
non-power-of-two block width on every backend.
