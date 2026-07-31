# Compute Segmented Reduce Contract

This page owns the kernel-only segmented reduce primitive contract. The parent
[Compute](../../compute.md) contract routes here and keeps graph-wide identity,
lowering, prepared handoff, and telemetry authority outside this owner.

## Scope

Public authority:

- `/kernel/include/kernel/program/compute/segmented/reduce/model.hpp`
- `/kernel/include/kernel/program/compute/segmented/reduce/plan.hpp`
- `/kernel/include/kernel/program/compute/segmented/reduce/identity.hpp`
- `/kernel/include/kernel/program/compute/segmented/reduce/reference.hpp`
- `/kernel/include/kernel/program/compute/segmented/heads.hpp`
- `/kernel/include/kernel/program/compute/segmented/signed.hpp`

Verification authority:

- `/kernel/tests/contract/program/compute/segmented/reduce.cpp`
- `/kernel/tests/contract/program/compute/segmented/reduce/`
- `/kernel/tests/contract/program/compute/graph/admission.cpp`

`SegmentedReduceDesc`, `SegmentedReducePlan`,
`SegmentedReduceHash`, `SegmentedReduceResult`,
`HashSegmentedReduce(...)`, `PlanSegmentedReduce(...)`,
`ReferenceSegmentedReduceSumU32(...)`,
`ReferenceSegmentedReduceSumU64(...)`,
`ReferenceSegmentedReduceCountNonzeroU32(...)`,
`ReferenceSegmentedReduceCountNonzeroU64(...)`,
`ReferenceSegmentedReduceMinU32(...)`,
`ReferenceSegmentedReduceMinU64(...)`,
`ReferenceSegmentedReduceMaxU32(...)`, and
`ReferenceSegmentedReduceMaxU64(...)` are the unsigned surface.
`ReferenceSignedSegmentedReduce(...)` is the one signed and Fixed stored-lane
reference used by both Node CPU execution routes. These are the planning,
identity, and CPU-reference surface for deterministic segmented reductions.

## Non-Goals

Segmented reduce support in `/kernel` does not execute Metal or Vulkan
work, does not mutate resident Compute buffers, does not call backend APIs, and does
not make hardware cache, occupancy, PMU, timing, or speedup claims. Node
backend execution for CPU, Metal, and Vulkan requires its own contracts
and evidence.

## Planning

`PlanSegmentedReduce(...)` is pure and deterministic. It consumes only a
caller-provided `SegmentedReduceDesc` and must never call node, OS, backend,
Compute, filesystem, clock, Metal, Vulkan, Foundation, driver, allocator,
hardware discovery, runtime cache, or timing APIs.

The contract reuses `ReduceOp` and `ReduceElement` as the single operation and
element authority. It admits `ReduceOp::Sum`, `ReduceOp::CountNonzero`,
`ReduceOp::Min`, `ReduceOp::Max`, `ReduceElement::U32`,
`ReduceElement::U64`, nonzero element count, and nonzero block size. Segment
heads are represented as one u32 flag per element; flag `1` starts a new
segment and flag `0` continues the current segment.

A successful `SegmentedReducePlan` records op, element, element count, element
byte width, segment-head byte width, block size, block count, pass count, temp
value bytes, temp head bytes, status bytes, total temp bytes, `ok = true`, and
reason `ok`. Temp storage is checked in u64 as:

- `temp_value_bytes = element_count * element_bytes`
- `temp_head_bytes = element_count * 4`
- `status_bytes = 4`
- `temp_bytes = temp_value_bytes + temp_head_bytes + status_bytes`

Every multiply and sum is checked before admission. The shape uses
`pass_count = 1` for a single block and `2` otherwise, representing block-local
segmented reduction and deterministic block-boundary combination.

## Identity

`HashSegmentedReduce(...)` derives the primitive descriptor hash from op,
element enum, element count, and block size only. The same
descriptor facts produce the same hash regardless of backend runtime state,
resident storage authenticity, pipeline caches, command queues, measured
timings, or adapter selection.

Kernel graph validation admits `NodeKind::SegmentedReduce` as a collective node
only when it carries a nonzero segmented-reduce descriptor hash, nonzero
element count, no map operation hash, and valid logical buffer refs. This graph
admission records semantic identity only; CPU, Metal, and Vulkan
execution and node evidence require separate backend contracts.

## CPU Reference

The CPU reference helpers traverse elements in ascending index order. Element
zero must be a segment head, and every segment-head flag must be `0` or `1`.
At a head, the previous segment is emitted before starting the next segment
from the current element. `Sum` adds values exactly in segment order;
`CountNonzero` counts nonzero values in each segment; `Min` and `Max` compare
the declared unsigned values. U32 helpers fail when an emitted segment sum or
count cannot fit the u32 output domain. U64 sum helpers fail on u64 addition
overflow; u64 nonzero counts fail only if the count accumulator overflows.

Callers may only consume output buffers when the returned result is ok;
rejected reference results are failure evidence, not semantic output. The
result records the number of segments and the final segment total. Structural
head rejection has higher severity than numeric overflow. The CPU reference
keeps the success path single-pass: a numeric rejection validates only the
unread head suffix before it publishes the reason, so a later malformed head
deterministically yields `compute_segmented_reduce_segment_invalid`.

The signed reference uses one signed 128-bit total carrier. With at most
`2^64 - 1` inputs, a 32-bit stored sum has magnitude below `2^95`, while a
64-bit stored sum has magnitude below `2^127`. Both therefore fit exactly and
only the final declared-width fit check can reject. Fixed reductions use that
same stored signed integer law. Node does not own a second signed segmented
traversal.

Segmented reduce primitive tests follow the owner-local contract shape:
`segmented/reduce.cpp` is only the runner, while `segmented/reduce/` owns
planner rejection, deterministic identity, plan shape, graph admission
identity, and CPU reference semantics.

## Rejection Vocabulary

SegmentedReduce rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_segmented_reduce_invalid` |
| Unknown segmented reduce operation | `compute_segmented_reduce_op_unsupported` |
| Unknown element width | `compute_segmented_reduce_element_unsupported` |
| Zero element count | `compute_segmented_reduce_count_zero` |
| Zero block size | `compute_segmented_reduce_block_invalid` |
| Temp byte arithmetic overflow | `compute_segmented_reduce_temp_overflow` |
| Missing CPU reference input, segment-head flags, output buffer, or segment-count pointer | `compute_segmented_reduce_buffer_invalid` |
| First segment head missing or segment-head flag outside `0`/`1` | `compute_segmented_reduce_segment_invalid` |
| CPU reference exact in-segment sum cannot fit the declared element domain | `compute_segmented_reduce_sum_overflow` |
| CPU reference exact in-segment nonzero count cannot fit the declared element domain | `compute_segmented_reduce_count_overflow` |
