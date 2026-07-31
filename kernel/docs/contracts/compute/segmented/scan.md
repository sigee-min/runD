# Compute Segmented Scan Contract

This page owns the kernel-only segmented scan primitive contract. The parent
[Accel](../../compute.md) contract routes here and keeps graph-wide identity, lowering,
prepared handoff, and telemetry authority.

## Scope

Public authority:

- `/kernel/include/kernel/program/compute/segmented/scan/model.hpp`
- `/kernel/include/kernel/program/compute/segmented/scan/plan.hpp`
- `/kernel/include/kernel/program/compute/segmented/scan/identity.hpp`
- `/kernel/include/kernel/program/compute/segmented/scan/reference.hpp`
- `/kernel/include/kernel/program/compute/segmented/heads.hpp`
- `/kernel/include/kernel/program/compute/segmented/signed.hpp`

Verification authority:

- `/kernel/tests/contract/program/compute/segmented/scan.cpp`
- `/kernel/tests/contract/program/compute/segmented/scan/`
- `/kernel/tests/contract/program/compute/graph/admission.cpp`

`SegmentedScanDesc`, `SegmentedScanPlan`,
`SegmentedScanHash`, `SegmentedScanResult`,
`HashSegmentedScan(...)`, `PlanSegmentedScan(...)`,
`ReferenceExclusiveSegmentedScanU32(...)`,
`ReferenceInclusiveSegmentedScanU32(...)`,
`ReferenceExclusiveSegmentedScanU64(...)`, and
`ReferenceInclusiveSegmentedScanU64(...)` are the unsigned surface.
`ReferenceSignedSegmentedScan(...)` is the one signed and Fixed stored-lane
reference used by both Node CPU execution routes. These are the planning,
identity, and CPU-reference surface for deterministic segmented sum scans.

## Non-Goals

Segmented scan support in `/kernel` does not execute Metal or Vulkan work,
does not mutate resident Compute buffers, does not call backend APIs, and does not make
hardware cache, occupancy, PMU, timing, or speedup claims. Node backend
execution for CPU, Metal, and Vulkan requires its own contracts and
evidence.

## Planning

`PlanSegmentedScan(...)` is pure and deterministic. It consumes only a
caller-provided `SegmentedScanDesc` and must never call node, OS, backend,
Compute, filesystem, clock, Metal, Vulkan, Foundation, driver, allocator, hardware
discovery, runtime cache, or timing APIs.

The contract admits `SegmentedScanOp::ExclusiveSum` and
`SegmentedScanOp::InclusiveSum`, `SegmentedScanElement::U32` and
`SegmentedScanElement::U64`, nonzero element count, and nonzero block size.
Segment heads are represented as one u32 flag per element; flag `1` starts a
new segment and flag `0` continues the current segment.

A successful `SegmentedScanPlan` records op, element, element count,
element byte width, segment-head byte width, block size, block count, pass
count, temp value bytes, temp head bytes, total temp bytes, `ok = true`, and
reason `ok`. Temp storage is checked in u64 as:

- `temp_value_bytes = element_count * element_bytes`
- `temp_head_bytes = element_count * 4`
- `temp_bytes = temp_value_bytes + temp_head_bytes`

Every multiply and sum is checked before admission. The shape uses
`pass_count = 1` for a single block and `2` otherwise, representing block-local
segmented scan and deterministic carry fixup.

## Identity

`HashSegmentedScan(...)` derives the primitive descriptor hash from op, element
enum, element count, and block size only. The same
descriptor facts produce the same hash regardless of backend runtime state,
resident storage authenticity, pipeline caches, command queues, measured
timings, or adapter selection.

Kernel graph validation admits `NodeKind::SegmentedScan` as a
collective node only when it carries a nonzero segmented-scan descriptor hash,
nonzero element count, no map operation hash, and valid logical buffer refs.
This graph admission records semantic identity only; CPU, Metal, and Vulkan
must enforce it identically.

## CPU Reference

The CPU reference helpers traverse elements in ascending index order. Element
zero must be a segment head, and every segment-head flag must be `0` or `1`.
At a head, the running sum resets before emitting the current element's result.
Exclusive reference writes the running sum before adding the current input;
inclusive reference adds first and then writes. This preserves segment boundary
order and selected arithmetic law independently of backend schedule.

U32 helpers fail when an emitted or final in-segment sum cannot fit the u32
output domain. U64 helpers fail on u64 addition overflow. Callers may only
consume output buffers when the returned result is ok; rejected reference
results are failure evidence, not semantic output. Structural head rejection
has higher severity than numeric overflow. The CPU reference keeps the success
path single-pass: when it first observes numeric overflow, it validates only
the unread head suffix before publishing the reason. A later malformed head
therefore deterministically changes the result to
`compute_segmented_scan_segment_invalid` without adding a second head pass to
successful scans.

The signed reference sign-extends each 32-bit stored value into i64, or each
64-bit stored value into signed 128-bit storage. A scan checks every observable
prefix before another addition, so its carrier only needs one extra stored
addition: magnitude at most `2^32` for a 32-bit lane and at most `2^64` for a
64-bit lane. Fixed scan uses the same stored signed integer law; scale does not
change addition or overflow.

## Backend Carry Obligation

CPU, Metal, and Vulkan must test overflow against the actual running segment
prefix in ascending element order. A block-local prefix evaluated from zero is
not an overflow oracle for a segment that entered the block with a nonzero
carry. For signed and Fixed storage, the carry can either cancel an otherwise
overflowing local prefix or make an otherwise representable local prefix
overflow.

For a multi-block segment, the backend decomposition therefore has one
canonical ownership rule:

1. `block` materializes modulo-width local prefixes and validates only
   subsegments that start at a head within that block;
2. `prefix` records the actual carry entering each block;
3. `offset` replays the leading subsegment from that carry in ascending order,
   writes its inclusive or exclusive prefixes, and records every actual
   transition overflow.

Every in-segment addition is checked exactly once: after a local head by
`block`, or before the first local head by `offset`. The carry metadata is used
to partition work, never as a substitute for observing intermediate prefixes.
The stable rejection remains `compute_segmented_scan_sum_overflow`.

Cross-backend contracts include both directions that a local-only oracle gets
wrong: a negative incoming carry followed by a locally overflowing positive
sum must succeed when every global prefix fits, while `MAX + 1 - 1` across a
block boundary must reject at the `MAX + 1` transition even though the final
block total returns to `MAX`.

Segmented scan primitive tests follow the owner-local contract shape:
`segmented/scan.cpp` is only the runner, while `segmented/scan/` owns planner
rejection, deterministic identity, plan shape, graph admission identity, and
CPU reference semantics.

## Rejection Vocabulary

SegmentedScan rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value | `compute_segmented_scan_invalid` |
| Unknown segmented scan operation | `compute_segmented_scan_op_unsupported` |
| Unknown element width | `compute_segmented_scan_element_unsupported` |
| Zero element count | `compute_segmented_scan_count_zero` |
| Zero block size | `compute_segmented_scan_block_invalid` |
| Temp byte arithmetic overflow | `compute_segmented_scan_temp_overflow` |
| Missing CPU reference input, segment-head flags, or output buffer | `compute_segmented_scan_buffer_invalid` |
| First segment head missing or segment-head flag outside `0`/`1` | `compute_segmented_scan_segment_invalid` |
| CPU reference exact in-segment sum cannot fit the declared element domain | `compute_segmented_scan_sum_overflow` |
