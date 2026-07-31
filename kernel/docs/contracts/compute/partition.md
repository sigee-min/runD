# Compute Partition Contract

This page owns the kernel-only stable partition primitive contract. The parent
[Accel](../compute.md) contract routes here and keeps graph-wide identity, lowering,
prepared handoff, and telemetry authority.

## Scope

Public authority:

- `/kernel/include/kernel/program/compute/partition/model.hpp`
- `/kernel/include/kernel/program/compute/partition/plan.hpp`
- `/kernel/include/kernel/program/compute/partition/identity.hpp`
- `/kernel/include/kernel/program/compute/partition/reference.hpp`

Verification authority:

- `/kernel/tests/contract/program/compute/partition.cpp`
- `/kernel/tests/contract/program/compute/partition/plan.cpp`
- `/kernel/tests/contract/program/compute/partition/identity.cpp`
- `/kernel/tests/contract/program/compute/partition/reference.cpp`
- `/kernel/tests/contract/program/compute/graph.cpp`

`PartitionDesc`, `PartitionPlan`, `PartitionHash`,
`PartitionResult`, `HashPartition(...)`,
`PlanPartition(...)`, and the typed `ReferenceStablePartition*` entry points
are the planning, identity, and CPU-reference surface for deterministic stable
partition over 32- or 64-bit flag and value storage.

## Non-Goals

Partition support in `/kernel` does not execute backend work, does not
mutate resident Compute buffers, does not call backend APIs, and does not make
hardware cache, occupancy, PMU, timing, or speedup claims. Node backend
execution requires its own contracts and evidence.

## Planning

`PlanPartition(...)` is pure and deterministic. It consumes only a
caller-provided `PartitionDesc` and must never call node, OS, backend, Compute,
filesystem, clock, Metal, Vulkan, Foundation, driver, allocator, hardware
discovery, runtime cache, or timing APIs.

The contract admits only nonzero `element_count`, `flag_bytes` equal to `4` or
`8`, and `value_bytes` equal to `4` or `8`. The public Flow Partition side
input is U32; the wider flag form remains available to internal graph owners
whose same-width mask lineage is U64. Widths outside those choices fail closed
as `compute_partition_invalid`.

A successful `PartitionPlan` records element count, flag byte width, value
byte width, one false-prefix scan extent, total temp bytes, pass count,
`ok = true`, and reason `ok`. For binary group indicators
`f_i = 1` when flag `i` is false and `0` otherwise, the one canonical
exclusive prefix is `F_i = sum_(j<i) f_j`. The true prefix is derived rather
than stored:

- `T_i = i - F_i`
- `F_N = F_(N-1) + f_(N-1)`
- `T_N = N - F_N`
- `scan_temp_bytes = element_count * 4`
- `temp_bytes = scan_temp_bytes`

The multiply is checked before admission. The shape has `pass_count = 3`,
representing binary classification, one false-prefix scan, and ordered
scatter. Backend physical scan sub-dispatch counts remain execution evidence;
they are not semantic plan or graph identity.

## Identity

`HashPartition(...)` derives the primitive descriptor hash from element count,
flag byte width, and value byte width only. The same
descriptor facts produce the same hash regardless of backend runtime state,
resident storage authenticity, pipeline caches, command queues, measured
timings, or adapter selection.

Kernel graph validation admits `NodeKind::Partition` as a
collective node when the node carries the partition descriptor hash, nonzero
element count, valid read/read/write buffer roles, and no map operation hash.
The graph identity input is still only kind, primitive hash, element count,
logical buffer ids, roles, and order; the detailed descriptor payload remains
caller-owned and is rechecked by node before program admission.

## CPU Reference

The typed `ReferenceStablePartition*` entry points preserve the original
ascending input order inside both false and true groups and write the complete
false group before the true group. They cover every admitted 32/64-bit
flag/value width pair without reinterpreting the value payload. The reference
first counts the false group from flags, then performs one ascending stable
scatter over flags and values. False element `i` writes to `F_i`; true element
`i` writes to `F_N + (i - F_i)`. Since `0 <= F_i <= i`, the derived true rank
cannot underflow. For a true element, `F_N + i - F_i <= N - 1`, so the target
cannot overflow the admitted U32 backend index range. False targets occupy
`[0, F_N)` and true targets occupy `[F_N, N)`; each prefix increases by one
inside its group, proving disjointness, completeness, and stable order. The
all-false boundary gives `F_N = N, T_N = 0`; the all-true boundary gives
`F_N = 0, T_N = N`. Zero count remains rejected before `N - 1` is formed.
The scatter writes each input value exactly once and has no size-dependent
reference path.

Callers may only consume output buffers when the returned result is ok.
Rejected reference results are failure evidence, not semantic output.

## Rejection Vocabulary

Partition rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Default-constructed or non-admitted plan/reference value, unsupported flag byte width, or unsupported value byte width | `compute_partition_invalid` |
| Zero element count | `compute_partition_count_zero` |
| Temp byte arithmetic overflow | `compute_partition_temp_overflow` |
| Missing CPU reference flags, values, output, false-count, or true-count pointer | `compute_partition_buffer_invalid` |
