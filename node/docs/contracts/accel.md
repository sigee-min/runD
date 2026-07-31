# Node Accel Contract

## Scope

Node owns backend discovery, authenticated resident resources, prepared
execution, and runtime evidence for CPU, Metal, and Vulkan. Kernel owns graph
meaning, numeric policy, planning, ordering, and deterministic reference laws.
The public product projection is `rund::compute`; Node Accel types remain an
internal execution boundary.

There is no automatic execution fallback. A caller selects one backend, and a
missing adapter or rejected capability returns that backend's precise reason.
`Fake` exists only for internal diagnostic contracts and is never selected by
the product Compute surface.

## Authority

- [Adapter selection](./accel/adapter.md)
- [Runtime resources](./accel/runtime/resources.md)
- [Resident context](./accel/context.md)
- [CPU SIMD](./accel/cpu/simd.md)
- [Kernel stream](./accel/kernel/stream.md)
- [Compact](./accel/compact.md)
- [Gather](./accel/gather.md)
- [Histogram](./accel/histogram.md)
- [Reduce](./accel/reduce.md)
- [Scan](./accel/scan.md)
- [Scatter](./accel/scatter.md)
- [Segmented Scan](./accel/segmented/scan.md)
- [Segmented Reduce](./accel/segmented/reduce.md)
- [Sort](./accel/sort.md)
- [Stencil](./accel/stencil.md)

Implementation authority is `/node/src/accel`. Public value authority is
`/accel/include/accel`; Node must not mirror those types. Verification authority
is `/node/tests/contract/accel` and the registered cases under
`/node/tests/contract/cases/accel`.

## Invariants

- Adapter identity and capabilities are frozen once at selection.
- Context, kernel, and buffer admission authenticate both control-block and
  stored-pointer identity.
- A compiled graph has one immutable token and one ordered backend operation
  table. Common code has no second primitive switch.
- Each graph node owns one inline tagged operation. The variant tag is the sole
  primitive-kind authority and only the active descriptor/plan pair occupies
  storage. Compile moves that operation into the immutable execution table;
  run planning borrows it and retains only the common dispatch plan. CPU,
  Metal, and Vulkan therefore cannot observe a different primitive payload or
  a second per-run plan copy.
- CPU, Metal, and Vulkan consume the same graph identity, stored format, and
  numeric policy. Backend timing, cache state, and dispatch geometry never enter
  semantic hashes.
- Resident execution performs no implicit download. User-visible transfer
  counters change only at explicit upload, write, or read boundaries.
- Kernel [checked arithmetic](../../../kernel/docs/contracts/checked.md) is
  the sole owner of count addition and
  count-by-width multiplication. Sequence packing, resident admission,
  primitive shape checks, resource-range planning, accelerator policy, and
  native work-evidence accumulation consume it directly; Node adds only the
  `u64` to host-size boundary conversion. For
  `M = 2^64 - 1`, addition rejects `a > M - b`, multiplication rejects
  `b != 0 && a > M / b`, and host conversion rejects `a > SIZE_MAX`; no
  overflowing expression is evaluated before its predicate.
- Prepared synchronous and asynchronous execution share the same owner and
  claim; concurrent reuse fails with `compute_job_busy`.
- Fake, Metal, and Vulkan execution consume one dispatch-window validator.
  The test backend cannot accept a partition, sequence projection, or resident
  full-range shape that a native backend rejects.
- Kernel primitive planners own the complete descriptor-to-plan equality law.
  Node calls the canonical `*PlanMatchesDesc` predicate and adds only runtime
  binding identity, usage, alignment, and frozen device limits; it keeps no
  duplicate list of planner fields. The admission cost is constant in the
  primitive descriptor and occurs before execution, so admission adds no
  dispatch, allocation, element pass, or backend-specific state.
- Primitive status words have one backend-neutral decoder beside their
  resident shape owner. Metal and Vulkan completion paths transport the word
  and publish backend evidence, but do not repeat its semantic reason table.
  Unknown nonzero words fail closed with the primitive's `*_invalid` reason;
  they can never produce a failed result carrying `ok` as its reason.
- Gather, Histogram, Partition, Scatter, Segmented Scan, and Stencil each own
  one backend-neutral host parameter model under their domain directory.
  Metal and Vulkan local headers include that model directly and contain no
  declaration, alias, or re-export of the parameter type. Their shader-language
  declarations remain backend artifacts, but the source contract checks every
  field name, order, and width against the C++ model's exhaustive
  `sizeof`, `alignof`, and `offsetof` assertions.

## Verification

The focused owners are `accel.kernel-core`, `accel.kernel-numeric`,
`accel.backend-fixed`, and `accel.backend-runtime`. Wider Compute parity lives
in the Standalone Compute contracts; Runtime verifies scheduling integration
without duplicating the numeric matrix.
