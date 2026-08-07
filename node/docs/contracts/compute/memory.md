# Compute Memory Ownership Contract

Node owns the mapping from Compute object ownership to public `MemoryStats` and
`MemorySnapshot`. Public category meaning and graph payload formulas live in
[`docs/reference/compute.md`](../../../../docs/reference/compute.md); Kernel
private executor and lowering-owner byte oracles live in
[`kernel/docs/contracts/compute/handoff.md`](../../../../kernel/docs/contracts/compute/handoff.md).

## Capacity exception boundaries

Compute boundaries that intentionally project both `std::bad_alloc` and
`std::length_error` consume one source-private exception classifier. That leaf
owns only the accepted C++ exception classes: the calling boundary retains its
typed `Reason`, result construction, and phase-local rollback. Any other active
exception is rethrown unchanged, preserving ordinary propagation and existing
`noexcept` termination. A boundary that authorizes only `std::bad_alloc`
remains explicit and is not silently widened.

## Recipe value-route ownership

Each reusable Flow recipe and each canonical Graph construction state owns one
contiguous `ValueIdArena`. Map and Primitive steps retain only ordered
`(offset, count)` ranges into that state owner; they do not retain per-step
input and output vectors. The arena is the sole storage authority for those
routes, and every liveness, canonical-order, description, compilation, and
runtime-compaction consumer resolves a read-only span from it.

A zero-input Map is canonical for index and constant construction, so range
validity and range emptiness are distinct. `valid(range)` first proves
`offset <= size` and `count <= size - offset`; a valid zero-count input range
is then admitted for Map. Every Map output range and both Primitive ranges are
nonempty. An invalid range cannot be reinterpreted as a valid zero-input Map.
Route storage also rejects a span borrowed from the same arena, preventing a
growth reallocation from invalidating its own source. Current construction
paths pass stack, fixed-inline, or independent vector spans and never use an
arena view as storage input.

The arena has no fixed-capacity path or fallback. It owns a move-only
`unique_ptr<uint32_t[]>`, size, and exact capacity rather than delegating the
capacity law to a standard container. Let `V` be the number of stored value
IDs and `C` the retained capacity. A growth allocates the greater of the
required size and `C + ceil(C/2)`, subject to the effective limit
`min(UINT32_MAX, SIZE_MAX / sizeof(uint32_t))`. After a nonempty store,
retained capacity is less than `1.5V + 1` IDs. Successive capacities grow by at
least `1.5`, so all prior-capacity copies total less than `3V + 2` IDs. At most
one allocation precedes the ordered old-ID, input, and output copies;
allocation failure leaves size unchanged, and the optional ranges are
published only after all copies complete.

Flow and Graph step construction use the arena's callback-scoped publication
transaction, not a separately maintained rollback list. The arena captures the
exact pre-store size and passes the staged ranges only to the callback that
materializes one Map or Primitive step. The transaction commits when that
callback returns; an exception restores the logical arena prefix while
retaining any safely grown allocation. No transaction object can outlive its
arena. The arena itself is non-movable and rejects nested publication, so owner
relocation and reentrant staging cannot detach rollback from its target.
Publication is the sole mutation entry; there is no direct-store bypass.
Consequently a route tail without an owning step is unrepresentable on all four
Flow/Graph Map/Primitive append paths. Allocation-fault and transaction-scope
contracts prove exact prefix rollback, typed terminal failure, and successful
retry for both construction states.

Program graph schedule capacity is not expression capacity. One Program admits
16,384 ordered graph nodes while each Map expression and emitted ComputeIR
remains bounded at 1,024 nodes. Graph values are admitted against the derived
1,048,640-resource envelope, but vectors and the value-ID arena retain only the
authored amount. Fusion planning likewise allocates `O(N)` cold workspace for
the actual graph and releases it before Program publication. Raising the
Program envelope therefore does not create a maximum-size per-Program table,
does not widen a shader IR, and adds no warm Job or Pipeline storage.

On the allocation-counted contract workload of 1,024 routes with two inputs
and one output each, the exact 1.5-growth sequence performs 18 arena
allocations while checking all 3,072 IDs after repeated relocation. This is a
recipe and Graph-construction allocation bound. The published CPU runtime graph
has distinct Map and Primitive vectors whose lifetime extends through Program
execution; those owners remain part of Program memory.

## Expression lowering scratch ownership

Multi-output Map lowering validates each distinct immutable expression state
once. While emitting consecutive roots from the same state, it also retains one
node-value vector and one byte-per-node visitation vector. A completed node is
therefore emitted once for the whole same-state root group. Switching to a
different state clears logical contents but retains scratch capacity; output
order and state ownership remain unchanged.

For roots with reachable node sets `R_1 ... R_m`, one same-state group performs
`|union(R_i)|` DFS visits and emission attempts. Each canonical node lookup is
performed once per group. BuildContext canonicalization remains the sole IR
authority, so canonical bytes and operation hash are independent of how many
roots reach the same node.
Scratch retention is bounded by the largest referenced node ordinal and the
public maximum of 16 outputs.

`expression/state.hpp` is the sole owner of expression operand cardinality.
`expr_arity(ExprOp)` maps every admitted operation to zero, one, two, or three
SSA operands; an unknown encoded operation maps to `InvalidArity`. Liveness,
projection, graph validation, and replay all consume that same mapping. They do
not carry independent operation lists, so adding an operation cannot make one
phase traverse a different dependency graph. The mapping is compiled once in
`expression/arity.cpp`; support is derived from that same arity law and
consumers retain only its declaration. No consumer carries a local per-node
operation switch; traversal order, allocation, emitted IR, and the number of
visited nodes remain fixed by the shared mapping.

## Program host ownership

Public `Device`, `Buffer<T>`, and `Program<Signature>` handles retain only their
shared state owner. Backend selection, buffer count, and Program input/output
extents live in that state; the handles do not mirror those values in
per-wrapper fields. `backend()`, `size()`, `output_size()`, `capacity()`, and
binding validation observe the same frozen owner used by execution.
Constructing a Program handle is therefore `O(1)` in input/output arity after
compilation instead of recopying the already materialized `I + O` shape values.
All typed Program shapes inherit that observer behavior from one private
handle owner. Moving transfers its sole shared-state handle; the source becomes
invalid, `backend()` returns `ProgramInvalid`, and execution checks that state
before shape matching instead of substituting CPU or a zero-sized Program.
Input type and extent vectors are the only Program input-shape authority. They
are nonempty and have equal cardinality when compilation publishes the Program;
run, resident-Job creation, and Job write admission reject any violated
invariant. There is no first-input scalar mirror or empty-vector fallback.

`DeviceState` owns one immutable backend-operation pointer. CPU devices leave
it null and execute the CPU Job branch. Native devices use the single Accel
adapter for allocation, transfer, compile, run, resident Job, and staging
memory observation. The memory API therefore does not contain a second native
backend switch or call native symbols from the CPU archive. The state cost is
one host pointer; dispatch happens once per public operation and never inside
an element or tile loop.

### Logical Buffer initialization

A successful `Device::buffer<T>(count)` publishes a new logical Buffer whose
entire `count * sizeof(T)` payload is all-byte zero on CPU, Metal, and Vulkan.
The guarantee applies equally to fresh native allocation and to storage
reacquired from a backend pool: old-owner bytes are cleared before the new
owner is published. A failed clear publishes no Buffer. Pooling therefore
reuses physical capacity without exposing a stale logical payload.

Allocation carries one internal initialization intent. The public Buffer path
uses `Zeroed`. A path that proves it overwrites every payload byte before any
read uses `FullOverwrite`: `Device::upload`, complete resident input upload,
private dense View staging, and complete-write graph memory chunks are such
owners. These are cold ownership rules only; they are not a substitute for an
execution reset when a Job or Pipeline reuses storage.

The graph-memory planner fails closed unless the first access is a dense
write with zero byte offset, exact logical byte size, exact element count and
width, and stride equal to element width; a read in that same graph node is
forbidden. Graph construction supplies exactly one execution-coverage proof
for every node. A Domain write may claim its active prefix only when every
consumer uses the same count or a descendant count and preserves any writer
predicate. A general bounded, predicate, segmented-reduce, or scatter write
cannot claim coverage merely because its declared access spans the capacity.

Let `O_r` be the bytes of one authored output View or internal logical
resource `r`, and let `W_r` be the bytes proved overwritten before their first
read. If `W_r != O_r` and there is no proved Domain subset, the compiler
records the first writer as `graph::Resource::reset_node`. The resource resets
exactly `O_r` once immediately before that writer:

```text
for every byte b read in the invocation:
  b in W        => the first writer defines b
  b not in W    => the first-write reset defines b as all-byte zero
```

The reset frontier is part of the same closed lifetime `[first_r, last_r]`
used by arena placement. Two internal reset resources may reuse an aligned
range exactly when `last_a < first_b` or `last_b < first_a`; the later range is
cleared after every earlier use and immediately before its own first writer.
Any logical output read before that first writer, or read and partially written
by the same first node, is rejected because no first-write reset can initialize
that read without changing operation order.
This changes neither operation order nor arithmetic. Each logical range retains
one exact reset route even when ranges share an offset or physical arena
Buffer. Physical-owner equality never deduplicates reset routes. Therefore, for
reset set `R`,

```text
reset_bytes = sum(r in R, bytes(r))
reset_count = |R|
```

Padding and unrelated arena intervals are not reset. A Pipeline recurrence
invokes its body Program at every prepared occurrence, so each occurrence
consumes the same Program-owned reset routes. An external partial-write output
resets its complete authored View, including a strided View. Inputs, the
current recurrence bank, and transactional published state are explicit carry
authorities. A transactional pending bank requires a complete overwrite before
publication.

CPU, Metal, and Vulkan consume the canonical order `(reset_node, binding)`.
Control suppression does not suppress initialization: CPU clears before it
evaluates the node control, and native reset commands remain ahead of the
controlled dispatch. A reset after prior commands has one prior-work-to-reset
visibility edge and one reset-to-writer edge. A writer's ordinary hazard edge
is not emitted a second time when those reset edges already cover it.

`reset_node` and the matching last-use frontier remain original Program node
ordinals. Compute lowering writes `BufferInit::Zero` only on that exact
first-write binding. The marker participates in Kernel graph identity and is
the sole accelerator reset intent; `AccelRun` has no reset coordinates or
route list. Accelerator fusion retains a gap-free half-open source interval on
every final execution step and applies the unique monotone projection

```text
project(i) = j  where step[j].begin <= i < step[j].end
```

before native scheduling and physical-lifetime validation. Thus fusion changes
only the execution coordinate of a reset, never its authored ordering,
identity, byte range, or arithmetic meaning. Token mint matches each surviving
marked binding to its exact original node and seals its binding, projected
first write, and projected alias last use. Alias last use is selected by the
maximum original source ordinal, not an incidental final-binding traversal
position. A marked internal intermediate removed by legal fusion has no
physical range and produces no reset plan; every materialized marked binding
must produce exactly one.

For `S` final steps, `B` canonical graph binding occurrences, and `R`
surviving reset routes, token mint spends `Theta(S + B)` once and retains
`8 * S + 16 * R` logical bytes. Reset-free preparation is `Theta(1)` and
allocates no reset storage. For `R` reset routes, binding projection is
`Theta(R)`; the independent physical-overlap proof remains `O(R log R)`.
No Job, Pipeline occurrence, or backend rebuilds the graph-to-step authority.

Metal and Vulkan clear a public Buffer at its cold ownership transition.
Those allocation writes do not increment upload or download counters. Program
resets are separate execution work: `MemoryPlan::{reset_bytes,reset_count}`
reports the canonical bytes and logical reset ranges, while
`Stats::{reset_bytes,reset_commands}` reports work consumed by that invocation.
Warm `Job::run`, Pipeline steps, and every recurrence occurrence follow the
same exact-node frontier. There is no invocation-start mirror, cold-only path,
or backend-specific fallback.
Native View lowering builds one binding-ordinal index while it creates dense
transfers. For `B` graph bindings and `V` lowered transfers, cold replacement
resolution is therefore `O(B + V)` time and `O(B + V)` storage, rather than a
linear search per binding. Compute lowering emits the reset marker while it
already visits the canonical access, so it builds no value-to-Write table,
accelerator reset vector, or per-Job route projection. Preparation resolves
each token-owned target once and freezes one contiguous reset span on its
native step. Warm encoding then visits exactly `S + R` entries, not `R * S`.

`accel/kernel/reset` is the sole native reset value and arithmetic authority.
`model.hpp` separates the raw byte `Spec` from the constructor-closed proved
`Range` and owns the standard-layout 40-byte `Params` shared by Metal bytes and
Vulkan push constants. `projection.hpp` owns the
allocation-free unique dense-View replacement lookup, and compiled
`proof.cpp` projects either the authored resident range or its dense
replacement and validates it once during preparation. For target storage
extent `S`, offset `o`, count `n`, stride `s`, and element width `w`, it first
requires `n > 0`, `w in {4, 8}`, four-byte alignment, `s >= w`, and `o <= S`.
It then proves the final exclusive byte without an overflowing product or sum:

```text
n - 1 <= (S - o - w) / s
```

The proof makes both `n*w` payload accounting and
`o + (n-1)*s + w` address materialization safe and stores that proved exclusive
end in the Range. A backend applies only its native address-width
specialization afterward: the common `WordAddressable` projection tests the
proved descriptor-relative end against a word limit after validating the
proved origin. Metal adds no value rule. Standalone Vulkan dense reset remains
one `vkCmdFillBuffer` and does not inherit the shader limit. Strided resets and
every Pipeline-private captured reset, including dense ranges, use the fixed
256-lane shader and therefore share its u32 descriptor-relative word limit and
checked multi-dispatch command count. Metal remains exact `dispatchThreads`
for both dense and strided ranges. Those execution forms are backend
mechanics, not alternate value or validation policies.

`BuildResetBinds` mints that Range before backend selection. A sealed
`BoundReset` retains resident identity, extent, usage, and lifetime separately,
but derives every lookup geometry field from the one proved Range; it never
retains a writable raw-geometry mirror. CPU clearing, overlap analysis,
reservation accounting, and non-replacement Metal/Vulkan preparation consume
the same Range. A native dense View replacement changes the physical range and
therefore receives one new proof against the replacement's actual extent.
Prepared native reset records then compose their handle or descriptor around
the proved Range; repeated encoding performs no range, overflow, alignment,
replacement, allocation, or payload-copy work.

The public `Run` receipt retains its private state in a 1,152-byte,
`uint64_t`-aligned inline store. The source-private `RunState` is 1,120 bytes
with 8-byte alignment, so the checked bound is `1,120 <= 1,152` with 32 bytes
of reserve. `Result<Run>` is 1,160 bytes on the checked 64-bit ABI. The reserve
is an explicit stack and ABI footprint tradeoff for isolating private layout
growth; it is neither heap storage nor extra initialized/copied payload.
Construction, copying, moving, and destruction are compiled owners; public
headers never require the complete private state. A warm
`Program::run(Buffer, Buffer)` therefore still performs zero SDK heap
allocations, and copying or moving its receipt adds no owner allocation. The
copied receipt retains independent mutable read telemetry over the same shared
buffers. The compile-time size, alignment, and nothrow checks plus the
allocation and telemetry-divergence oracles live in `compute.reuse`.

### One-shot host result ownership

A single-output `Program::run` or Flow `collect` allocates the final
`vector<R>` first and downloads directly into its typed, `alignof(R)`-aligned
storage. The erased execution boundary validates the Program output type,
logical count, and exact byte width before execution, then uses the same typed
read owner as resident execution. There is no intermediate byte-vector result
and no reinterpretation copy. Integral and fixed 32-bit domains therefore use
exactly four bytes per element; their 64-bit domains use exactly eight.

Let `B = output_count * sizeof(R)`. The host result owns one payload allocation
with peak live payload `B`. A nonempty one-shot CPU or accelerator read performs
exactly `3B` of host payload traffic: result value initialization (`B` written),
resident or staging source read (`B` read), and result write (`B` written).
Every backend computes the canonical FNV-1a hash in that source-to-result copy
loop, so it adds no destination reread. A caller-provided explicit read omits
result initialization and therefore performs exactly `2B`. Empty-Program
materialization writes the caller destination once and derives the all-zero
payload hash as `offset * prime^B mod 2^64` by exponentiation by squaring,
without rereading those zero bytes. Both routes retain one readback, transfer
accounting, and an output hash over the final bytes. A zero-length result
performs no download or transfer and records the canonical empty hash through
that same read owner.

Vulkan resident payload and workload-sized collective scratch are
device-local. Host-visible coherent memory is restricted to explicit staging,
small parameter blocks, and host-observed status words. On a discrete-memory
adapter this prevents every shader pass from being bounded by the host-link
memory tier; on unified memory it preserves the same placement contract without
claiming a separate physical heap. Public transfer intervals retain arbitrary
byte offsets through the canonical four-byte aligned staging range defined by
the Accel runtime resource contract. Typed Compute transfers are naturally
four- or eight-byte aligned and therefore pay no range expansion. Every Vulkan
upload and download, including the general one-buffer surface, consumes the
same increasing-order partition whose normalized staging slice is at most the
frozen staging budget. A one-slice full upload submits without a host fence
wait; the command slot retains staging and Vulkan target storage until
completion, and same-queue ordering makes the following prepared run see the
bytes. Larger transfers complete their bounded slices in order and cannot
materialize a payload-sized staging allocation. A full command envelope
applies bounded condition-variable backpressure rather than exposing transient
queue pressure as a transfer error.

Metal and Vulkan public resident handles, not their adapter registries, own
native buffers. Vulkan separates the public adapter-owning handle from
fence-retained native storage. Final storage release may enter one
memory-class pool only when both its count and retained-byte caps admit it.
The common cap is 32 buffers and `32 * staging_budget` bytes per pool; larger
buffers are destroyed and deterministic oldest-first eviction makes room
without allocation. Active plus pooled mapped staging is the physical
`MemoryStats::staging.current` authority, so returning a staging lease does not
hide retained native memory. Prepared-owner staging remains derived from active
lease deltas and is not inflated by unrelated cold pool retention.

The warm one-input CPU allocation oracle is exactly one `operator new` call:
the typed result payload. The Program's serialized convenience Job, BufferState
owners, internal buffers, prepared execution state, and route storage are
retained from the first call. No warm execution allocation is hidden outside
the oracle; CPU buffer payloads are already resident in that cached Job.

Bounded and ordered multi-output terminals use the schema-aware Job reader, but
their short-lived execution owner is read-only. Let `I`, `O`, and `A` be the
physical byte sums of external inputs, distinct physical outputs, and planned
internal arena chunks. `A` is not the sum of authored intermediates. The
virtual arena is partitioned into raw U32 Buffer owners containing canonical
typed subviews at checked byte offsets. One-shot construction owns exactly
`I + O + A`; it neither
allocates the resident pending-write input term `I` nor prepares a second
backend binding set. A writable resident Job owns
`2I + O + A`, and both input sets are prepared before publication.
The `compute.reuse` contract measures the one-shot physical allocation delta from
the Device meter, checks the pending-input row on the resident counterpart, and
then writes and reruns that resident Job to prove the double-buffer law was not
narrowed. Its four-element CPU physical oracles are exact: the Bounded filter
retains 308 bytes during the call (52 external bytes plus one 256-byte-aligned
internal arena) and the two-output record retains 48 bytes. The oracle derives
this as external resource bytes plus `MemoryPlan::physical_bytes`; it never
reconstructs placement from logical resource extents.
Cold `operator new` call
counts are intentionally not a contract: they include standard-library owner
construction and would reject a valid allocation reduction or library change.
The same exclusion applies to native accelerator command execution: a process-
wide replacement `operator new` can observe driver-internal command encoding,
whose count legitimately grows with dispatch and barrier count. Warm
accelerator reuse is instead proven by product telemetry
(`pipeline_compiles == 0`, `buffer_allocations == 0`, `uploaded_bytes == 0`,
and `download_events == 0` before explicit readback) plus stable output and
dispatch evidence. Comparing a multi-pass collective's process-wide call count
to a one-dispatch Map is not a resource-lifetime invariant and is not a test
authority.

### Memory Plan

`graph::Info::memory` is the immutable `MemoryPlan` evidence for this
ownership and its exact first-write reset frontiers.
The planner has two private responsibilities with one published result:
`resource/memory.cpp` derives lifetimes, materialization classes, and reuse
proofs; `resource/memory/arena.cpp` consumes that frozen model and owns the
deterministic aligned placement algorithm. Neither side may reconstruct the
other's policy, and backends consume only the resulting `MemoryPlan`.
`logical_bytes` is the saturating sum of authored internal extents;
`live_bytes` is the maximum closed-interval live sum; `physical_bytes` is `A`;
`allocation_count` is the retained arena Buffer count; `reset_bytes` and
`reset_count` are the exact logical Program reset ranges. Per-run `Stats` owns
reset work for the observed execution.

### Preflight

The public resident-window and memory-admission contract is owned by
[Compute Graph Services](../../../../docs/reference/compute/services.md#resident-windows).
`windows<Max, Tile>` fixes both dimensions at compile time and the body Program
expresses tile-local values through `resident<Max, Tile>`. Its ordinary
`MemoryPlan` therefore describes the body graph exactly as authored; it has no
budget, maximum, tile, occurrence, or backend-allocation fields.
The large-window contract applies this proof to both 32-bit and 64-bit Fixed
storage at `Max = 516096` and `Tile = 8192`: the large plan's transient bytes
and allocation count equal the corresponding one-tile control, while its 63
prepared occurrences increase reuse rather than workspace capacity.

`PipelineBuilder::plan()` computes the private Pipeline allocation rows and
publishes their `PipelinePlan` summary before creating any Pipeline-owned
payload, CPU run storage, or accelerator command owner. `prepare()` consumes
that same frozen plan. Its `resource::Plan::lifetimes` length is the canonical
distinct-resource count: admission reserves the resource rows, admission rows,
and pointer-ordinal index from that exact count, then requires all three sizes
to match. It must not reserve from the larger authored-binding count or repair
an over-reserve with a terminal `shrink_to_fit()` allocation/copy.
`persistent_bytes` counts referenced caller Buffer storage. The checked
admission equations are

```text
prepared_bytes = prepared_buffer_bytes
               + prepared_host_bytes
               + prepared_tile_bytes
               + prepared_native_bytes
peak_bytes     = state_bytes + transient_bytes + prepared_bytes
total_bytes    = persistent_bytes + peak_bytes
```

`peak_bytes` remains the exact logical runD-owned admission amount. CPU
preparation additionally publishes two physical coordinates without changing
that meaning. `arena_extent_bytes` is the exact byte span of the sealed CPU
mapping before page rounding. If `A` is the exact arena payload already
present in `peak_bytes` and `C` is that mapping's independently page-rounded
commitment, then

```text
committed_peak_bytes = peak_bytes - A + C
```

The planner obtains `C` from the sealed layout, not by rounding
`peak_bytes` as one aggregate allocation. `MemoryBudget` continues to compare
`peak_bytes`; the Device aggregate governor reserves
`committed_peak_bytes`. Process RSS, allocator metadata, worker stacks, and
driver-private storage are observations outside both exact payload fields.

The four prepared components are disjoint. `prepared_buffer_bytes` is the
exact logical payload of dense View normalization and accelerator primitive
scratch Buffers. `prepared_host_bytes` reserves retained Pipeline, private-Job,
route, and CPU Program-run objects and their product-owned container payload.
`prepared_tile_bytes` is the exact CPU worker/tile-executor, collective, and
primitive scratch payload. `prepared_native_bytes` reserves explicitly sized
backend command, parameter, status, and descriptor payload owned by runD.
Host and native structural envelopes are checked upper reservations: runtime
materialization may consume less but cannot consume more in any byte or object
count. Opaque allocator headers and driver-private allocation granularity are
not guessed as bytes; backend object capacities are gated before native calls
and their actual retained high-water remains telemetry.
Primary and transactional-alternate accelerator streams retain their final
owners together, but their cold preparation phases are serialized by the
Pipeline builder. `accumulate_serial_memory` is the sole prepared-owner
composition authority for this lifetime and uses:

```text
current = sum(current_i)
peak    = current + max_i(peak_i - current_i, 0)
```

`cumulative` and `reused` remain saturating sums and `budget` remains the
largest owner budget. This law also covers active/pending prepared Job storage.
It prevents two mutually exclusive finalizer workspaces from being reported as
simultaneously live while retaining every final current owner. Storage that is
actually concurrent within one owner continues to use additive composition.
Metal Pipeline ICB storage is the explicit exception to “unqueryable” native
bookkeeping: adapter opening probes `allocatedSize` for the exact descriptor at
all power-of-two capacities from 1 through 65,536. For command count `D`, the
reservation charges every full 65,536-command class plus the
next-power-of-two tail class exactly, exposes their sum as
`backend_command_native_bytes`, counts the native objects as
`backend_command_chunk_count`, and adds one 16-byte retained host record per
chunk. Materialization repeats the same decomposition, requires each actual
`size` and `allocatedSize` to equal the adapter calibration row, and gates the
actual chunk/byte totals against the frozen upper. Those device-derived
allocation dimensions remain outside semantic fingerprint identity and are
instead checked field by field under that identity.

View raw-word slots may serve sequential uses of different scalar types and
retain the strongest natural scalar alignment among those uses. They are
suballocated at the maximum of that alignment and the selected backend's
storage-offset alignment, so both alignment holes and backing-owner count are
known before allocation.

Pipeline logical, live, and physical workspace reports share the fixed base
`B = state_bytes + prepared_bytes`; `persistent_bytes` is excluded because its
Buffers remain caller-owned. The logical term adds each ordinary Program's
`graph_info.memory.logical_bytes` per occurrence and, for every nested group,
adds `K * (L_seed + N * L_action + L_fold)`. The live term adds only the
maximum `graph_info.memory.live_bytes` across executable ordinary, Seed,
Action, and Fold templates. The physical term adds `transient_bytes` and is
exactly `peak_bytes`. Every multiplication and addition is checked and no
report is clamped to another report. The public equations and symbols are
owned by [Compute](../../../../docs/reference/compute.md).

The scratch planner consumes the admitted Kernel operation sequence and emits
the exact temporary requests used by Scan, segmented Scan/Reduce, Sort,
Compact, Partition, Reduce, and ScatterReduce. For storage-page size `P`,
alignment `A`, and ordered request `r`, it places each operation's requests in
the first page whose aligned end remains at most `P`; otherwise it appends one
page. Operation boundaries reset placement because canonical dispatch barriers
close the prior temporary lifetime. If the largest operation envelope uses `q`
pages and, among operations with that page count, the maximum aligned terminal
extent is `L`, the Program requires

```text
scratch(Program) = (q - 1) * P + L
```

Pipeline steps are serial, so the Pipeline arena repeats the same maximum
envelope across Programs. Sequential operations, Programs, recurrence phases,
and transactional alternates therefore share the same View and scratch owners.
Planning evaluates each distinct Program's immutable Kernel operation sequence
once in first-declaration order; repeated recurrence routes reuse that result
instead of rescanning the Program graph. This changes cold planning from a
route-times-graph walk to one graph walk per distinct Program while preserving
the same maximum envelope and first-failure priority.
The best-fit pass retains one owner-usage vector across all steps and clears
its logical contents between placements; step count does not create allocator
traffic for that workspace.
`pipeline/plan/space.hpp` is the sole word-limit and overflow-safe alignment
authority shared by transient and View placement.
For `L` logical chunk occurrences and `U_s` physical owners touched by step
`s`, peak-envelope measurement visits `L + sum(U_s) <= 2L` entries. It reuses
the placement workspace as the touched-owner list and never clears or sums all
owners for every step.
Prepared Jobs may only borrow their sealed offsets. A single request larger
than `P` is rejected before allocation; no private backend allocation path
remains for Pipeline scratch. `scratch_bytes` and `scratch_count` continue to
describe the shared Buffer arena used by accelerator primitives. CPU
Map/collective/primitive execution storage instead has one allocation-free
plan per distinct Program whose mutable maximum envelopes are merged into one
Pipeline-wide `CpuPreparedArena`. Compact Program wrappers and immutable
indices remain additive; worker/tile state, collective arrays, and primitive
slabs are materialized once for the complete serial Pipeline and charged to
`prepared_host_bytes` and `prepared_tile_bytes`. They are never
occurrence-owned or disguised as accelerator scratch pages.
The exact Metal nested-aggregate specialization follows the same ownership
law without adding a scratch page. Common admission must identify two dense,
non-overlapping U32 Seed intermediate ranges, each with capacity at least the
outer-window bound `K`. Because the specialization replaces that complete
Seed/Action/Fold stream, those otherwise idle plan-owned ranges hold the
aggregate's low-word and status SoA rows for the duration of the two-command
ICB. This is `8K` bytes of logical reuse inside the already counted transient
arena, not a new allocation or an addition to `prepared_bytes`,
`scratch_bytes`, or `allocation_count`. A missing, aliased, strided, or short
range rejects the specialization and keeps the canonical stream.
`publish_bytes` reports terminal copy traffic and is not retained storage.
The largest and peak step/iteration/chunk coordinates identify Program
workspace, while `view_bytes` and its step/iteration/binding coordinates
identify the dominating View requirement. The same selected View reports
`view_span_bytes`, `view_backing_bytes`, `view_offset_bytes`,
`view_stride_bytes`, `view_element_bytes`, `view_count`, and
`view_alignment`. Together with `DeviceInfo::storage_bytes` and
`storage_alignment`, these fields make descriptor admission attributable
without backend-private headers.

`MemoryBudget{bytes}` compares the fixed `PipelinePlan::peak_bytes`.
Failure returns `PipelineMemoryBudget` before Pipeline-owned state, workspace,
View/scratch Buffers, CPU Program-run storage, private Jobs, template-registry
state, or accelerator command materialization. The same accelerator
reservation is shared by primary and transactional-alternate construction;
route charges accumulate per stream while an equal immutable template is
charged once. Every byte component and descriptor/native-object count is
rechecked against that single limit before the corresponding backend
allocation. Vulkan therefore either stays inside the admitted structural
reservation or returns a Compute failure before command/descriptor expansion;
there is no unbounded retry or process-terminating fallback.

Recurrence planning keeps route and immutable-template dimensions separate.
For every normalized terminal/history variant class `E`, its exact group
capacity is `sum(route_copies(r) * group_count(r), r in E)`, where
`route_copies` is the frozen one- or two-stream generation stride. Metal owns
one fixed group table at that capacity. Vulkan owns the same table plus
`capacity * dispatch_window_count` descriptor sets. These template-owned
dimensions are charged once per class; only route storage scales per stream.
Any arithmetic overflow, undersized fixed capacity, or descriptor/native count
that exceeds the reservation rejects before the corresponding allocation.
Pipeline materialization cannot manufacture an implicit local budget. A
missing or invalid frozen registry reservation is a common-accounting failure
before backend registry binding; every valid Metal/Vulkan path therefore
passes through the same public-plan limit.

Backend cold structure keeps compact descriptions and physical commands as
different dimensions. For canonical route `r`, let `a_r` be its compact
authored-entry count in one stream, `o_r` its physical occurrence count,
`d_r` its Program-step description count, and `q_r` its frozen one- or
two-stream `route_copies`. Retained step descriptions are exactly

`D = sum(q_r * a_r * d_r)`.

Retained status-source, status-entry, and telemetry-description counts use the
same `q_r * a_r` scale. Status commands, telemetry commands, and
occurrence-local status parameter payload instead use `q_r * o_r`; backend ABI
parameters that are not status-owned are added as their own checked command
payload. No retained description is charged as an outer-times-inner physical
command, and no occurrence command is hidden inside a retained description
count. Before any corresponding native reserve or allocation, Metal and
Vulkan compare the actual described/captured count against each independently
frozen field and reject `compute_pipeline_capacity` at the stable route/node
location on mismatch.

The body/view dispatch field is the backend physical capture upper, not the
logical ComputePlan dispatch statistic. Primitive stage expansion and dense
View normalization are included there; reset, status, telemetry, window
control, and publication commands remain independently counted. Vulkan's
deliberate 1,024 retained-template-step compile ceiling does not prevent safe
plan-only inspection. Materialization checks that ceiling during common
accounting, before registry publication or a Vulkan call, and preserves the
crossing template and Program-node coordinate in the public failure. Its
stable native reason key is `compute_pipeline_template_step_capacity`; the
public projection is `PipelineCapacity`, not a generic lowering failure.

`PipelinePlan::scratch_bytes` and `scratch_count` expose the logical
accelerator scratch backing and page count. `allocation_count` remains the
retained Buffer-owner count and is not repurposed as a host/native allocation
counter. `Pipeline::memory_snapshot()` enumerates shared owners once and
labels scratch separately in Resident and Device categories; Device bytes are
the actual physical allocation and may exceed logical Buffer payload through
backend allocation granularity. Host/native current and peak telemetry are
compared with their planned reservations after preparation. Native allocation
failure retains its typed public Reason plus stable preparation location and
native reason key when the rejecting route is known.

### Device Pipeline Admission

`DevicePipelineMemoryLimit{bytes}` configures one aggregate Pipeline admission
boundary when a Device opens. The public overloads are additive:

```text
open(Target, DevicePipelineMemoryLimit)
open(Target, Compile, DevicePipelineMemoryLimit)
open(Session&, Target, DevicePipelineMemoryLimit)
```

The existing open forms construct the same accounting authority with capacity
`UINT64_MAX`; “unlimited by configuration” does not bypass accounting. A zero
limit fails Device open with `DevicePipelineMemoryCapacity`. Device open also
captures the positive POSIX host page size once in `DeviceState`; every
Pipeline backing calculation consumes that immutable fact and does not call
`sysconf` again. Failure to obtain that fact rejects Device open with
`DeviceCapacity`.

For one frozen Pipeline plan, let

```text
Q = PipelinePlan::committed_peak_bytes
P = PipelineMemoryPlan::publication_committed_bytes
0 <= P <= Q
```

The frozen plan computes resource Buffer count/bytes/lifetime, step View
geometry, publication hazards, command contribution, and target ownership from
the ordered resolved resource, step-resource, publication, and state-control
records. Publication alternatives carry only source/bank and target Views plus
their state/output coordinates; the referenced state control alone carries
count, `Max`, `Tile`, expected terminal value, and final parity. Memory planning
does not re-read authored bindings or publication scalars after those records
have been sealed. Before sealing, a publication source is only a typed
window-control/logical-output coordinate. The ordinary control's sole
`ordinary_step` or the nested `NestedTemplateShape`'s projected `fold_first`,
together with the
shared output projection and final-bank resolver, selects its canonical
producer; no derived terminal-step copy, base-step mirror, or full source
binding survives beside the step-resource table. A target-only shared owner is a cold
materialization locator and is cleared after `PipelineResource` assumes
lifetime ownership. Thus accounting, admission, CPU execution, and backend
preparation cannot each project a different offset, count, stride, owner,
bound, or final bank from the authored edge.

`Q` is the sole Device charge expression. It is the conservative sum of the
plan's individually page-rounded explicit mappings plus its separately owned
exact commitments. Device admission never substitutes `peak_bytes`, rounds
`peak_bytes` as one mapping, probes current process memory, or treats zero as a
fallback sentinel. The Pipeline-local `MemoryBudget` comparison against
`peak_bytes` runs first. Only after that succeeds does prepare reserve exactly
`Q` from the Device Budget, before materializing a Pipeline-owned Buffer,
host/CPU execution owner, private Job, or native command owner. A capacity miss
is fail-fast `DevicePipelineMemoryCapacity`; there is no wait queue, retry, or
second governor. The storage hierarchy mutex makes competing reserves
linearizable, so at most the capacity-admitted set crosses materialization.

One successful reserve remains the only capacity decision. After preparation
and restore complete, `Reservation::partition(P)` divides that existing ticket
without allocation or re-admission. The `P` ticket is committed with
`physical_bytes = 0, allocated_bytes = P` and retained by
`PipelinePublicationState`; the exact `Q - P` ticket is committed the same way
and retained by `PipelineState`. Planned conservative bytes are not presented
as producer-measured physical allocation. Reverse declaration order destroys
all owners before releasing their ticket, and the Device owner itself remains
alive through both releases. A failed prepare owns its unsplit ticket outside
the complete build/preparation owner, so all partially materialized state is
destroyed before the reservation is refunded.

A cold restore may adopt an already-live publication authority when its ordered
owners are identical. Its publication ticket must already be committed for
exactly `P` on the same Device. Missing, uncommitted, nonzero-physical, or
differently charged tickets fail closed; no inferred or peak-based fallback is
allowed. After the obsolete target authority has been destroyed, the new
reserve's duplicate `P` partition is refunded and only its private `Q - P`
ticket is committed. Thus multiple Pipeline wrappers share one publication
charge, while a `LatestDeviceState` keeps that charge and the resident owners
alive after every Pipeline wrapper is destroyed.

`Device::pipeline_memory()` returns the narrow synchronized
`DevicePipelineMemoryReport`. Its `committed_bytes` and `preparing_bytes` map
to storage allocated and reserved charges; capacity, availability, peaks, and
operation counters come from that same snapshot. `admission_count` counts
accepted root reservations. Commit and release counts are ticket operations,
so a successful split may contribute a publication and a private operation;
partition itself never increments admission. `Device::memory()` remains the
independent actual host/device/transfer allocator and backend telemetry
surface. Neither report overwrites or derives the other.

The frozen preflight plan also owns one compact Pipeline boundary vector.
Canonical `resource::analyze` hazards and shared-workspace reuse set its bits;
`PipelinePlan::barrier_count` is their exact population count. Preparation
consumes the same resource plan and vector, so command-reference capacity,
backend expansion, and allocation placement cannot invent a second barrier
count.

Budget cannot change `Max`, `Tile`, occurrence count, Program shape, chunk
placement, or backend. CPU, Metal, and Vulkan consume the same canonical graph
and Pipeline ownership law; the selected backend contributes only the dense
View requirements its binding ABI needs.

All fixed `K = ceil(Max / Tile)` execution entries reuse the body Program, one
Program workspace, and one primitive scratch arena. Placement changes only
physical addresses. Operation order, reduction tree, numeric policy, overflow
fold, barrier frontier, and publication order remain unchanged, so
suballocation cannot change result bits. A warm Pipeline run changes only
resident control values; it performs no payload allocation, plan mutation, or
placement search.

For a nested `tile_repeat<N>` body, let Action input tuple
`T = P || Q`, where `P` is the Action output prefix and `Q` is its invariant
tail. Let `K = ceil(Max / Tile)`. The cold planner retains disjoint owner
families with the following conceptual decomposition:

```text
NestedFootprint =
    PersistentCompactQueue
  + OuterState(O)
  + ScheduleControl(K + N)
  + bytes(Q)
  + 2 * bytes(P)
  + max_X Workspace(X)
  + max_X View(X)
  + max_X Scratch(X)
  + PreparedRouteMetadata(K + N)
  + PipelineCpuPreparedArena
  + UniqueProgramDescriptors
  + BackendMaterializationReservation

where X is one of {Seed, Action, Fold}
```

Each `max_X` is the capacity of one serially reused owner family, not a sum
over Programs. `PersistentCompactQueue` is the one caller-owned queue and
count already included in `persistent_bytes`; Pipeline retains its owner but
does not duplicate its payload. `OuterState(O)` is the ordinary
publication/transaction state required for `O` and does not grow with `K`.
`ScheduleControl` is the fixed-width resident selector, status,
command-reference, and outer/inner schedule payload.
`PreparedRouteMetadata` contains only occurrence-specific bindings and
coordinates. `PipelineCpuPreparedArena` contains the one merged mutable CPU
worker/tile, collective, and primitive envelope: reusable slabs are maximums,
while persistent descriptor objects and transform tables are additive.
`UniqueProgramDescriptors` contains the compact CPU run wrappers and immutable
indices plus immutable backend-template state once per distinct
Program/template variant. `BackendMaterializationReservation`
contains the checked command, descriptor, parameter, status, and profile
structure required by the selected backend. The plan accounts for resident
control Buffers in `state_bytes`, shared Program workspace in
`transient_bytes`, dense View/scratch backing in `prepared_buffer_bytes`, and
all other cold retained preparation owners in the remaining prepared
components. Consequently `peak_bytes` excludes the caller-owned compact queue
while `total_bytes` includes it once through `persistent_bytes`.

`BackendMaterializationReservation` is monotonic but is not required to be an
affine function of `K`, `N`, or physical command count. Metal adds the exact
calibrated ICB size-class term and compact chunk records, so crossing a
power-of-two tail class changes the finite-difference slope. Cross-backend
window contracts therefore own exact compact template/command counts,
component reconciliation, and monotonic retained growth; the Metal ICB planner
contract remains the sole authority for the piecewise byte equation. Repeating
that equation as an assumed constant per-route byte coefficient would create a
second, device-inaccurate reservation authority.

Seed writes the initial `P` bank and the sole `Q` bank. The two `P` banks then
alternate for all Action iterations and all outer windows; Fold borrows the
selected `P || Q` view. No owner is sized as `Max * bytes(T)`, `K * Tile`
scratch, `N` workspaces, or `K * N` Jobs or banks. Compiled Program ownership
is three immutable handles regardless of both bounds. CPU mutable execution
storage is one Pipeline-wide arena; serial Seed, Action, Fold, outer, and inner
routes retain only their frozen binding route and Program descriptor and borrow
that arena. Route-template and prepared-metadata growth is `O(K + N)`. Native
command references and observed dispatches are accounted independently and
cannot be presented as a duplicated Program graph, worker scratch, collective
state, or primitive payload owner.

Within one `CpuMapRun`, the simultaneously live worker scratch is one
contiguous `workers * scratch_words` slab. The worker ordinal selects its
checked fixed-stride slice; there is no heap owner or pointer lookup per
worker. Preflight and materialization consume the same overflow-checked
product, and retained-memory observation counts that slab once. This preserves
worker disjointness while removing `workers - 1` cold allocations per Map
without changing tile order, SIMD counters, or warm execution storage.

The Program-owned `CpuGraphStorage` retains compact Map and collective run
wrappers in two exactly reserved dense arrays. Their mutable worker, tile,
collective, and scratch spans borrow the Pipeline-wide `CpuPreparedArena`;
these arrays are not a second payload owner. Immutable step-to-dense indices
provide constant-time lookup; a missing step uses the canonical invalid index.
The planner charges the dense element bytes and both index arrays before
materialization, and construction reserves the proved active counts before the
first emplacement. There is therefore no per-Map or per-collective owner
allocation, vector relocation, or warm pointer chase. This changes allocation
topology only: Program order, step identity, serial scratch ownership, and
retained byte observation remain unchanged.

Nested planning preserves two coordinates. The plan publishes `K`, `Tile`,
and `N`, and every largest-workspace, peak-envelope, and View location records
outer window and inner iteration independently when applicable. Seed and Fold
locations retain their phase without inventing an Action iteration. A
flattened `k * N + j` value is neither a location nor an admission count.
Each coordinate is range-checked, then only the compact `O(K + N)` route
total contributes to `PipelineRouteCapacity`; flat-only schedules retain
`PipelineIterationCapacity`, and the product contributes to neither.
Native command capacity is checked separately against the selected backend's
published limit.

All shared K/N cardinality is projected from one constructor-closed
`NestedTemplateShape`. It owns phase spans, compact and retained route counts,
per-route occurrence counts, authored and transduced command totals,
outer-to-Fold selection, and Action parity ownership. Build, frozen plan, and
runtime state copy that pointer-free value as a phase handoff. Public memory
planning and backend preparation may validate the copy against the sealed
Window control by invoking the same constructor, but may not rebuild
`ceil(Max/Tile)`, `K * (N + 2)`, Fold bank counts, or phase boundaries. The
runtime `NestedTemplateGeometry` combines the shape with exact resident
count/terminal handle identity; backend-native descriptor, ICB, alignment, and
limit equations remain specialized and are not aliases of the shared shape.
Fold routes with zero projected occurrences remain valid retained templates;
validity comes from successful shape projection, while accounting scales their
execution contribution by zero.
Window-publication count and backend command reservation consume the same
Shape-projected `outer_bound`; neither memory planning nor backend preparation
recomputes K from the retained copy geometry `(Max, Tile)`.

The zero-count plan owns the same cold capacities but executes no Seed,
Action, or Fold. Only the explicit recurrent `O` output prefix participates in
the zero-count bank seal; append-only `W` outputs are neither matched to Fold
inputs nor copied or published. A tail window changes only live count, not any
retained capacity. Count overflow fails before payload writes, so it cannot
make a speculative tile owner live or publish an outer bank. A Seed, Action,
or Fold failure is folded into the fixed control owner before the corresponding
mutable route is reused. Warm CPU, Metal, and Vulkan execution may change
control contents and counters, but it cannot allocate, resize, rebind, or
re-place any owner.

Here, rebind means a post-prepare mutation of retained Job, Buffer, typed View,
arena descriptor, or prepared-pipeline owner identity. Executing the frozen
Metal size-class ICB chunks from a fresh single-use outer command buffer does
not change that identity and is not a rebind. The public
`rebinding_count` is therefore zero by construction, not sufficient evidence
by itself. The nested-window contract fixture captures the unique normal and
transactional alternate Jobs across its nested and transactional binding
oracles, their Program/workspace/Buffer owners and View descriptors, shared
arena bindings, and the available primary/alternate opaque prepared-pipeline
owners; every identity must compare equal after successive executions, and
the nested identities must also survive overflow.

That definition must not erase Metal's host boundary from the execution model.
The Metal warm path creates the required outer command buffer and encoder,
passes the frozen unique-resource array through one bulk residency call, walks
only `C = ceil(D / 65,536)` compact 16-byte chunk records, executes their ICB
ranges, commits, and observes completion. It visits no command, binding,
indirect-grid, or recurrence-state table and restores no bytes.
`rebinding_count` remains an identity-mutation counter; zero descriptor-schedule
traversal is proved by the fixed-chunk executor structure, while the chunk
calls, bulk residency call, and submission CPU work remain explicit.

Metal cold finalization owns one non-retained pointer-index workspace. With
`D` captured commands and `B` captured Buffer-binding rows, the one table has
`bit_ceil(2 * max(D, B))` pointer slots and is cleared and reused between the
pipeline-state and resource passes. Resolved native window rows remain live
through this pass, so `host_transient_bytes` adds their frozen structural upper
to the pointer table rather than taking the larger of the two. Runtime memory
evidence uses the rows' actual exact-reserved capacity. Source specialization
remains a separate serialized high-water and neither workspace can become part
of the frozen warm owner.

The small admission proofs use no heap mirror: deferred nested Fold state and
profile occurrence counts use the fixed Pipeline route-capacity stack
envelope, direct-aggregate status counting uses twelve inline binding rows,
and publication resolution uses the public 32-leaf array. Status replacement
lookup borrows a frozen status-table span; occurrence coordinates are written
into that same slice only until `setBytes` snapshots it and are then restored.
Status and telemetry description vectors reserve their checked compact-entry
upper once. Their encoded control commands and parameter snapshots consume the
separate physical-occurrence upper; neither dimension substitutes for the
other.
Metal Pipeline capture derives Buffer-binding storage from the command
producers rather than from the 31-slot ABI envelope. Each operation manifest
publishes `u_r`, its highest authored non-guard argument index plus one; a
dense View transfer contributes three slots only when that transfer exists.
Map derives its value instead of using a primitive constant:

```text
u_map = max(input_count + output_count + 2,
            controlled ? 6 : 0,
            has_unique_checks ? unique_check_count + 4 : 0)
```

The additions are checked and `u_map <= 30`, reserving index 30 exclusively
for the Pipeline guard. Route occurrence expansion repeats a prefix but does
not add index spaces, so route and stream composition use `max(u_r)`. Active
Pipeline producers extend that same maximum: open/close use four, status reset
uses three, status fold uses six or seven with profiling, telemetry uses nine,
and window publication/canonicalization, terminal publication, and nested
aggregate control use eight. For `D` planned commands and resulting non-guard
prefix `U`, the frozen Buffer-binding-row upper is

```text
B = D * (U + 1)
```

where the extra row is the guard upper; an explicitly unguarded aggregate may
consume less. Capture preallocates exactly `D` command rows and `B` binding
rows before the first encoder callback. `D` adds the body/view dispatch upper
once, then reset, status, telemetry, one Metal window-control command per
window, the common exact publication-command upper, status reset, and the two
open/close commands. In particular, the body/view window-capture subset is not
added a second time. `backend_parameter_bytes` is both the payload upper and
the sole parameter-vector capacity upper. Before resize, capture proves
`aligned_offset + length <= backend_parameter_bytes`; capacity grows from 256
bytes by checked doubling capped at that frozen upper, and every reserve must
return the requested capacity exactly. Capture rejects any producer index
`>= U`, and checks every append and the final actual size/capacity against the
frozen uppers. Thus command and binding rows have no geometric growth, while
parameter growth has a deterministic planned ceiling and no allocator-dependent
slack. The current Pipeline producers call no dynamic threadgroup-memory
binding API, so the former threadgroup row, mask, snapshot, planner field, and
ICB replay path do not exist. A future producer requiring dynamic threadgroup
memory must first add an explicit producer manifest and frozen capacity law.

Map source specialization has no heap scratch owner. Admission bounds the
binding count by `kMaxComputeBindingCount`; specialization stores at most two
decimal-literal edits per binding in one fixed stack array, canonicalizes that
active span in place, counts the final recipe exactly, and allocates only the
retained source string. A canonical Map therefore contributes zero
source-string scratch. A recurrence miss copies only the canonical Map metadata
needed to validate and specialize its minimal one-shot artifact; that
separately frozen metadata envelope is its only `source_transient_bytes`
contribution and is destroyed before native compilation. The source allocation
itself is reserved immediately to the final backend upper, edited in place,
moved into the executable cache, and charged once as retained template
storage. The retained source upper still covers the maximum 20-digit unsigned
base and stride literals; the backend recipe extends that scalar separately for
any controlled/private wrapper reserve.

Source text cardinality and `std::string` allocation are separate accounting
dimensions. For a planned final text upper `T`, the retained host reservation
uses the conservative external-storage envelope `T + 64`; runtime measures
the actual external capacity (including the terminator) and rejects before
cache publication if it exceeds that envelope. Exact authored UTF-8 bytes stay
reported as template-source bytes. A non-Map Metal source that must grow from
raw text `R` into its Pipeline-private ABI additionally charges `R + 64` as
the serialized source-transient high-water. Map main, control, and check
recipes reserve their final wrapped envelope during their single
materialization, so they retain the zero-transient law above.

Metal template-registry telemetry walks each published type-erased owner once.
The leading template discriminator selects either the Program wrapper or the
Map-recurrence wrapper; observation then counts the wrapper and every runD-owned
vector by its actual capacity, including Program steps, Map window/stride/check
plans. A recurrence wrapper retains only its immutable Program signature,
terminal/history discriminator, and native Map owner; binding layout, source
recipe, and history pitch identity are reconstructed by the common normalized
plan during cold lookup and have no second retained Metal copy. Fixed primitive
pipeline tuples are counted as their runD wrapper. The `shared_ptr` targets for `MTLLibrary` and
`MTLComputePipelineState` are opaque driver objects and contribute no invented
host byte estimate. Specialized source strings are owned by the adapter-global
source/library cache rather than the Pipeline registry, so that cache remains
their sole observation authority and registry telemetry cannot double-count
them across Pipelines.

Vulkan template-registry telemetry follows the same single-owner rule. It
observes the published recurrence wrapper, its fixed group metadata, exact
descriptor-set and descriptor arrays, command buffers, and runD-owned native
handle arrays by retained capacity. The wrapper retains only the immutable
Program signature and terminal/history selector required to rebuild the common
normalized plan; it does not mirror binding, source-plan, history-pitch, or
route identity. Opaque driver-private allocations remain native object counts,
not fabricated byte estimates.

The planner admits a nonzero arena range only after a dense `Full` write or a
proved `Domain` write. Domain coverage uses canonical count identity and
`Resource::parent` lineage, not a mirrored capacity number. If the reader count
is the writer count or its descendant, the read prefix is a mathematical subset
of the written prefix. Writer predicates must also match. Unproved partial
lifetimes carry their first writer in `reset_node` and enter the same aligned
lifetime arena. Every logical reset range resets exactly once at that writer
even when disjoint lifetimes share an offset or Buffer.

For reusable ranges, canonical 256-byte-aligned best-fit releases storage only
when `last_use < next.first_use`. This placement is the sole reusable-storage
authority. The planner compares the ordinary
packed arena with its same-node destructive Map placement and keeps the smaller
arena extent; a tie prefers the proved destructive alias.
The destructive proof requires one eligible dense full-write output and at
least one arena-backed, same-shape source whose final read is that node. The
Map must be pointwise, so each candidate read uses the same logical ordinal as
the output write. When multiple sources qualify, the smallest canonical
resource ID wins; `Resource::source` records that candidate. An
indexed Map evaluates `output[i] = f(source[index[i]])`. Unless identity
indexing is separately proved, a write by lane `j` can precede another lane's
read of `source[j]`. Therefore a Map containing `ReadAt` is non-destructive.
Its storage may still use the ordinary arena once the strict closed-lifetime
condition `last_use < next.first_use` holds.
Address-ordered and aligned-fit ordered trees coalesce and select free ranges
in `O(log R)` per resource. Together with the CSR use index, planning is
`O(R + A + R log R)` time and `O(R + A)` memory instead of rescanning or
resorting all free ranges for every placement.
The CSR count array becomes its fill cursor after prefix offsets are sealed;
there is no second `R * sizeof(size_t)` cursor owner. One `Live{start, stop}`
array owns the node sweep, and arena membership is derived from the validated
internal visibility and nonempty lifetime rather than stored as another bit
set. Arena packing performs one stable ordinary/large partition in an
`R`-entry ID array, preserving canonical ID order within each class. The
published candidate `Layout` directly owns final owner IDs and owner-local
offsets; virtual placement offsets are localized in place and are not copied
into a parallel result. Active heaps reserve the admitted class cardinality.
The large-class min-end heap is formed incrementally in that reserved vector
with `push_heap` and `pop_heap`; it never bulk-heapifies an empty reserved
range. Ordinary chunk measurement sizes owner and extent storage from the
proved virtual upper bound before its single sweep.
Accelerator binding admission proves the resulting two-dimensional
memory-by-lifetime layout with an exact sweep: physical intervals are visited
in address order, expired intervals leave a min-end heap, and a compressed
step tree rejects any closed-lifetime intersection. This is `O(R log R)` time
and `O(R)` cold memory, replacing the quadratic pairwise reset scan without
weakening overlap rejection. External reset ranges remain non-aliasable.
Ordinary virtual placement is cut at
`min(1 GiB, backend storage limit)` boundaries. No ordinary range crosses a
chunk, so a backend with a multi-gigabyte single-Buffer limit cannot turn an
entire large Program arena into one driver residency object. A logical range
larger than that ordinary chunk but no larger than the backend storage limit
receives an offset-zero owner and cannot share it with an overlapping
lifetime. Expired large owners are reselected by deterministic best fit over
`(capacity, owner ID)`, so nonoverlapping assignments contribute their maximum
size rather than their sum. A proved pointwise destructive transition extends
the same large owner instead of allocating another one.
The release/selection pass is `O(L log L)` for `L` large ranges. The selected
chunk IDs, large-owner IDs, and local offsets are fingerprinted and consumed by
CPU, Metal, and Vulkan. Used owner extents, not virtual boundary gaps, form
`physical_bytes`; retained ordinary chunks and dedicated large owners form
`allocation_count`. Placement depends only on the canonical resource order,
closed lifetimes, destructive proof, fixed chunk ceiling, and frozen storage
limit. It does not inspect active work, timing, cache state, or runtime memory
pressure and therefore is not a workload-size strategy.
Pipeline consumes these owners directly. Its one cross-step placement pass may
reuse an owner or pack ordinary chunks within the same ceiling, but it never
post-coalesces the result into a backend-limit-sized Buffer.
The local contract proves the scale boundary without allocating the scale
payload: one synthetic step declares eight simultaneously live maximum
ordinary owners and must retain exactly eight offset-zero owners. Separate
small CPU, Metal, and Vulkan executions prove that the same planner coordinates
are consumed correctly. This decomposes placement scale from execution
semantics; it does not introduce a reduced execution graph or runtime mode.
Ordinary and destructive placement preserve the same materialized values,
dispatches, memory traffic, and arithmetic. The implemented cost is therefore
the exact lexicographic pair `(physical_bytes, allocation_count)`, with a tie
preferring the proved destructive transition. The planner does not label arena
packing as fusion, streaming, or rematerialization.
Every reuse frontier becomes an explicit graph barrier. CPU, Metal, and Vulkan
consume the same raw owner and byte offsets, while typed routes retain element
width and Fixed policy. Storage identity never becomes a scalar operand or
reduction-order input.

An internal lifetime's first-use node must contain a dense write and no read of
that value; a same-node read/write birth is rejected because the graph defines
no intra-node access order. Resource uses are indexed once in contiguous CSR
form, so liveness and Domain checks are `O(R + A)` instead of rescanning every
node for every resource. Plan evidence does not replace live `MemoryStats`, and
no allocation or graph reconstruction occurs on warm runs.

Logical output order and physical output ownership are separate. Repeating one
value, or returning an identity map beside its source, records another logical
projection to the same physical output. It does not create a map node, output
buffer, dispatch, or payload copy. An external input still materializes once at
the output boundary because input and output storage roles are distinct; any
repeated logical fields then alias that one materialized output. With at most
16 outputs, projection construction is bounded compile-time work and adds zero
warm execution traffic. `compute/program/output.hpp` owns the one bounded
logical-count and logical-to-physical index rule consumed by introspection,
convenience execution, Job reads, and Pipeline planning; those paths do not
repeat or reinterpret the alias vector.

The convenience cache contains at most one ordinary read-only Job and is
serialized by one Program-owned gate. Host-input execution updates its existing
input buffers in place and allocates only the returned typed vector. Caller-
Buffer execution reuses the cached Job when all buffer identities match and
performs zero warm SDK heap allocations. Switching between host and Buffer
forms, or changing Buffer identities, replaces that one Job. The cached Job's
back-reference is lifetime-borrowed from its enclosing Program, so ownership is
acyclic; a returned `Run` receipt receives its own strong Program owner before
crossing the public boundary. Explicit resident Jobs remain the concurrency
surface and never lock the convenience cache. The `compute.reuse` contract
executes two distinct host inputs through one Program while a third thread
observes its memory, proving that the serialized convenience surface cannot
cross-contaminate inputs or race its cached owner. It also proves both sides of
the lifetime boundary: destroying a host-run Program releases the borrowed
cache, while a Buffer-backed `Run` remains readable after its public Program
handle is destroyed.

Caller-Buffer convenience execution keeps one dynamic ownership authority.
The structural binding pass checks shape, type, alias, and Device identity;
the immediately following Device claim acquisition performs the poison and
busy checks atomically, so validation does not take one poison lock per
binding. A cache miss constructs the Job from that already validated binding
set without repeating the same scan. Terminal write generation or failure
poison and release are then published in one claim-vector pass under one
Device-gate acquisition, with publication ordered before writer release.

`Program::memory()` walks each retained owner once. `ProgramState` owns its
top-level vector allocations, including one canonical U32 chunk-rank
permutation, and nested `graph::Info` capacities. A CPU Program then owns one
`CpuGraphProgram`, one uniquely owned `CpuRuntimeGraph`, every live `CpuProgram`
and `CpuCollective`, and their Kernel/CPU-SIMD private allocations.
Each `CpuProgram` owns its `ComputeMap` descriptor, compact `PreparedRun`, tile
executor handle, and scalar tile/scratch configuration. Compilation publishes a
dedicated `CpuRuntimeGraph`: its values are `(type, count)` pairs and its ordered
steps are Map, Scan, or Primitive alternatives. A Map step owns input/output
value-ID routes. A Primitive step owns its routes, selected output, operation
kind, and one validated active Kernel plan. Planning happens once during
compilation; execution consumes the published plan. The reusable Flow
construction graph remains caller-owned and reusable.

`Job::memory_snapshot()` also measures each Buffer owner once. The same walk
adds the per-buffer snapshot rows and saturating resident/physical/reuse totals;
the completed totals then publish the snapshot summary. With `B` active,
pending-input, internal, and output Buffer owners, buffer observation is
`Theta(B)` and exactly `B` `measure_buffer` calls, rather than one summary walk
followed by a second row-emission walk. Entry order, truncation accounting, and
the summary remain unchanged when the caller supplies fewer row slots.

One Job-local CPU step state is shared by blocking and submitted execution. It
holds the canonical step index, active pass, collective context, statistics,
and the Scan-pass dispatch count carried across completion; it does not retain a second route table,
binding vector, or result buffer. Blocking and submitted transports invoke the
same route/bind/finish owner, so this state adds no warm allocation or payload
copy. The carried Scan dispatch count is one inline saturating-counter input used
to preserve exact zero-work and multi-pass statistics across the asynchronous
completion boundary.

`compute/cpu/view.cpp` is the sole runtime CPU Buffer-View footprint
authority. It projects an owner and authored element View once into a validated
first-byte pointer plus byte base, effective byte stride, logical payload
bytes, count, and element width. Map retains its expected count and scalar
width policy; collective and reference-primitive callers additionally require
a dense footprint; reset and Pipeline publication accept the validated
strided footprint. Bounded control reads additionally require the canonical
graph value to be `U32` or `U64` and its width to match the View; raw arena
storage type is not a second value-type authority. Each caller maps rejection
to its existing public Reason, and no caller retains another footprint state.

For storage size `S`, element offset `o`, count `n`, element stride `s`, and
width `w`, admission requires `w > 0`, `s > 0`, and `o*w` to fit. A zero-count
View admits a base no greater than `S` and touches no pointer. For `n > 0`, the
single owner proves

```text
((o + (n - 1)*s)*w + w) <= S
```

without evaluating an overflowing sum or product. After `b = o*w` and
`r = S - b`, it rejects unless `w <= r` and evaluates the equivalent predicate

```text
n - 1 <= ((r - w) / w) / s
```

All divisors are positive. Integer-floor associativity makes this equivalent
to `(n - 1)*s*w <= r - w`, while the division form cannot overflow. The same
proof makes `n*w` and the effective byte stride safe to materialize. Counts
zero and one canonicalize the irrelevant byte stride to `w`.

Strided reset and Pipeline copy loops consume that proof once, then advance
the validated pointers. They retain one unavoidable payload pass and perform
no per-element overflow or bounds validation. The projection is allocation
free, owns no payload, performs no virtual dispatch, and is not retained by a
Job or Pipeline.

Private `compute/size.hpp` is the single exact host-width addition and
multiplication law used to derive byte counts, offsets, and bounded cardinality
products. It operates on `size_t`, publishes an output only after the complete
operation succeeds, and never saturates or wraps. Callers retain their existing
typed failure and precedence; the helper owns only representability. Exact U64
graph and arena arithmetic instead consumes `kernel/core/checked.hpp`, while
telemetry totals consume the saturating counter law. Those three semantics are
not interchangeable. The CPU View contract exercises `{0, 1, SIZE_MAX}`,
multi-factor products, and unchanged output after failure before its exhaustive
footprint oracle.

Let `R` be the exact logical retained extent of that runtime owner:

```text
R = sizeof(CpuRuntimeGraph)
  + V(values) + V(steps)
  + sum_map(V(inputs) + V(outputs))
  + sum_primitive(V(inputs))
```

The active plan is inline in each Primitive alternative and is therefore
already counted by `V(steps)`. Scan has no nested allocation. The unique owner
adds no shared-reference control block or atomic reference-count traffic.

CPU collective scratch is kind-specific. For prepared tile capacity `T` and
`W = sizeof(CpuCollectiveWide) = 16`, Scan retains `2*T*W` Tile bytes in its
tile-total and prefix arrays, while Reduce retains only `T*W` bytes in its
tile-total array. Reduce never allocates, resizes, or observes a prefix array.
Bounded execution may shrink the live element counts of the arrays but cannot
exceed their prepared capacities; the memory snapshot therefore continues to
report the exact retained capacity. Removing the unused Reduce prefix owner
saves `16*T` bytes, one allocation, and `T` value-initialization stores per
Job without changing tile or merge order.

### Exact CPU Map retained formula

Let `V(x) = capacity(x) * sizeof(x::value_type)` for a vector. These are logical
retained element extents: allocator size classes, padding, and bookkeeping are
excluded. All products and sums saturate at `uint64_t` maximum. For one live
compact CPU Map,
with prepared instruction vector `I`, fixed-format vector `F`, and tile-executor
oracle `E`, the exact contribution is:

```text
Host(Map) = sizeof(CpuProgram)
          + V(I)
          + V(F)
          + E.state_bytes
          + E.async_context_bytes

Tile(Map) = E.workspace_bytes
          + E.failure_slot_bytes
          + E.worker_tile_bytes
```

`ComputeMap`, the dispatch function pointers, `PreparedRun` scalar counts and
flags, `scratch_words`, `workers`, and `tile_size` are inline in
`sizeof(CpuProgram)`. `V(I) + V(F)` is the complete dynamic `PreparedRun`
extent. No binding-count-dependent vector is hidden below that owner: read and
write counts are scalars, while each instruction carries its resolved binding
slot or immediate. Graph identity and tile count are read from their canonical
graph and plan owners.

### Exact CPU worker-scratch formula

Let `N = I.size()`, `M = N + 1`, `L` be the selected SIMD lane count, and
`A = sizeof(std::max_align_t)`. Before allocation, the compact runner requests:

```text
Scratch_raw = M * sizeof(uint8_t)
            + M * sizeof(ValueVec)
            + M * L * sizeof(WideScalar)
            + alignof(ValueVec)
            + alignof(WideScalar)
            + alignof(uint8_t)

Scratch_words = ceil(Scratch_raw / A)
Scratch_requested_per_worker = Scratch_words * A
```

Worker SIMD scratch uses an aligned overwrite buffer whose capacity equals the
requested word count, so telemetry reports exactly
`workers * Scratch_requested_per_worker`. Preparation does not value-initialize
those bytes: every admitted SIMD instruction invalidates its destination and
writes it before any dependent read. Instruction-plan
construction is one `O(N)` Program-preparation operation. A physical tile
executes that prepared schedule, while each run performs an
`O(binding_count)` fixed-view binding update. These are structural bounds, not
a measured wall-clock claim.

### Exact CPU primitive scratch ownership

`CpuGraphStorage::scratch` remains empty when every runtime step needs no
primitive temporary storage. Otherwise it is one compact per-step variant
table whose active alternatives contain typed pointers only. Every pointed-to
descriptor is placement-constructed in the single `CpuPreparedArena`
`primitive_objects` segment; every descriptor is trivially destructible and
borrows typed spans from that same arena. `prepare_cpu_scratch` therefore
performs zero heap allocations.

Let `A_T(n) = n*sizeof(T)`, let `r` and `c` be matrix rows and columns, let `h`
be RHS columns, and let `v = min(r, c)` for thin SVD vectors. The frozen
primitive/mode mapping is:

| Primitive/mode | Arena descriptor | Arena Tile claim |
| --- | --- | --- |
| segmented scan/reduce, compact, gather, histogram, partition, reduce, stencil, matrix | none | `0` |
| transform | lane-width twiddle span | `A_Lane(plan.twiddle_count * 2)` = `plan.workspace_bytes` |
| sort/argsort U32 | U32 sort spans | `A_U32(count) + A_U32(count)` |
| sort/argsort U64 | U64 sort spans | `A_U64(count) + A_U32(count)` |
| scatter | keys/marks spans plus shared epoch pointer | shared `2*A_U32(max scratch_slots) + sizeof(U32)` |
| factor LU or Cholesky | none | `0` |
| factor QR | lane-width QR spans | `A_Lane(r*c) + A_Lane(c*c) + A_Lane(r)` |
| solve from LU or Cholesky factor | none | `0` |
| solve from QR factor | lane-width QR-factor span | `A_Lane(r*h)` |
| solve matrix with LU | lane-width matrix-LU spans | `A_Lane(factor_count) + A_U32(aux_count)` |
| solve matrix with Cholesky | lane-width matrix-Cholesky span | `A_Lane(factor_count)` |
| solve matrix with QR | lane-width matrix-QR spans | `A_Lane(r*h) + 2*A_Lane(r*r) + A_Lane(r)` |
| spectrum Eigen | lane-width Eigen spans | `2*A_Lane(r*r) + A_Lane(r)` |
| spectrum SVD, values only | lane-width SVD-values spans | `2*A_Lane(c*c) + A_Lane(c) + A_U64(c)` |
| spectrum SVD with thin vectors | lane-width SVD-vectors spans | `2*A_Lane(c*c) + A_Lane(c) + A_Lane(r*v) + A_U64(c)` |
| spectrum SVD with full vectors | lane-width SVD-vectors spans | `2*A_Lane(c*c) + A_Lane(c) + A_Lane(r*r) + A_U64(c)` |

Descriptor payload is additive because every prepared primitive keeps its
typed shape, while reusable numeric slabs are component-wise maxima across
serial primitives. Descriptor placement consumes a separately checked
`max_align_t`-padded byte count; `primitive_object_payload_bytes` preserves
the exact logical Host amount and never counts that padding as payload.
Transform tables are additive because their canonical twiddle values persist
per descriptor, while their descriptors and tables still live inside the one
mapping.

Scatter has a dedicated authority rather than borrowing the generic U32 slab.
`PlanScatter` alone derives `scratch_slots` as the least power of two at least
`2*element_count`; Pipeline planning takes the maximum across all serial
Scatter descriptors. Every descriptor binds the matching prefix of one keys
slab and one marks slab and shares one epoch word. Each invocation advances
that epoch, so equal targets in different Scatter operations cannot appear as
duplicates. At the exact `UINT32_MAX` wrap boundary, execution clears the
complete maximum marks slab, not merely the active prefix, then resumes at
epoch one. This state transition is deterministic and performs no allocation.

Every span is admitted only where overwrite-before-read is structural. A
numeric failure may leave an inactive suffix indeterminate, but no consumer
reads that suffix. A lane, key width, factor mode, solve input mode, or
spectrum vector mode mismatch rejects instead of selecting a second scratch
authority. Warm transform execution borrows its generated table and never
regenerates it per run.
Factor QR, solve-from-factor, matrix-input QR, and spectrum workspaces are
reused serially for each batch, so their scratch extent is independent of
`batch_count`. Matrix-input QR consumes its internal row-major Q/R workspaces
directly and retains no generated `Q|R` payload. Matrix-input LU and Cholesky
still retain the generated factors for every batch; their `factor_count`, and
LU `aux_count`, include `batch_count` and are consumed directly from the frozen
`SolvePlan`.

An accelerator Program owns one `AccelProgram` and one immutable kernel token.
Every execution Job owns a distinct fixed-view binding allocation sized from
the Program's frozen graph-binding count. A convenience Job is nested once in
Program memory; resident Jobs report their storage in Job scope. Preparation
fills Job-local storage and writable resident Jobs retain two active/pending
prepared owners. No resident prepare, synchronous run, or asynchronous submit
locks the Program convenience gate. Because the public kernel owner is opaque,
Node authenticates it through a private `shared_ptr` control-block capability
and measures the complete typed token tree once at compilation. Capability
lookup is `O(1)` through the source-only deleter type and stored kernel id; it
uses no process-global token table, mutex, weak entry, or owner-order scan. The
frozen result includes one final execution-step allocation and its nested
artifact, metadata, and overflow-binding allocations. A fused token's retained
owner is the final execution tree plus an inline precomputed
original-dispatch-count scalar. Observation reads that cached result and does
not repeat capability lookup or token traversal.

Accel graph compilation and retained execution use one inline active-operation
variant per node. Inactive primitive descriptor/plan pairs consume zero node
storage, and finalization moves the active pair into the immutable execution
step. Run planning does not copy that pair: its common `ComputePlan` carries the
already admitted dispatch count while CPU, Metal, and Vulkan borrow the same
retained operation. For `N` nodes and primitive payload widths `W_i`, inline
primitive ownership is therefore `O(N * max_i W_i)`, not
`O(N * sum_i W_i)`. The variant is allocation-free and backend-neutral.

Shared Device/context pointees, the separate context and resident-registry
owners, allocator bookkeeping, and physical internal-buffer payload are
different owners and are excluded from Host/Metadata. The kernel token has no
registry entry to exclude. Internal payload is reported exactly once as Host
physical storage for CPU or Device physical storage for an accelerator.

## Counter arithmetic

Every cumulative Compute and accelerator telemetry value consumes the sole
[Counter Arithmetic](../counter.md) owner directly. Compute retains no local
accumulate, release, remaining-value, delta, or capacity-product formula.
Relaxed-atomic gauges apply the same common value law inside their
compare-exchange loop; the successful exchange is the linearization point.
The common absorbing-maximum rule prevents a saturated Metal, Vulkan,
CPU-buffer, or coordinator-frame gauge from reappearing as an apparently exact
smaller value after release or snapshot subtraction.

The law covers public execution statistics, resident-write statistics,
Program-cache event statistics, CPU dispatch/tile/SIMD aggregation, Job
transfer/frame/run accounting, backend runtime counters, and all memory
composition. `WriteStats` contains only produced facts (`copies`, `uploads`,
and `bytes`). SDK allocation growth is proved by the dedicated allocation
oracles and represented by `Stats::buffer_allocations` and `MemoryStats`; no
always-zero host-allocation coordinate is published.

## Observation invariants

- Every capacity product, ownership sum, cumulative counter, gauge release,
  and snapshot delta follows [Counter Arithmetic](../counter.md).
- Inline-string capacity is part of the enclosing object's `sizeof`, so it is
  never counted again. A zero-capacity string counts zero even on a standard
  library that points empty strings at shared static storage outside the object.
  Other externally owned string storage counts capacity plus one terminator
  slot; allocator rounding and bookkeeping remain outside the logical oracle.
- `memory()` and `memory_snapshot()` allocate nothing, perform no backend work,
  and hold only the existing owner-lifetime mutex required by that surface.
- Every public Job result schema retains exactly one shared `JobState` owner.
  That cold construction also allocates exactly one out-of-line
  `JobTerminalState` for the optional `RunState` receipt and failed statistics;
  `Job::memory()` counts it once. Pipeline's private prepared step Job has no
  public receipt, leaves that owner absent, and therefore pays neither its
  bytes nor an allocation. The schema changes read shape, not resident
  execution, typed writes, statistics, or memory observation ownership. A
  multi-output read resolves
  output `i` at the compile-time prefix sum
  `sum(j = 0..i - 1, schema_leaf_count[j])`; it does not rescan schema widths
  at runtime.
- A Node-hosted Job records its coordinator-frame budget from the scheduler's
  immutable configured coroutine-frame limit. Admission reads that one scalar
  in `O(1)` without flushing scheduler batches, scanning resource registries,
  materializing a scheduler statistics snapshot, or projecting a resource
  sub-snapshot. `MemoryCounter::budget` is therefore the configured per-frame
  admission maximum; it is not current or resident frame usage.
- CPU compiled plans and every Job runner are distinct owners and are each
  counted once. A cached convenience Job is nested in Program scope; an
  explicit resident Job is counted only in Job scope. A prepared CPU Pipeline
  owns exactly one `CpuPreparedArena`: Job bindings, route arrays,
  `JobWorkspace` objects and offsets, and the maximum execution/scratch
  envelope are typed slices of that mapping. Jobs and recurrence alternates
  borrow those slices and cannot retain a second vector or heap owner.
- Accelerator binding arrays and standalone prepared resources are Job-owned.
  The one cached convenience Job is included by Program observation, while two
  public resident Jobs share the immutable kernel token but no mutable binding
  or intermediate-buffer owner. Pipeline is the deliberate serial-sharing
  boundary: one frozen workspace-owner map makes every recurrence route borrow
  the same workspace identity, and distinct steps map nonzero Program arenas
  through one deterministic aligned owner placement. On CPU the workspace
  object and its buffer/offset arrays are arena-borrowed; when a Program has no
  chunks, dense Views, or other workspace consumer, the canonical workspace is
  intentionally null. Accelerator workspaces retain their backend-owned
  representation. Host planning and materialization consume the same owner
  map; neither re-derives ownership from mutable iteration fields.
  The Pipeline execution gate and fixed cross-step/cross-occurrence visibility
  frontiers prevent concurrent or stale use. Shared Buffer payload is measured
  once in Pipeline `shared_memory`; reset and full-write chunks share the same
  planned owners and remain excluded from their Job rows.
- A complete Program or Job snapshot sums back to every summary category
  exactly. Host/Metadata and physical Internal rows never overlap.
- Changing only a diagnostic Map label has zero compiled retained-memory delta.
  Deeper IR increases the compact instruction/format owner. An additional
  binding changes the compact descriptor and scalar read count while its
  Program route/`graph::Info` owners produce the retained delta. An additional
  graph step changes topology and tile ownership.

The executable authorities are `compute.memory` and `compute.graph-services`.
They verify compact graph ownership, active-prefix lineage, deterministic arena
offsets, destructive alias admission, CPU/Metal/Vulkan subview parity,
exact runtime-graph and compact-plan capacity bytes, owner deltas, exact
category sums, allocation-free observation, convenience and Job warm
stability, explicit-window Pipeline plan identity, `peak_bytes` budget
rejection before Pipeline allocation, and a
representative authenticated accelerator token. `compute.pipeline` owns
cross-Program maximum-envelope sharing, visibility, output parity, and memory
reconciliation. `compute.flow` owns exact cold/warm CPU, Metal, and Vulkan
output for two partial `u64` scatter resets whose original frontier coordinates
cross fused Map regions. `accel.kernel-core` independently enumerates all 769
gap-free fusion partitions and reset-source positions for one through seven
original steps and checks the sealed projection against the interval model.
That semantic split follows the product ownership boundary; duplicating the
same oracle in a broader suite is not evidence. The Debug-only line-table
policy that bounds this template-heavy owner's symbol volume is owned by
[`build/graph.md`](../build/graph.md); it does not remove a memory row or change
Release code generation.

The registered `tests/contract/compute/memory.cpp` source is the ordered case
runner. Counter observation, Program ownership, primitive scratch, value-route
arena, CPU graph storage, and accelerator ownership live in one-word
translation units below `tests/contract/compute/memory/`. `local.hpp` and
`model.hpp` are the sole shared counter, snapshot, and accounting fixture.
Each semantic leaf has its own rebuild closure. Memory rows, allocation
oracles, backend order, failure codes, and case identity are invariant.
The focused `accel.kernel-core` reset leaf independently proves 32- and 64-bit
dense and strided projection, the 40-byte parameter layout, exact window
counts, unique replacement selection, offset/stride/end overflow rejection,
Vulkan word-address admission, and the common
`accel_kernel_reset_invalid` reason.
