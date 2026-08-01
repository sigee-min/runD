# Compute Graph Services

This reference owns the public, domain-neutral graph-service contract layered on
the canonical Compute Flow recipe and Kernel graph. It does not introduce a
second graph builder or execution engine.

## Ownership

The domain caller owns semantic state, snapshots, ticks, module policy, and
replay. Compute owns only typed values, ordered resource access, canonical
graph identity, compilation, and execution. Resource and graph metadata never
contain domain labels such as body, contact, temperature, texture, or ECS.

The authority path is:

```text
Flow recipe
  -> selected-output dependency closure
  -> canonical GraphState resource graph
  -> Kernel graph validation and 128-bit fingerprint
  -> cached or newly lowered Program
  -> CPU, Metal, or Vulkan execution
```

`GraphState` remains private. `graph::Info` is an immutable observation of the
same admitted graph and cannot mutate scheduling or execution.

The public observation vocabulary is nested under `compute::graph`:
`Fingerprint`, `Value`, `Visibility`, `Operation`, `Resource`, `Access`,
`Node`, `Barrier`, `MemoryPlan`, and `Info`. The namespace and
`graph/info.hpp` path carry
the graph context, so none of these values repeats a `Graph` prefix.

## Stable Filter And Bounded Active Sets

`filter(predicate)` preserves source order. Its logical construction is a
fused selected/rejected predicate-mask Map, deterministic count planning, and
stable partition:

```text
source index:  0 1 2 3 4
predicate:     1 0 1 0 1
logical output 0 2 4
```

The result is `stage::Bounded<Count>` with capacity equal to the source
capacity and an explicit resident U32 or U64 logical-count value. The payload
prefix and logical count are graph resources. Worker completion order,
backend queue order, and physical allocation order cannot affect output order.
Every bounded consumer reads the resident count directly; it cannot perform a
hidden host count read.

For Exact input, the selected and rejected masks are two outputs of one Map
node over the shared predicate DAG. They are not two input traversals. For `N`
source elements, mask scheduling therefore performs one `N`-element Map rather
than two; the following Reduce and Partition remain distinct because they own
the resident count and stable payload order. Bounded re-filtering additionally
combines the predicate with its prior resident active-count lineage.

`compact({.capacity = N})` accepts U32 flags and emits the source index of each
nonzero flag in strictly increasing source order. `N` is a hard upper bound on
the actual number of selected indices, not a request to truncate the result. If
the selected count exceeds `N`, the whole run fails with
`compute_compact_capacity_insufficient`; no partial prefix is published. This
order, capacity, and failure contract is identical on CPU, Metal, and Vulkan.

The public result is `stage::Bounded<std::uint32_t>`, never an Exact
capacity-sized sequence. Its capacity is `N`; its resident logical count is a
canonical U32 CountNonzero Reduce over the same flags. The count retains the
Compact capacity check as a guard dependency, so projecting only `.count()`
cannot prune or bypass overflow validation. The authenticated backend Compact
primitive remains the domain-neutral two-buffer operation `(flags, indices)`;
the bounded count is ordinary graph composition rather than a hidden host
readback or a second backend-specific Compact ABI.

Bounded map preserves only the logical prefix as semantic output, and bounded
scan, reduce, sort, window, filter, and later bounded composition consume the
resident count directly. A terminal reads the one-element count first and then
transfers exactly the logical indices prefix. Unused capacity bytes are neither
returned nor downloaded, and stale tail storage from an earlier resident run
cannot affect a later underfilled result.

The one-shot terminal owns one read-only binding set. It does not construct the
inactive input storage reserved for resident `write()`, and it prepares its
active inputs, physical outputs, and graph-internal buffers once. A writable
resident Job deliberately retains two independently prepared input binding sets
so a complete write can commit by swapping owners without allocation.

Zero is a valid compact capacity only for a zero-element source through the
usual default-capacity rule. That case compiles as a canonical backend-free
empty Program and returns an empty value sequence with resident U32 count zero.
Its graph has no executable nodes, but introspection still records the value
output as zero elements and the count output as one four-byte element. Renaming
the builder does not change its fingerprint and therefore reuses the same
Program-cache entry. A nonempty Program whose selected count is zero reads the
count and records the empty value hash without a zero-byte CPU copy, driver
download, or value transfer.

`group_by` uses the same compact operation to pack group-head indices and
reuses compact's resident count as the group count. It does not retain a second
CountNonzero or Sum authority for that value. Before its capacity-wide Gather,
it masks every inactive physical head-index slot to valid source index zero
from that resident count, so a many-groups to fewer-groups rewrite cannot
dereference stale tail storage.

## Canonical Fingerprint

`graph::Fingerprint` starts from the full 128-bit Kernel graph identity after
selected-output liveness pruning, then mixes the ordered Compute resource and
Program-interface schema. It includes canonical node order, operation or
primitive identity, element count, logical resource IDs, ordered access roles,
ordered inputs and outputs, scalar types and capacities, numeric domain, fixed
format, rounding, overflow, and approximation policy. It excludes diagnostic
Map names, pointer values, allocation order, backend object handles, queue
state, timestamps, and domain metadata.

External Flow inputs therefore expose only `input<T>(count)` and
`zip_input<T>(count)`: ordered type, count, numeric policy, and declaration
position are their complete identity. There is no discarded input label.
Map names remain the graph diagnostic-label surface and are unchanged; they
remain excluded from canonical fingerprint and Program-cache identity.

The bound one-shot surface admits every lvalue contiguous range through the
same exact-element `BorrowedRange` contract. Container class, allocator, span
extent, and whether the lvalue is const do not enter graph identity; only the
canonical element type, count, and numeric policy do. C arrays and custom
span-compatible contiguous ranges therefore produce the same graph as a
vector or span of the same type and count. The caller's storage is borrowed
without a host copy, so temporaries, initializer lists, proxy references, and
implicit element conversions are rejected rather than represented by a
second binding adapter.

For Fixed resources, scalar identity includes storage width, integer bits,
fraction bits, rounding, overflow, and approximation policy. Changing any one
of those fields changes `graph::Fingerprint`, and therefore the Program-cache
key and backend artifact identity.

Typed multi-input Flow construction preserves that identity per input.
In particular, `zip_input<T>` records `T`'s declared `(I,F)` even when another
Fixed input has the same storage width, and selecting that input in a Map keeps
its format through the explicit `quantize<T>()` storage boundary. Same-width
inputs with different binary points are never inferred from one another.
Independent `zip_input<T>` admission stores the supplied `T` format directly;
it never inherits or validates `(I,F)` against the preceding Flow output.
Same-domain primitive side inputs remain a separate contract: they retain the
current value's rounding and overflow policy and reject a different storage
format before primitive admission.
Every Flow element Input node reads this stored per-resource format. Downstream
Exact, Bounded, and Scalar map/filter/combine/zip construction cannot replace
an admitted stored policy with a width-derived default; `(I,F)`, rounding,
overflow, and approximation remain explicit resource identity.
Primitive inputs likewise retain independent approximation provenance. Both
factor-input and matrix-input Solve require the factor or matrix and
right-hand side to share storage `(I,F)`, rounding, and overflow policy, while
the exact right-hand-side input remains `Exact` and the derived factor or
direct Solve result remains `Deterministic`; construction never rewrites one
resource's provenance to match another resource. Complex admission follows the
same boundary: imaginary input must share `(I,F)`, inherits the active real value's
rounding and overflow, and retains its own approximation.

Fingerprint equality means canonical graph equality under the graph policy
already encoded by that fingerprint. A Program cache never crosses its opened
Device or process image, so equality does not publish a cross-Device,
cross-ABI, or persisted-artifact contract.

A zero-capacity Flow admits no backend nodes. Its Program therefore exposes an
empty node/barrier list while retaining the typed recipe resources, ordered
Program interface, and semantic recipe identity in the fingerprint. It uses
the same Program cache and execution contract, but running it produces typed
empty outputs without creating a fake dispatch or barrier.

Zero work does not use a second Flow description or fingerprint algorithm.
The selected-output closure is lowered through the same private `GraphState`
builder as nonzero work, including canonical Map expressions, Scan operation,
primitive descriptor options, buffer roles, and logical outputs. Kernel recipe
identity admits those semantic nodes without requiring a nonzero element count;
execution admission still rejects them. Only the public `graph::Info` execution
view omits their nodes, accesses, and barriers. Consequently rename, temporary
creation order, and dead branches remain identity-neutral, while `value + 1`
versus `value * 2`, inclusive versus exclusive Scan, or a primitive-option
change remains a distinct cache key even when both runs return empty storage.

## Generic Resource Graph

`graph::Info` exposes the resources, nodes, dependencies, and barriers that the
canonical graph actually executes.

The compiled owner is `node/src/compute/graph/compile/`. Its lowering pass
consumes each `graph::Info::Node::accesses` row as the single ordered binding
authority for Program bindings, CPU references, and accelerator references.
No backend reinterprets Map or primitive ports to reconstruct that order.
CPU runtime steps are formed in that same canonical node pass; there is no
second traversal of `GraphState` after Kernel lowering.

The construction owner is `node/src/compute/graph/build/`. Map, Scan,
primitive, and output admission are separate physical units but append to one
ordered `GraphState`; they do not own alternative node orders. A multi-output
Map validates each distinct immutable expression state once. The at-most-16
state identities use a fixed exact index with no allocation, while every
`ExprRef` still receives its own root type check in authored output order.

Primitive input validation captures the first matching count input in the same
ordered pass that validates its value, shape, type, and Fixed policy. Control
binding lowering consumes that captured ordinal and does not search the input
interface again. Output-source and identity membership use a fixed exact index
at the 16-output product bound. Its 32 slots keep load at or below one half and
linear probing is bounded by 16 admitted keys; neither collision placement nor
pointer identity enters graph state or fingerprinting.

Graph input IDs are strictly increasing because each `graph_input_count`
appends one new value before publishing its interface ID. Selecting an external
input as an output therefore uses allocation-free binary membership
`O(log I)`, not an `O(I)` scan for every selected output. Output projection uses
a fixed 16-entry array, so the common direct-output path allocates only the
retained ordered output interface and does not allocate disposable source and
projection vectors. Duplicate-output order and canonical identity projection
remain unchanged.

For `V` values, `I` inputs, `O` outputs, `B` bounded-input rows, `R` logical
resources, `C` retained chunks, `S` executable steps, and `A` ordered accesses,
graph compilation outside the primitive-specific planner is:

```text
interface projection  O(V + I + O + B)
storage placement     O(A + R log R + C)
chunk rank            O(C log C)
reset ownership       O(A + R log R)
ordered lowering      O(S + A)
```

Graph description creates the `V` resources once with internal visibility,
then marks the `O` outputs and `I` inputs in two direct ID passes. Inputs are
marked last to preserve the single input-precedence rule when one logical value
appears in both interface sets. Binding validity is checked once after node
materialization, preserving construction-error precedence while keeping
interface projection `O(V + I + O)` instead of a per-resource membership search
`O(V(I + O))`.

Each ordered access is authored once together with its public node, Kernel
buffer reference, and memory-write metadata. Finalization folds read-byte
accounting into the same `O(A)` access pass that extends the canonical
fingerprint. The resource analyzer still receives one `O(R + A)` structural
projection because its domain-neutral `resource::Resource` and
`resource::Access` schema is independent of Compute introspection; that
projection carries no second ordering, visibility, or identity authority.

Reset route discovery is one resource pass followed by canonical
`(reset_node, resource)` ordering. Physical-owner equality does not deduplicate
routes. Storage keeps
one extent pass before route construction because aliased resources can enlarge
a retained chunk; this is the required two-phase placement proof, not duplicate
semantic lowering. Program compilation then seals one U32 permutation ordered
by `(descending chunk count, local chunk ordinal)`. Pipeline planning and
workspace materialization consume that immutable permutation directly; they do
not allocate, rebuild, or sort an occurrence-local order. These bounds describe
compile-time work and do not claim runtime device throughput.

`graph::NodeCapacity`, `graph::ValueCapacity`, `graph::BuffersPerNode`, and
`graph::OutputCapacity` expose the checked envelope. One Program admits up to
16,384 canonical ordered graph nodes after liveness and identity removal but
before fusion. `Info::authored_nodes` reports that admitted pre-fusion graph;
it is not the raw number of fluent Flow calls. `Info::lowered_nodes` reports
the selected executable node count, which is also `Info::nodes.size()`. This
schedule envelope is separate from the unchanged 1,024-node limit of one Map
expression/ComputeIR.
The authored count participates in the canonical fingerprint: a Program cache
cannot return an executable-equivalent entry carrying a different caller's
authored admission evidence.
The compiler may therefore keep a large simulation action in one Program and
one submission while forming deterministic maximal fused Map regions that each
fit the per-kernel IR limit. It does not select a different strategy by backend
or workload size. Logical graph storage is proportional to authored nodes and
resources; the derived construction ceiling is 1,048,640 values from
`16384 * 64 + 64`, not a resident allocation made by every Program.

It also exposes `read_bytes`, the saturating sum of the declared byte range
for every canonical read access. This value is computed once after resource
access construction and then retained by the Program:

```text
read_bytes = sat_sum(access.element_bytes * access.element_count
                     | access.mode == Read)
```

It is an exact canonical graph-plan quantity. Backend fusion, caches, SIMD
loads, and GPU memory transactions can change physical device traffic, so the
field is not labeled or interpreted as hardware bytes. Overflow saturates at
`UINT64_MAX`; telemetry marks that observation saturated and compilation
continues unchanged.

### Memory Plan

`Info::memory` is the immutable compile-time `MemoryPlan`. Its fields are
derived from the same canonical graph:

```text
logical_bytes    = sat_sum(bytes(r) | r is internal)
live_bytes       = max_n sat_sum(bytes(r) | n in [first(r), last(r)])
physical_bytes   = sat_sum(retained physical arena extents)
allocation_count = retained physical Buffer owner count
reset_bytes      = sat_sum(bytes(r) for each logical reset range r)
reset_count      = logical reset range count
```

These Program plan quantities are not allocator or per-run counters. `Stats`
reports reset work for an observed execution, and `MemoryStats` remains the
authority for current, peak, cumulative, reused, and budget observations.

### Resident Windows

The caller selects the one resident-window shape explicitly through
`PipelineBuilder::windows<Max, Tile>(...)`. Both template arguments are
positive, `Tile <= Max`, and

```text
K = ceil(Max / Tile) <= PipelineIterationCapacity
```

The body Program receives the resident count and canonical ordinal after its
authored inputs. Its Flow uses `resident<Max, Tile>(count, ordinal)` to express
the tile-local bounded stream, base, and ordinal. The Program is compiled once,
and Pipeline preparation freezes exactly `K` ordered occurrences that reuse
that Program and one workspace. Neither `MemoryBudget` nor device capacity
selects or changes `Max`, `Tile`, `K`, the body graph, or its storage extents.

For one execution, the resident device count `C` is validated once and every
prepared window `k in [0,K)` receives:

```text
base(k)  = k * Tile
count(k) = min(Tile, max(0, C - base(k)))
```

`C > Max` publishes the structured `compute_bounded_count_invalid` failure on
the device, sets payload work to zero, and publishes no output. The host does
not read `C`, rebuild the graph, choose a window count, or drive a loop.
For `C <= Max`, the nonempty intervals
`[base(k), base(k) + count(k))` are disjoint and their ordered union is exactly
`[0,C)`. Every logical index `i < C` therefore has the unique coordinates
`k = floor(i / Tile)` and `local = i - base(k)`. Window lowering evaluates the
same canonical operation at global index `i`; it neither changes numeric
policy nor reassociates a reduction. `Tile` is part of the authored Program
and Pipeline identity, so fold edges, source order, grouping, and stored bits
are fixed before execution.

Resident windows are one Pipeline recurrence over the authored body Program.
Preparation records all `K` entries in one prepared command graph.
CPU consumes them under one run call with zero native submissions; a nonempty
Metal or Vulkan attempt performs exactly one native submission. Neither route
performs a count or payload readback or warm allocation. The same plan supplies
CPU, Metal, and Vulkan the same explicit window order, Program chunks,
barriers, reset ranges, and fingerprint input.

#### Nested Tile-Local Repeat

Use `tile_repeat<N>(seed_program, action_program, fold_program)` when each
active resident window must run a fixed recurrence before contributing to one
outer state:

```cpp fragment
auto body = rund::compute::tile_repeat<N>(
    seed_program,
    action_program,
    fold_program);

auto prepared =
    rund::compute::pipeline(device)
        .windows<Max, Tile>(
            body,
            rund::compute::window(count),
            rund::compute::read(outer_seed, seed_external),
            rund::compute::write_final(outer_result),
            rund::compute::write_window(tile_results))
        .then(consumer,
              rund::compute::read(tile_results),
              rund::compute::write(consumed_results))
        .prepare();
```

The declaration is not independently executable or preparable; the enclosing
Pipeline retains the three compiled Programs once. With flattened tuples
`S` for Seed-external input, `T` for complete tile state, `P` for inner carry,
`O` for outer state, and `W` for append-only tile output, admission requires:

```text
Seed  : (S..., U32 count, U32 ordinal) -> (T...)
Action: (T...)                         -> (P...), P is a prefix of T
Fold  : (O..., T...)                   -> (O..., W...)

windows reads         (O..., S...)
windows final writes  (O...)
windows window writes (W...)
```

Writing `T = P || Q`, one active window seeds `T`, evaluates Action exactly
`N` times by alternating two `P` banks while retaining one `Q` bank, then
folds the final `T` into `O`. The complete Fold precedes the next Seed.
Only Seed receives the canonical total runtime count and outer ordinal; it
derives the tile-local tail count with `resident<Max, Tile>`. Any Action
invariant derived from either coordinate is explicit in `Q`.

Each `W` leaf has `Tile` elements and its `write_window` target has `Max`.
After every successful Fold, runD publishes only the active dense slice into
that caller-owned target. The nested declaration has exclusive write ownership
until it is sealed; a later `.then(... read(tile_results) ...)` in the same
Pipeline then reads the complete target naturally. The planner inserts the
exact device write-to-read boundary and keeps the target as the binding—there
is no host round trip, second Pipeline, or `O(Max)` intermediate copy. Later
writes to that owned Buffer remain rejected.

The prepared schedule is hierarchical. It retains `O(K + N)` route templates
and fixed control, not a flattened `K * N` collection of Program graphs, Jobs,
workspaces, bindings, or banks. Seed, Action, and Fold share the maximum
serial workspace, dense-View, and primitive-scratch envelopes. Authored
occurrences, physically encoded Program occurrences, and observed dispatches
remain separate evidence. A status-free element-local Map Action with exact
ping-pong carried views and invariant tail is proved once as a tile transducer:
one device invocation evaluates all `N` transitions in order and retains the
carry in registers. Its physical Program-occurrence shape is `K * 3` for Seed,
Action, and Fold while logical inner work remains `K_active * N`. An indexed,
controlled, telemetry-bearing, collective, aliased, or otherwise unproved
Action keeps all `N` physical invocations and its exact failure coordinates.

`C = 0` performs only resident preflight, seals the explicit recurrent `O`
prefix, and publishes the initial `O`; `W` is neither paired with a Fold input
nor copied.
A partial last window receives its exact active count. Seed must retain that
count in invariant `Q` whenever Action or Fold needs bounded tail access; the
Pipeline preserves it but does not infer a new count lineage across compiled
Program boundaries. `C > Max` suppresses every nested payload command and
publication before returning `compute_bounded_count_invalid`. An optional
`until<Index>` observes a U32 leaf of current `O` between outer windows; a
Fold-produced match suppresses only subsequent windows, never a suffix of the
current fixed inner recurrence. Seed, Action, or Fold failure suppresses all
later work and publication, and mutable route status is consumed before route
reuse. Telemetry checks the same resident stopped state before reading a
mutable route counter, so an inactive occurrence cannot count data left by an
earlier active occurrence.

Plan, status, statistics, and profile evidence retain the logical Pipeline
step plus independent outer-window and inner-iteration coordinates; Seed and
Fold identify their phase without a synthetic inner iteration. Bounds and
locations are never represented by `k * N + j`. CPU follows that order in one
Pipeline run. Metal and Vulkan reuse at most two parity Action
Job/prepared-resource owners across the cold-frozen stream and perform one
native submission with no warm allocation, compilation, descriptor growth,
rebind, count readback, or fallback.

Metal normal command buffers are single-use. Cold preparation therefore owns
one fully guarded static ICB: Pipeline-private kernels reserve Buffer index 30,
unowned controls bind zero, recurrence-owned payloads bind the device-private
owner stop word, and formerly indirect primitives use checked maximum grids
with their resident logical guards. A warm attempt creates the required outer
command buffer, declares the frozen resource array with one bulk call, executes
the whole ICB through one range call, and commits once. It visits no frozen
command, range, binding, indirect-grid, or recurrence-state rows. This closes
the schedule-sized host loop without claiming zero CPU participation in Metal
residency, submission, completion, or fixed control observation. Vulkan's
primary command buffer remains recorded during cold preparation.

The corresponding `rebinding_count` means mutations of those retained
Job/Buffer/View/prepared-owner identities after preparation. Cold native
capture and emission of already frozen descriptors are not mutations. The
counter is zero by construction and is interpreted together with the
`compute.window` frozen-binding snapshot oracle; a zero counter alone is not
the ownership proof.

`PipelineBuilder::plan()` returns the exact public `PipelinePlan` summary used
by `prepare()`: caller-owned `persistent_bytes`, Pipeline-owned state and
transient logical Buffer bytes, typed backend View and primitive scratch
storage in `prepared_bytes`, publication traffic, `peak_bytes`, `total_bytes`,
allocation/reuse/publication counts, and largest/peak coordinates.

For nested tile-local recurrence, that summary additionally distinguishes
outer-window count, `Tile`, inner-iteration count, compact route-template
count, and authored occurrence capacity. Its largest, peak, and View
locations carry separate outer and inner coordinates. Runtime dispatch totals
remain `Stats` evidence and are not synthesized from either prepared count.
`barrier_count` likewise reports the exact nonzero boundaries in the compact
schedule produced by the canonical resource hazard analysis plus shared-arena
reuse. It is not `prepared_command_count - 1`; the authored count is
`K * (N + 2)` while a pure nested route table has at most `K + N + 2`
boundaries and a proved Action transducer needs only `K * 3` physical Program
occurrences. `plan()` freezes that boundary vector and `prepare()` consumes it
without reconstructing another hazard policy.

```text
peak_bytes = state_bytes + transient_bytes + prepared_bytes
```

The prepared term is not an opaque post-prepare estimate. Pipeline planning
derives the dense View requirements from the frozen graph binding routes,
places them in deterministic aligned subranges of bounded backing owners, and
makes every prepared Job borrow that arena. The alignment is an immutable
capability of the selected backend combined with the strongest natural scalar
alignment of every slot. Sequential uses of different scalar types may share
the same raw-word slot. `view_bytes` and its
step/iteration/binding coordinates name the largest requirement. Its addressed
span, backing extent, offset, stride, element width, count, and authored
alignment are available in the corresponding `view_*` fields.
The same plan walks the admitted Kernel operation sequence once, derives each
collective's exact simultaneous temporary requests, and first-fit packs each
operation into storage-sized aligned pages. Operation boundaries reset that
placement behind an explicit visibility barrier. `scratch_bytes` and
`scratch_count` expose the maximum serial operation and Program envelope.
Prepared Metal and Vulkan primitives borrow those offsets; neither owns a
hidden scratch allocation.
`MemoryBudget{bytes}` compares this complete
planned payload and returns `compute_pipeline_memory_budget` before
Pipeline-owned Buffer materialization. `memory_snapshot()` labels scratch
separately in Resident and Device categories. Backend allocation granularity,
driver metadata, and transient transfer pools are measured after prepare by
`Pipeline::memory()`; their allocation failures retain the owning typed reason.
Budget is an admission ceiling, not a window planner or a fabricated
device-memory bound.

An internal value is live on the closed node interval `[first_use, last_use]`.
Closed intervals make a value read by a node overlap every value written by
that same node. An internal lifetime must be born at a dense write-only node
frontier; a read of that value at its first-use node is rejected because graph
nodes define no intra-node access order. Inputs and outputs never enter an
arena.

There are two initialization proofs. `Full` means the first producer writes
the complete logical extent. `Domain` means it writes the active prefix. A
Domain consumer is covered only when its count is the same canonical count or
a descendant through `Resource::parent`, and any writer predicate is identical
on the consumer. Thus for writer count `c_w` and reader count `c_r`:

```text
c_r == c_w or parent*(c_r) == c_w  =>  c_r <= c_w
                                      => [0, c_r) subset [0, c_w)
```

`Resource::active` names the count governing that value and `parent` records
only stable Filter/Compact count descent. There is no mirrored count limit in
the memory planner; primitive and Map control capacities remain the one runtime
validation authority. The planner builds a contiguous resource-to-use index,
so lifetime and Domain validation visit `O(R + A)` resource/access entries
rather than rescanning all nodes for every resource.

`MemoryPlan` evaluates the canonical Program graph at its authored resource
extents. Every reusable internal range enters one deterministic
256-byte-aligned virtual placement in `(first_use, resource id)` order. A
segment is released only when `last_use < next.first_use`; best-fit minimizes
remaining bytes and then offset. A Pipeline window body is already a Program
authored at the explicit `Tile` extent, so its Program `MemoryPlan` contains no
window-size, window-count, budget, or backend allocation row.
The product contract checks `Max = 516096`, `Tile = 8192`, and `K = 63`
independently for 32-bit `Fixed<16,16>` and 64-bit `Fixed<20,44>`. For each
format, the `Max` plan has exactly the same transient bytes and allocation
count as its `Max = Tile` control; only caller-owned persistent input bytes and
the fixed occurrence reuse count grow with `Max`. The same contract rejects a
one-byte-short `MemoryBudget` before allocation and prepares the exact admitted
plan at its peak budget on CPU, Metal, and Vulkan.

The planner also evaluates a proved destructive-Map placement. The Map must
have one eligible dense full-write output. Every source candidate must be an
arena-backed, same-shape, pointwise read whose lifetime ends at that Map node.
When several inputs satisfy the proof, the smallest canonical resource ID is
the source. `Resource::source` records that minimum candidate. The destructive
placement is kept when its arena extent is no larger than the ordinary
placement; equal extents prefer the proved alias.
Free ranges have one address-ordered tree for adjacent coalescing and one
`(aligned usable bytes, aligned offset)` tree for the exact best-fit choice.
Each release, merge, split, and selection is `O(log R)`, so the complete
lifetime plus placement pass is `O(R + A + R log R)` time and `O(R + A)`
memory. It does not sort or scan the complete free set per resource.
The virtual arena is partitioned at
`min(1 GiB, backend storage limit)` physical boundaries. No admitted ordinary
range crosses a chunk. A logical range larger than that ordinary ceiling uses
an offset-zero owner and never shares that owner with a simultaneously live
range. Nonoverlapping large ranges use deterministic interval ownership:
expired owners are indexed by `(capacity, owner ID)`, the smallest fitting
capacity wins, otherwise the largest smaller capacity grows, and equal
capacities choose the smallest owner ID. The pass is `O(L log L)` for `L`
large ranges and retains `O(L)` metadata. One owner contributes the maximum
assigned size rather than their sum. Chunking and large-owner reuse use one
graph-canonical algorithm on CPU, Metal, and Vulkan. The selected target's
frozen storage limit can only lower the ordinary owner ceiling; active work,
timing, cache state, and runtime memory pressure never select a different
plan. `physical_bytes` sums used owner extents rather than virtual address
gaps, and `allocation_count` counts retained owners.

The current admitted placement candidates have identical authored arithmetic,
materialized value set, dispatch topology, and recomputation cost. Their cost
comparison therefore reduces exactly to the lexicographic pair
`(physical_bytes, allocation_count)`; a tie keeps the proved destructive Map
transition. No unmeasured coefficient or host-specific tuning enters graph
identity. Future fusion, streaming, or rematerialization candidates may enter
this owner only with a bit-level proof and explicit memory-traffic, dispatch,
and recomputation evidence; arena packing alone is not described as
materialization elimination.

A value without either initialization proof records its first writer in
`Resource::reset_node`. It enters the same deterministic 256-byte-aligned
lifetime arena as complete-write values. Two ranges may overlap physically only
when their closed lifetimes are disjoint; the later partial-write range resets
at its exact first writer, after the earlier range's last use. Every logical
range retains one exact reset route even when several ranges share an offset or
physical Buffer; owner equality never deduplicates routes. Each prepared
Pipeline occurrence invokes its Program through that same schedule. External
partial outputs reset their complete authored View. Inputs and explicit
recurrence or transactional carry are not reset.
Lowering records this once as `BufferInit::Zero` on the exact first Write.
CPU graph execution consumes the same init fact; accelerator compilation seals
the surviving binding and its fusion-projected lifetime into the kernel token.
No Job or Pipeline invocation carries a second reset-coordinate list.

CPU, Metal, and Vulkan consume the same chunk group, byte offset, byte extent,
element width, and alignment. Physical arenas use raw U32 storage; typed routes
retain their original scalar and complete Fixed policy. Storage identity and
offset never become arithmetic inputs. Alias transitions emit explicit graph
reuse barriers, so placement changes neither operation order nor visible bits.
The plan fields that affect placement and initialization participate in graph
fingerprinting; pointer values, allocator order, and device addresses do not.

A resource records:

- canonical logical ID;
- scalar value type, element count, element width, and total byte extent;
- typed `Rounding`, `Overflow`, and `Approximation` policy values for Fixed
  resources, without exposing their byte encoding as a second API;
- input, output, or internal visibility;
- active-count identity and count-parent lineage;
- proved destructive source identity;
- first and last node use;
- physical chunk group and byte offset;
- whether the physical owner requires zero initialization.

The domain-neutral `resource::analyze` service accepts logical resources,
alias-group byte offsets, and exact checked scalar footprints. One access is
the union

```text
U(i in [0, count)) [offset + i * stride,
                     offset + i * stride + element_bytes)
```

where `element_bytes` is in `[1, 8]` and `stride >= element_bytes`. Access
offsets are relative to a logical resource; the resource alias offset places
that footprint in its alias group's byte address space. It rejects zero,
overflowing, out-of-range, noncanonical, unknown-resource, overlapping-lane,
and invalid access-mode values. `resource::AccessMode` is the one read/write
vocabulary used by both planner input and `graph::Info`; there is no
graph-specific mirror or mode conversion.

The original contiguous spelling remains valid as
`{offset_bytes, size_bytes}`. It is exactly the byte-lane special case
`element_bytes=1`, `element_count=size_bytes`, and `stride_bytes=1`; callers
choose either that spelling or the exact strided fields, never both.
`graph::Access::size_bytes` remains the logical payload byte count for public
introspection while the three strided fields retain its exact footprint.

Two scalar lanes overlap exactly when their start difference `d` satisfies
`1 - left.element_bytes <= d <= right.element_bytes - 1`. The planner tests
that bounded interval and solves the strided start equality as a bounded
linear Diophantine equation. A footprint with at most 64 lanes is enumerated;
larger pairs use extended Euclid, so the decision does not walk capacity-sized
envelopes. This matters for interleaved structure-of-arrays views: even and odd
lanes may share the same byte envelope without acquiring a false dependency.
An exact intersection in one alias group generates a dependency whenever at
least one access writes. Each ordered node pair retains one barrier at the same
index as that dependency. Its canonical conflict prefers a cross-resource
witness because that row also proves alias-group visibility, and retains both
complete footprints plus a deterministic byte-intersection witness; further range
conflicts in the same ordered pair cannot strengthen the total command order
and are not stored. When both
footprints are contiguous (`stride_bytes == element_bytes`), that witness is
the complete contiguous overlap interval; this preserves identical planning
and introspection for the original range spelling and its exact strided
spelling. A genuinely strided pair records its canonical first intersecting
lane interval. Read/read
intersections remain unordered. Lifetimes are the first and last node that
access each logical resource, and the planner is their only writer.
`resource::intersects` exposes this same checked predicate to same-step
admission; Pipeline and graph planning therefore cannot diverge or fall back
to conservative envelope overlap.

For canonical zero work, a nonempty resource list with zero nodes is valid only
when the access stream is empty. Lifetime vector position is the canonical
resource order, so `lifetimes[i]` describes `resources[i]` without storing a
second resource id. The resulting lifetimes are all `resource::NoNode` and the
dependency/barrier lists are empty; a zero node count paired with any
access remains `compute_resource_graph_incomplete`.

Whole-value Flow accesses are the canonical special case
`offset=0`, `element_bytes=sizeof(T)`, `count=elements`, and
`stride=sizeof(T)`. `graph::Info` is produced by the same planner, and its
fingerprint includes alias identity, alias offsets, and all four footprint
fields.

A node records its canonical index, generic operation kind, element count,
ordered resource accesses, and ordered predecessor dependencies. A read after
a write and a write after a read or write depend on every earlier overlapping
conflicting access. `graph::Info` retains at most one executable barrier before
each later node. If any conflict at that boundary crosses logical resources,
the representative is the latest such conflict; otherwise it is the latest
same-resource conflict. This selection preserves the cross-resource arena
reuse bit while the binding-role schedule proves same-resource visibility.
`graph::Barrier` exposes that representative's logical resource ids, shared
alias group, exact physical overlap range, and access modes. It does not store
a third `resource` field that merely repeats `after_resource`.

For hazard set `H` and later node `j`, the executable projection is

```text
E[j] = OR(h in H, h.after_node == j)
```

The retained barrier array contains exactly one deterministic witness for every
true `E[j]`. Therefore `barrier_count <= node_count - 1`, and CPU, Metal, and
Vulkan consume the same boundary vector.

The planner keeps earlier accesses in one deterministic AVL interval index per
alias group and emits dependencies and their canonical witnesses in canonical
newest-first discovery order. Each subtree stores its physical envelope and
latest canonical ordinal. A best-first frontier prunes disjoint subtrees,
visits possible overlaps in descending ordinal order, and stops immediately
when one conflict covers the later access's complete envelope. It never first
collects and sorts the whole candidate set. Hash tables are membership indexes
only and are never iterated, so bucket layout cannot affect output order.

For `A_g` accesses in alias group `g`, index construction is
`O(sum_g A_g log A_g)`. Let `V_i` be the index rows expanded before the
visibility frontier for access `i`, and `K_i <= V_i` the exact envelope
candidates. Best-first reporting costs
`O(sum_i V_i log V_i)`. Repeated complete-range accesses expand only the
balanced path to the newest frontier instead of all older overlaps; true
multi-range dependencies remain output-sensitive. Each large-footprint
comparison performs at most 15 bounded
extended-Euclid solves because public scalar widths are at most eight bytes; it
does not scale with element count. Dependency membership is expected `O(1)`.
The retained dependency/witness output is bounded by the distinct ordered node
pairs, while `graph::Info` reduces those witnesses to at most one executable
boundary per later node.

## Program Cache

A `ProgramCache` is bound to exactly one opened `Device`. For a graph `G` in
that cache `C`, the complete lookup key is:

```text
key_C(G) = graph::Fingerprint(G)
```

The fingerprint already includes graph and numeric policy. Device identity and
the compiler/runtime ABI scope `C` itself rather than becoming repeated key
fields: entries never cross the cache-bound Device, the owning process image,
or a persisted cache. Diagnostic names, Flow object addresses, host buffer
addresses, temporary recipe IDs, and invocation order are not key inputs.

`on(device, cache)` builds a deferred Flow with the Device as its sole target
authority and validates that the cache belongs to that Device. Compilation
performs canonical graph description before backend lowering. A ready match
returns the existing Program state without compiling, allocating graph
resources, or creating backend pipelines again. Concurrent misses for one key
coalesce into one compilation; misses for different keys may compile in
parallel. Failed compilation is delivered to all current waiters and is not
retained as a ready cache entry.

Each entry has exactly one outcome: pending compilation, a ready Program, or a
failure Status being published to waiters. These alternatives do not coexist
as independent flags and optional payloads. A successful publication must own
a non-null Program; a null success is normalized to a compile failure before
publication, and a failed entry is removed after its waiters observe it.

`ProgramCache::Stats::misses` is the sole count of cache-admitted compilations.
Ready reuse increments `hits`, and callers coalesced behind an in-flight miss
increment `waits`; there is no second compile-count mirror.

The cache has an explicit positive ready-entry capacity and LRU retention.
In-flight entries may temporarily exceed that capacity and are not evicted.
`clear()` removes ready entries but cannot cancel an in-flight compilation.

The ordered fingerprint map is the only membership index. With ready capacity
`C` and `I` in-flight compilations, lookup performs `O(log(C + I))` semantic
fingerprint comparisons. A ready hit takes one mutex interval and allocates
nothing. One miss creates exactly one pending entry and map node while holding
that interval; same-key followers find that pending entry and allocate nothing
before waiting. Moving candidate allocation ahead of admission would let `W`
racing callers create `O(W)` discarded nodes, so no speculative allocation
path exists. No pointer, allocation order, or call ordinal enters the key.

Ready entries form one intrusive oldest-to-newest list. A ready hit moves its
entry to the newest end in `O(1)`. Immediately before one builder publishes,
the ready count `R` is at most capacity `C`; changing that one entry from
in-flight to ready gives `R <= C + 1`. Completion therefore appends once and,
only when `R = C + 1`, unlinks the unique oldest ready entry and erases its
stored map position in amortized `O(1)`. It performs no entry scan, timestamp
comparison, or allocation and cannot select an in-flight entry. The list is
the sole eviction-order authority; map iteration order is irrelevant.
`stats()` reads the ready count and derives in-flight count as
`map_size - ready_count`, so observation is `O(1)` and creates no second entry
state.

Unlinking an eviction victim or the ready entries removed by `clear()` occurs
under the cache mutex, but destruction of each detached Program owner occurs
after releasing it. Backend pipeline or resource destruction therefore cannot
extend the cache critical section or block an unrelated hit, miss, in-flight
publication, or waiter notification. `clear()` transfers existing map nodes;
it allocates no second retirement container payload and leaves in-flight nodes
in place.

## Asynchronous Compilation

Include `<rund/compute/async.hpp>` only in translation units that call this
terminal. The header completes the existing deferred Flow member and owns the
standard-future machinery; it does not define another Flow or compile route.

`std::move(flow).compile_async()` transfers one deferred Flow recipe to the
compile service bound to that Flow's Device and returns a standard future of
the same `Result` type as synchronous `compile()`. Synchronous and asynchronous
compilation are therefore two terminal verbs on one Flow surface; there is no
free async function or second graph builder.

`Compile {workers, capacity}` is the explicit positive resource envelope.
Its object-local defaults are two reusable workers and 64 queued recipes, but
no process-global service or global resource constant exists. A standalone
Device owns the service selected by `open(target, compile)`. A Session owns
one lazily started service selected by `SessionConfig::compile`, and every
`open(session, target)` Device binds weakly to that same service.
`Device::compile()` observes the exact bound envelope. A Device opened without
a Compile envelope remains synchronous-only and rejects async admission
instead of borrowing a hidden service.

The service does not create one thread per request and does not invoke
`std::async`. Admission uses a preallocated slot ring sized exactly to
`capacity`, with one `head` and one occupied count. Each request first reserves
its tail slot without constructing a packaged task or future. A full ring
therefore returns `Reason::AsyncCompileCapacity` in `O(1)` time with zero heap
allocations and without invoking the recipe factory. A successful reservation
then constructs the packaged task and future outside the queue lock and
commits the ready task into that exact slot. It never blocks for capacity,
grows, spills to another queue, or runs the recipe synchronously. The recipe is
consumed exactly once. When the Flow was created with `on(device, cache)`, asynchronous
callers participate in the same cache-miss coalescing contract.

Reserved, ready, canceled, and empty are the complete slot state machine.
Commit order cannot change execution order: workers consume only the head, so
the reservation order is the FIFO authority even when later factories finish
first. Allocation or future construction failure cancels its reservation.
Canceled head slots are reclaimed synchronously; a canceled slot behind an
earlier request remains ordered and is reclaimed when it reaches the head.
Service stop cancels every uncommitted reservation before draining committed
tasks. Shutdown waits only for committed work; canceled factories have no
publication route.

For queue capacity `C` and worker count `W`, let `R` be uncommitted
reservations and `Q` be committed queued tasks. The ring invariant is
`0 <= R + Q <= C`; at most `W` additional tasks execute, so the service owns at
most `C + W` admitted recipes. Ring storage is allocated once with the service;
reserve, commit, cancel, push, and pop allocate no queue node. Reservation
takes one mutex-protected critical section but never waits for capacity.
Because `head < C` and `occupied <= C`, the tail sum is below `2C`; one
comparison and subtraction wraps it without integer division. Packaged-task
and future construction retain their ordinary checked allocation contract,
but occur only after a slot exists. Failure cancels that slot and returns no
future.

An already-invalid Flow returns its preserved first `Reason` before service
lookup, reservation, packaged-task construction, or future construction. This
keeps a moved-device `DeviceInvalid` failure authoritative instead of
translating it to `AsyncCompileUnavailable`, and prevents invalid recipes from
consuming compile capacity.

The installed async header owns only the typed packaged-task/future adapter and
a narrow type-erased enqueue declaration. Worker lifetime, the queue, locking,
notification, shutdown, and partial-start cleanup have one compiled owner in
the Compute implementation. If `H` SDK translation units include Compute,
those service internals are parsed once by the product source. Runtime queue
order and future semantics are independent of SDK translation-unit count. The
source-private service is the direct contract-test seam;
there is no public service object with independent state.

Independent Device or Session owners have independent rings, so one owner's
burst cannot consume another owner's admission capacity. The returned future
observes the service-owned packaged completion. Ring pop order is FIFO;
multiple workers may complete independent recipes in either order. Destroying
the future does not join a per-call worker because no such worker exists. Program
publication occurs only after complete graph validation and backend
preparation; no partially compiled Program is visible. Service shutdown drains
the admitted queue before joining its bounded workers. Lazy service startup is
transactional: if a later worker cannot start, every worker already started by
that attempt is stopped and joined before the failure is returned.

Session drain first stops service admission and lets already committed recipes
finish on the existing workers. Session close performs that same admission
boundary, drains admitted compilation, joins the configured workers, and
releases the sole Session owner. A Device that survives either boundary cannot
admit another recipe; its next `compile_async()` fails with
`Reason::AsyncCompileUnavailable`. This is the same typed failure used when a
Flow has no bound compile service.

## State-Independent Execution Artifact Cache

The Program cache is the sole public execution-artifact cache. It retains
validated graph plans, lowered kernels, pipelines, and immutable preparation
metadata. It never keys on or retains user input contents, output contents,
snapshots, ticks, random state, or domain state. Running the same cached
Program with different input buffers must execute again and produce the result
for those inputs.

Result memoization is not part of Compute. Compiled-artifact reuse and replay
remain separate ownership boundaries: a cached Program always executes again
for the supplied bindings.

## Rejection Vocabulary

| Gate | Stable reason |
| --- | --- |
| Null or moved-from cache | `compute_program_cache_invalid` |
| Zero ready-entry capacity | `compute_program_cache_capacity` |
| Flow and cache Device mismatch | `compute_program_cache_device_mismatch` |
| Canonical graph description failure | Existing owning graph or expression reason |
| `PipelinePlan::peak_bytes` exceeds `MemoryBudget` | `compute_pipeline_memory_budget` |
| Resident device count exceeds authored `Max` | `compute_bounded_count_invalid` |
| Bound async ring is full | `compute_async_compile_capacity` |
| Flow has no live bound compile service | `compute_async_compile_unavailable` |
| Async admission or completion allocation fails | `compute_async_compile_unavailable` |
| Unexpected internal compilation exception | `compute_program_compile_exception` |

## Verification

Contract evidence must prove:

- filter order and bounded logical-count parity across the supported backend
  matrix;
- fingerprint equality across diagnostic names and inequality across semantic
  graph changes;
- zero-capacity fingerprint equality after selected-output liveness pruning,
  including dead expression nodes and dead Flow branches;
- fixed construction-index behavior through a full 16-key collision chain,
  duplicate external-output materialization, ordered identity projection, and
  rejection of an identity value outside the admitted output set;
- resource lifetime, hazard dependency, barrier, and no-alias metadata;
- deterministic `MemoryPlan` logical/live/physical byte totals, 256-byte
  offsets, Full/Domain/zero initialization classes, count-parent cycle
  rejection, destructive-Map proof, and CPU/Metal/Vulkan typed-subview parity;
- explicit `windows<Max, Tile>` freezes
  `K = ceil(Max / Tile)` occurrences; boundary counts `0`, `Tile - 1`, `Tile`,
  `Max`, and `Max + 1` prove disjoint coverage, structured overflow, bit
  identity, one native GPU submission, no count readback, and zero warm
  allocation;
- `PipelineBuilder::plan()` and prepared `Pipeline::plan()` report the same
  `PipelinePlan`; a budget below `peak_bytes` fails before Pipeline-owned
  Buffer materialization;
- flat-only, pure nested, and nested-plus-ordinary plans report the exact
  compact hazard/workspace boundary population independently of expanded
  prepared command-reference count;
- cache hit, miss, coalesced concurrent miss, capacity eviction, clear, and
  device isolation;
- exact oldest-to-newest touch and eviction order, allocation-free ready hits,
  and `clear()` removal of ready entries without removing a pending compile;
- a burst of distinct in-flight successes restores the exact ready capacity,
  and evicted or cleared Program owners are destroyed outside the cache mutex;
- failed compilation is not retained: a same-key retry performs a new compile
  attempt and leaves no ready entry;
- asynchronous and synchronous Program/result parity;
- exact allocation-free async-ring full rejection without factory invocation;
- FIFO reservation order even when commit order differs, exception-safe
  reservation cancellation, immediate capacity recovery, and shutdown drain;
- partial async-service startup cleans up every worker already started before
  reporting the admission failure;
- different inputs execute again and are never answered by result
  memoization;
- the installed SDK exposes Flow as its single graph-building language and
  Program caching as its single compiled-artifact reuse service.

`compute.graph-build` owns the bounded construction index and duplicate
external-output projection contract without requiring an accelerator.
`compute.program-cache-concurrency` is the CPU service owner for an explicit
worker and queue bound, shutdown queue drain, transactional compile-service
startup, same-key asynchronous coalescing, in-flight
waiter publication, distinct-key in-flight capacity restoration, detached
eviction/clear destruction, and failed-build removal.
Its registered `program/cache/concurrency.cpp` source is only the ordered
failure-code runner. `program/cache/concurrency/model.hpp` is the sole cache
state, opaque Program owner, and destruction-gate fixture; the `service`,
`async`, `capacity`, `lifetime`, and `failure` leaves own the corresponding
semantic groups. Each leaf has an independent rebuild closure; case
registration, checks, result codes, and execution order are fixed.
`compute.program-cache-index` owns the semantic-fingerprint map, intrusive LRU
order, ready-count invariant, zero-allocation hit, and pending-preserving clear
contracts. The accelerator
`compute.graph-services` case retains graph identity, resource planning, and
CPU/Metal/Vulkan Program parity without duplicating those shared
concurrency checks.

The registered `graph/services.cpp` source is only the ordered case runner.
Its one-word translation units below `graph/services/` own resource, birth,
memory, identity, policy, cache, bounded, empty, fixed, asynchronous, and
backend semantics independently. `local.hpp`, `model.hpp`, and `fixed.hpp`
are their only shared fixture, Flow-construction, and fixed-result oracles;
`oracle.cpp` owns the compiled resource-graph validation and equality oracle.
Each semantic leaf is an independent rebuild owner while execution order,
graph identity, failure codes, backend coverage, and semantic ownership remain
fixed.
