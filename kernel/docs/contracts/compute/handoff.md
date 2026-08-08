# Compute Backend Handoff Contract

## Backend Handoff

`ComputeDispatchWindow`, `BindingSet`, and `ComputeBackendDispatch` are the SDK-free
prepared Compute handoff. The backend receives caller-owned context, `ComputePlan`, a
validated lowering artifact, sequence windows, and a binding set. Public
kernel headers must not include Metal, Vulkan, Objective-C, platform windowing,
or SDK loader headers.
Kernel-owned `ComputePlanShapeValid(...)`, `ComputePlanBytesValid(...)`, and
`ComputePlanScalarValid(...)` are the shared support predicates for
plan identity and arithmetic self-consistency. Prepared kernel code uses the
header/bytes predicate for semantic plan admission. Node backend validation may
add expected-API, frozen-caps, scalar-width ABI, binding, and artifact checks,
but it must not carry a second copy of the generic plan staging arithmetic.
The same [`kernel/core/checked.hpp`](../checked.md) `checked::add`,
`checked::sub`, `checked::mul`, and `checked::ceil` owner is consumed directly
by Node sequence, resident-range, primitive-shape, and native backend
admission. Result overloads publish only after success, and predicate negation
is the overflow query. Node may own native-width projection such as `u64` to
`size_t`, but it does not rename or mirror the 64-bit equations.

`ValidateRuntimeBindings` checks the binding obligations derived from
`ComputeMap` and `ComputePlan`: tile count, input buffer count, input data/stride and
addressability, parameter bytes and data presence, staged output
addressability, and staged value read safety. Node fake-adapter validation
uses these plan obligations and artifact-key fields, not binding-set
self-declared totals.
Binding values are owned by `binding/model.hpp`; validation is owned by
`binding/validation.hpp`. There is no mixed aggregate header that makes every
value-only consumer parse runtime-validation implementation.

Runtime binding rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Binding set is already non-ok | binding set `reason` |
| Tile count mismatch | `compute_binding_tile_count_mismatch` |
| Input count/bytes mismatch | `compute_binding_input_count_mismatch` or `compute_binding_input_bytes_mismatch` |
| Input pointer/stride/addressability invalid | `compute_binding_input_null`, `compute_binding_input_stride_invalid`, or `compute_binding_input_bytes_overflow` |
| Output byte obligation mismatch | `compute_binding_output_bytes_mismatch` |
| Missing resident output storage for a required resident-mode output | `compute_binding_output_missing` |
| Staged and resident output both claim the logical output | `compute_binding_output_mode_conflict` |
| Parameter byte/data mismatch | `compute_binding_param_size_mismatch` or `compute_binding_param_null` |
| Staged output pointer/stride/addressability invalid | `compute_binding_output_null` or `compute_binding_output_stride_invalid` |
| Resident descriptor identity/extent/usage/stride invalid | `compute_resident_id_invalid`, `compute_resident_bytes_invalid`, `compute_resident_usage_invalid`, or `compute_resident_stride_invalid` |
| Prepared binding fields disagree with plan/map | `compute_binding_mismatch` |

Resident binding descriptors, when present, do not relax staged byte
obligations. They are an alternate storage mode for the same logical binding
identity and must validate tile count, element bytes, stride bytes, count,
total bytes, and usage before dispatch. Kernel resident descriptors are opaque
POD facts and never carry platform handles. Kernel-owned usage constants distinguish
logical read inputs from logical write outputs. Resident inputs are optional
alternatives for logical inputs and must match the planned input count. Their
individual element widths must match the op-derived per-input ABI, and their
sum must match `input_bytes_per_tile`; validation must not trust resident refs'
self-declared widths as the ABI authority. The resident output is an optional
alternative for the single logical output. `BindingSet` carries each side as a
`ResidentBindingRange`: one caller-owned descriptor array, one caller-owned
handle array, and an optional `u64` index array that selects deterministic
per-step order. A null index array means contiguous identity order. Every
selected index must be less than `storage_count` before either array is
dereferenced; an empty range is canonical only when its arrays are null and
its storage count is zero. Node backends authenticate the selected handle
sidecars and must reject missing, forged, or alias-control-block tokens. A
binding set must not claim
that logical output through both staged and resident storage in one run, and a
required output must be claimed by one of those modes. A resident output
becomes CPU-visible only after an explicit download or explicit staged readback
completes.

Node graph execution owns resident descriptors and handles once in its run
binding store. A prepared graph pins that store for at least the lifetime of
every non-owning step view. Map steps store `u64` indices because their binding
arity is dynamic; fixed-arity primitives store direct descriptor/handle
pointer pairs so their CPU path does not pay a second indexed lookup. Thus a
Map edge costs `sizeof(u64)` and a fixed primitive resource costs two pointers,
rather than another `sizeof(ResidentBufferRef) + sizeof(shared_ptr<void>)`
owner. Neither path increments or decrements a `shared_ptr` reference count.
Contiguous direct DSL calls need no index storage at all. Backends consume the
checked views and do not reinterpret graph binding indices independently.

Direct backend validation cannot treat `BindingSet::input_element_bytes` as
authority. Backend runtimes derive the
expected logical input width from the frozen `ComputePlan` scalar and require every
staged span and resident input descriptor to match that width before any
dispatch-window execution. Outputs use that scalar width except for a single
canonical mask output, whose ABI may bridge 32-bit and 64-bit lanes in either
direction. `ComputePlanScalarValid(...)` admits only that single-output width
shape; multiple outputs remain scalar-width homogeneous. This byte-shape proof
does not authorize a general conversion: map expression validation and the
lowered artifact must independently prove that the width-changing root is the
canonical mask operation.
## Run Ordering

Public Compute CPU execution uses `ComputeTileExecutor`, which reuses
`CompileKernelProgram`, the physical tile planner, `RunPreparedProgram`, and
the installed `WorkerBackend` queue. Actual worker participation remains
under Kernel authority.

### Immutable plan and borrowed run storage

A successful `ComputeTileExecutor::prepare()` publishes one immutable
`ComputeTileRunPlan`. The plan owns the compiled Workspace arrays and the
frozen `KernelProgram`, schedule, fold graph, backend, physical-tile policy,
and scalar tile facts. Re-preparing the source executor publishes a new plan;
existing plan handles and bound runs keep the original plan alive and cannot
observe the replacement. A plan-only executor is not executable and rejects
`run` or `submit` with `compute_tile_run_storage_missing`.

Mutable execution state has one independent owner. `ComputeTileRunStorage` is
the fixed-address typed control object. `ComputeTileRunStorageView` supplies
six caller-owned typed spans:

- `T` failure-reason pointer slots
- `W` worker tile counters
- `W` worker partition counters
- `W` worker start-offset values
- `W` worker elapsed values
- `W` worker tail-wait values

Binding copies only POD program and prepared descriptors into the control
object. Their schedule and fold pointers continue to reference the immutable
plan owner. The mutable execution Workspace therefore does not clone the
plan's seventeen vector capacities. Worker statistics are passed to
`RunPreparedProgram` as the explicit typed sinks already present in its
request. The storage control pins the immutable plan until a later successful
idle rebind or its own destruction.

`ComputeTileRunPlan::storage_plan()` is the checked reservation authority. For
tile count `T` and worker count `W`, its disjoint logical extents are:

| Component | Exact extent |
| --- | --- |
| `state_bytes` | `sizeof(ComputeTileRunStorage)` |
| `workspace_bytes` | `W * (sizeof(u32) + 3 * sizeof(u64))` for the explicit worker-stat sinks |
| `failure_slot_bytes` | `T * sizeof(const char*)` |
| `worker_tile_bytes` | `W * sizeof(u32)` |
| `async_context_bytes` | zero; synchronous and asynchronous control is typed inline state |

`total_bytes` is the checked unsigned 64-bit sum. A failed multiplication or
addition closes as `compute_tile_run_memory_overflow` and is not admissible.
`MergeComputeTileRunStoragePlans` forms a serial max-envelope by taking
`max(T)` and `max(W)` and replanning from those components; it never sums
serial plans.

`ComputeTileRunPlan::bind(view)` validates every span component before
publishing a generation and performs no allocation, locking, backend work, or
Workspace reservation. A larger max-envelope view may be rebound to different
plans serially. The returned executor borrows the control and spans; they must
outlive synchronous return or asynchronous `finish()`. `make_run()` is the
standalone convenience owner over this same bind path. It owns exact typed
arrays and does not create a second Workspace representation or vector-backed
run owner. For nonzero `T` and `W`, standalone materialization performs seven
`operator new` calls: one wrapper plus the six typed arrays. External arena
binding performs zero allocations.

`bind(view, active_count)` is the bounded form for a frozen maximum plan.
`active_count` must be no greater than the prepared count; the default
`bind(view)` selects the full count. Binding does not recompile or repartition.
It retains the maximum plan's static worker and tile identity, publishes the
active tile prefix, clamps the final callback range to `active_count`, and
skips every frozen tile whose begin lies outside that prefix. Physical worker
loops use the active tile count, and result `count()`, `tile_count()`, completed
tile count, worker tile count, and tail extent describe the active prefix.
This bounded bind and its later sync or async execution allocate nothing.

`ComputeTileRunPlan::storage_plan()` is the sole planned-storage authority;
callers read its `retained` member for the checked run-memory view. An executor
reaches the same immutable value through `run_plan().storage_plan()` and keeps
no compatibility projection.
`MeasureComputeTileRunStorage(view)` measures an actual view, so an envelope
larger than one plan is reported at its full supplied capacities. Allocator
metadata and the optional standalone wrapper's pointer bookkeeping are outside
this logical typed-storage extent. A default, moved-from, failed, or
generation-stale executor reports no attached run-storage extent.

The control owns an atomic phase and a monotonically increasing generation.
Bind, synchronous execution, asynchronous submission, ready-state projection,
and finish are mutually exclusive phase transitions. Binding while sync or
async work is active, or after readiness but before `finish`, fails with
`compute_tile_run_busy`. A successful later bind invalidates every older
executor view; those views fail with `compute_tile_run_rebound`. Thus no plan
can replace pointers observed by in-flight workers.

### Tile run state and completion

Sync `run` and async `submit` use the same context-reset authority, and sync
return and async `finish` use the same result projector. A run resets only the
`W` worker counters and the two atomic run facts; it does not clear `T`
per-tile failure slots. A failing tile overwrites its own unique slot before
publishing its index. Completion loads the published lowest index and reads
that one slot, so it does not rescan all `T` slots.

The first-failure invariant is

```text
F = min { t | callback(t) failed }
```

over callbacks admitted by the static tile map. Publication is an atomic
minimum. Once a failure `f` is visible, only tiles `t > f` may be skipped, so
no tile that could lower the minimum is excluded. Each tile has one static
writer for its reason slot, and the release publication of `F` followed by the
completion acquire makes the selected reason visible. This preserves stable
lowest-tile identity without a mutex or a per-callback reader lock.

For a successful run, callback traversal performs exactly `T` tile
invocations. Auxiliary hot bookkeeping is worker reset and projection,
`Theta(W)`, plus `O(1)` first-failure projection. The executor adds no
allocation, buffer copy, schedule pass, workload-size strategy, or fallback.
Its hot synchronization is the existing atomic admission/completion evidence;
there is no lock acquisition per tile. Worker tile counters remain
single-writer under the frozen static worker map.

### Retained lowering-owner memory

`ExecutionMetadata`, `LoweringArtifact`, and the internal parsed-input owner
expose
`retained_dynamic_memory_bytes()` as the byte authority for allocations below
their inline object. These are the lowering owners retained by authenticated
accelerator graph steps. Each accessor is `noexcept`, allocation-free,
lock-free, read-only, and backend-work-free. It counts actual vector capacity,
every live nested string's external storage, and every live nested byte-vector
capacity;
it does not count borrowed runtime buffer pointers or the inline owner itself.
The enclosing execution-step owner counts that inline object once through its
outer-vector extent, then adds this dynamic term. Inline storage is counted
once, and externally allocated storage is counted once.

All capacity products and additions in these accessors saturate at
`2^64 - 1`. `LoweringArtifact` delegates its metadata subtree to
`ExecutionMetadata` and then adds source-text external storage and canonical-IR
bytes that are actually live; it does not carry a second emitted-payload or
metadata formula. The parsed-input owner counts the parsed name, binding-vector
capacity, every binding name and value-byte-vector capacity, and node-vector
capacity exactly once.

Product graph mint is a destructive retained handoff. A GPU step keeps source,
key, and metadata, but drops parsed input and never materializes an owning
canonical vector. A CPU step keeps key, metadata, and the one typed parsed
input, but drops source and canonical storage. Thus neither product token keeps
both representations. Direct caller-supplied `LoweringArtifact` execution is a
different trust boundary: it retains canonical bytes and source long enough to
perform full authentication and does not enter the private retained path.

Node's compact CPU SIMD `PreparedRun` follows the same enclosing-owner rule. If
`I` is its instruction vector and `F` its fixed-format vector, its complete
dynamic extent is
`capacity(I) * sizeof(PreparedInstruction) + capacity(F) *
sizeof(ComputeFixedFormat)`, with saturating products and addition. Parsed IR,
binding-plan vectors, canonical IR, execution metadata, and textual CPU
artifacts are absent from that retained owner.

String external storage has one observation rule in the retained oracle:
`std::less<const void*>` orders the data pointer against the string object
representation. Object-local data is SSO and contributes zero dynamic bytes;
external data contributes `capacity + 1` logical character slots, including the
terminator. The enclosing object already accounts for inline SSO storage. This
is a logical retained element-storage oracle; allocator size-class rounding,
allocation headers, and bookkeeping are deliberately excluded.

CPU SIMD parse, artifact, and compact preparation boundaries are owned by the
[CPU SIMD contract](./cpu/simd.md). Direct execution consumes the kernel-owned
`ArtifactAdmission` used by CPU, Metal, and Vulkan; it does
not define a second parse or artifact policy. That cold token carries the one
parsed admission into compact CPU preparation. Its transient expected emission
has no canonical vector: exact comparison borrows canonical input and compares
the one source owner once. Cold direct admission hashes the borrowed canonical
payload once, then reuses that key for header admission, parse, emission, and
exact comparison. The admission header contains declarations and result
layouts only; one compiled Kernel source owns the comparison and authentication
bodies for every consumer.

Product graph execution instead consumes the immutable capability-authenticated
step minted after final graph fusion and artifact emission. A private
source-only deleter seals the token's `shared_ptr` control block; admission
recovers the typed token from that capability in `O(1)` and validates its
stored id and aggregate facts. There is no process-global capability table,
mutex, or weak-entry scan. Warm admission checks only the retained plan/key/metadata
shape and CPU-versus-GPU ownership law; the returned counters are produced by
that function and are exactly parse `0` / emission `0`. CPU preparation
consumes the retained typed input directly. GPU preparation consumes retained
source and metadata with no parsed-input owner. The public API cannot construct
this private token, and every public backend entry still performs full
artifact admission. Prepared run/submit surfaces accept no artifact, so
subsequent execution remains parse-free, emission-free, and allocation-free
for artifact authentication.

The CPU executor receives pre-lowered SIMD state and a validated binding
layout, then invokes one slice per Kernel physical partition. It must not
compile, allocate a whole buffer, upload, or create a worker pool from a worker
callback. `ComputeTileExecutor::run(callback)` always uses its configured
worker backend. `run_with(worker_backend, callback)` uses the
explicit host backend for the same prepared tile plan. Compute execution never
consults an ambient/thread-local parallel runtime provider. Invalid, nested,
width-mismatched, or non-static explicit backends fail before tile callbacks;
failure never selects another backend. General Kernel `par()` keeps
its separately documented scoped-provider contract.

Accelerator execution does not enter the CPU worker queue. Kernel dispatch
windows remain range, dependency, staging, and evidence plans, while Node
submits one or a planner-bounded small number of backend command streams.

Staged execution requires CPU-visible staged output and validates plan,
artifact, tile count, input count, per-input element widths, per-tile bytes,
output bytes, and numeric-policy obligations before dispatch.

Resident execution uses explicit Node-owned descriptor and handle sidecars.
Resident output does not become CPU-visible implicitly; visibility requires an
explicit download or staged readback owned by the caller.

Resident direct execution uses deterministic contiguous semantic tile windows:
window `begin_sequence` is the first tile index in that dispatch window and
shader `gid` is local to the window. Backends bind resident inputs and output
from one frozen input-window plan derived from `ExecutionMetadata`. A binding
used by direct `Read` or as a `ReadAt` index is windowed and uses byte offset
`window.begin_sequence * element_bytes`. A binding used only by
`ReadUniform`, as a `ReadAt` source, or by both compatible base-anchored uses
binds `{begin_sequence = 0, tile_count = RequiredInputCount(...)}` for every
dispatch window. The staged path packs that same exact source-identity span;
it does not follow the output sequence or replicate a uniform scalar to the
dispatch width. A binding that mixes a base-anchored use with a windowed use
is rejected before execution unless an earlier owner split it into distinct
bindings. This prevents one descriptor base from silently changing either
address law. The runtime rejects non-contiguous resident views
(`stride_bytes != element_bytes`) with `compute_resident_stride_invalid`
rather than introducing a sequence-index shader ABI.

Compute completion order is not semantic authority. Graph dependency order,
stable output identity, and primitive reduction law remain the authority.

Warm staged and resident execution may alter physical allocation and dispatch
reuse, but they must preserve the same semantic tile sequence, tile-indexed
staged scatter where staged output exists, and preflight failure boundary.
Resident direct execution uses the
binding contract's implicit identity sequence for contiguous semantic tile
windows. It rejects an explicit sequence array instead of normalizing or
copying it. Resident execution performs zero sequence-array writes, and window
construction is proportional to the dispatch count `ceil(N / W)`, where `W`
is the frozen dispatch-window tile bound. CPU staged execution also uses
contiguous windows without materializing a sequence because the SIMD runner
consumes the full logical tile range.

Plan, artifact, binding, backend, dispatch-window, and staged-output
validation preserve their precise failing reason instead of selecting another
execution path.
The optional backend `last_error` hook is SDK-free node-owned evidence carried
through the kernel handoff. It is used only when backend execution returns
false; absent, empty, or `ok` last-error values still close as
`compute_backend_failed`.
## Telemetry And Evidence

Kernel plans expose declared dispatch count, dispatch-window tile count,
bytes-per-tile, staging bytes, and API identity. Node telemetry may explain how
caps were observed, but kernel evidence never proves device
availability, driver behavior, shader validity, occupancy, throughput, PMU
counters, hit rate, or timing.

Telemetry boundaries are fail-closed: deterministic plumbing and hash agreement
are not performance, hardware-cache, occupancy, PMU, hit-rate, timing, or
real-device proof. A real backend row needs separate measured evidence before
any performance statement.
## Verification

```bash
tools/test/run kernel.compute
tools/check/run
```
