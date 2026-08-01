# Accel Runtime Resource Contract

## Scope

Node owns native CPU, Metal, and Vulkan resource lifetimes. Public and kernel
headers contain only SDK-free values; platform handles stay in private Node
implementation files.

## Ownership

An admitted device owns one backend adapter and immutable operation table. A
context pins that device. Resident buffers, compiled kernels, prepared runs,
pipelines, descriptors, command storage, and completion state form an explicit
owner chain beneath it. Destruction proceeds from native prepared resources to
borrowed common plans and finally to the adapter.

Resident registries assign nonzero monotonic identifiers. CPU retains its
indexed weak table; Metal and Vulkan use adapter-local hash indices because
released identifiers leave no permanent slot. In every case lookup revalidates
identifier, owner, usage, width, and extent without a linear registry scan.
Metal and Vulkan public handles are the sole native-buffer owners. Final
release erases the weak registry row. Vulkan splits that public lifetime from
the native storage lifetime: an in-flight transfer retains only the storage
through its fence, while the public owner alone retains the adapter. The
storage then returns its device allocation to a fixed 32-slot resident pool.
A registry cannot keep a dead buffer alive, a transfer cannot extend adapter
ownership onto the completion thread, and a native buffer cannot be reused or
destroyed before its last GPU use.

Metal and Vulkan registry rows store type-erased public-owner and native-buffer
lifetime tokens. Lookup code therefore does not depend on either backend's
private owner layout. It locks both weak tokens while holding the resident
lock, validates public-handle identity, and returns the strong tokens with the
native buffer view. Metal's token owns its shared `MTLBuffer`; Vulkan also
stores the stable address inside its separately retained native-storage owner.
Creation and final release are the only translation units that know those
concrete owner and storage types.

Metal and Vulkan keep registry lifetime under a dedicated resident lock rather
than the command/telemetry lock. A path that needs both always acquires command
state first and resident state second; final public-handle or native-storage
release needs only the resident lock. This makes recursive adapter-lock
acquisition structurally impossible. Vulkan's fixed resident pool makes final
storage release allocation-free and chooses the smallest compatible
usage-class allocation that satisfies the requested byte extent.
Metal resident preparation holds its adapter lock once while resolving the
complete ordered input/output set, rather than revalidating a synthetic Device
and reacquiring the same lock for every binding. Forged identifiers,
different control blocks, shifted alias pointers, and cross-adapter resources
fail before native API calls.
Metal and Vulkan preparation project each complete ordered request array
through one backend-compiled batch-lookup owner and one resident-lock
acquisition. Primitive translation units provide only bounded request rows;
none duplicates registry iteration or hash-table implementation.
Each prepared primitive then retains the successful lookup result, not only
the projected native pointer. In particular, Vulkan Gather owns the strong
public-handle and native-storage tokens for values, indices, resident count,
and output until its prepared resource owner is destroyed. Descriptor bindings
are derived from those owners and are not a second lifetime authority. A
released caller handle therefore cannot leave an encoded Gather command with
a dangling `VulkanBuffer`, and an invalid output owner reports
`compute_resident_id_invalid` rather than a generic Gather-shape failure.
Context transfer admission is also the sole generic null, extent, and owner
validator for a batch. Its caller provides bounded route storage, and admission
projects logical `UploadEntry`/`DownloadEntry` rows into admitted
`UploadRoute`/`DownloadRoute` rows in place. The two vocabularies describe the
one state transition rather than two interchangeable request authorities. The backend
dispatch layer forwards those admitted rows without a second validation sweep;
Metal and Vulkan then perform the distinct native-registry resolution under
their one resident lock. For `N` requests, transfer admission is one
`Theta(N)` pass and uses no transient routing-vector allocation. Public
capability validation and native lifetime validation remain mandatory. The
native backend's resolved transfer plan is the sole temporary routing owner.
Node Compute bounds its largest transfer batch to 64 routes: Metal and Vulkan
download keep that complete common path in fixed inline storage, including
Vulkan chunk/barrier rows and incremental hashes, so reusable checkpoint
export adds no runD-owned C++ heap allocation after warm-up. A Vulkan export
still performs a native queue submission; any allocation inside the loader,
driver, or translation layer is implementation-dependent and remains visible
to process-wide instrumentation but is not route-plan allocation. Generic
callers above 64 routes use the backend-owned overflow plan and do not redefine
the bounded Compute contract.

CPU, Metal, and Vulkan resident buffers share one `ResidentEntry` identity,
shape, capability, and owner state prefix and one compiled view validator.
For a requested view with positive count, its addressed extent is

```text
extent = (count - 1) * stride_bytes + element_bytes
```

Both arithmetic operations are checked before the validator requires
`offset_bytes + extent <= ref.bytes <= entry.bytes`. Unknown usage, a
capability mismatch, zero shape, dense-stride mismatch, multiplication
overflow, addition overflow, range overflow, expired ownership, a different
control block, and an aliasing pointer are rejected with the same reason on all
three backends. A strided view is accepted only by the explicit view path.

CPU keeps an ID-ordered weak-owner array, so ID resolution is one checked
`id - 1` index operation rather than a scan or hash lookup. Each backend locks
the resident owner exactly once per request and carries that strong owner into
the successful result. Validation adds no allocation, buffer copy, virtual
dispatch, second registry pass, or repeated weak-owner lock while projecting
the backend handle. CPU primitives submit only bounded request rows to one
compiled batch owner; one lookup loop serves the complete batch.

## Prepared Execution

Preparation resolves resident bindings, validates backend limits, and acquires
reusable native resources once. Standalone Vulkan preparation also records the
complete ordered step stream into one Job-owned secondary command buffer at
this boundary. Pipeline-private preparation is an explicit internal mode: its
resources cannot be submitted through the standalone or Batch paths, and
Vulkan leaves them unrecorded until the Pipeline has fixed status ownership and
barriers, then encodes their prepared steps directly into the Pipeline's one
primary command buffer. A warm
run must not rediscover resident buffers, rebuild source, recompile a pipeline,
re-encode dispatches or barriers, or allocate workload-sized host command
storage. Metal retains its backend-native warm encoding boundary.

Dense View normalization follows the execution owner. A standalone Job owns
its dense storage. A Pipeline supplies the immutable typed View layout and
resident bindings from its canonical memory plan, so every sequential Program,
recurrence phase, and transactional alternate borrows the same planned arena.
The backing slots are raw words: sequential uses of different scalar types may
share one slot, while the planner preserves the strongest scalar and backend
offset alignment required by every use.
A Pipeline-private backend preparation that needs an unplanned dense View fails
with `compute_pipeline_memory_plan_invalid`; it cannot allocate a hidden
fallback owner. Vulkan records the 48-byte View transfer parameters as push
constants and uses only source and target storage descriptors.

Vulkan shader-module ownership is confined to synchronous executable
construction. One move-only module owner consumes the compiled SPIR-V,
supplies its handle to `vkCreateComputePipelines`, and destroys that handle
before the executable is published to either the Map or collective cache.
The published executable retains its exact artifact key and source,
descriptor layout and storage, pipeline layout, and `VkPipeline`. Pipeline
acquisition is serialized by the adapter, so the shader-module handle
invariant is

```text
module_current <= 1
module_current == 0 after each acquisition
```

The module handle is construction state, not Program state, graph identity, or
warm execution state. Its lifetime cannot affect dispatch order, descriptor
binding, Fixed arithmetic, status priority, output bits, or output hash.

Metal and Vulkan serialize command publication only for the adapter resources
that require it. Independent prepared jobs may remain in flight. Reuse of one
job is guarded by its own claim and fails with `compute_job_busy`; device and
runtime control failures keep their distinct reasons.

Metal prepared execution creates command buffers with unretained resource
references, so warm submission performs zero driver-side resource retains and
releases. The Job owns the complete prepared resource graph until synchronous
finish or the asynchronous completion callback clears the active submission.
No borrowed buffer can enter this path. Command-buffer completion therefore
precedes the last possible release of every encoded pipeline, buffer, and
parameter owner by construction rather than by an API fallback.

Vulkan owns one adapter-local envelope of eight command slots. Each slot owns
exactly one command pool, primary command buffer, fence, and optional two-query
timestamp pool. Preparation creates the complete ring before publishing a Job;
each prepared Job separately owns one command pool and one reusable secondary
command buffer whose descriptors, pipelines, buffers, push constants, resets,
dispatches, and barriers reference that Job's retained resource graph. Warm
submit therefore claims fixed primary storage, writes only the timestamp
wrapper plus `vkCmdExecuteCommands`, and performs no command allocation or
graph-sized re-encoding. The caller records and submits that primary wrapper
while holding only the queue-publication lock.
The completion service waits for the oldest submitted fence without that lock,
then retires callbacks in submission-sequence order. Eight independent Jobs may
remain submitted; a ninth explicit execution admission while all slots are
active fails immediately with `accel_vulkan_command_capacity`. Ordinary upload
submission instead applies bounded condition-variable backpressure until one
of those same slots retires. Queue pressure can therefore delay an upload but
cannot nondeterministically turn valid input into a transfer error. There is no
64-entry host backlog, busy polling, or completion-worker start adapter.
Zero-initialized resident creation applies the same backpressure before taking
resident state and recording its clear command; full-overwrite creation records
no clear command. Queued uploads can therefore delay zeroed allocation but
cannot surface transient command pressure as buffer capacity, and the wait
preserves command-lock-before-resident-lock ordering.

Synchronous and asynchronous execution share one prepared submission owner:
command claim, encode, end, submit, timestamp, finish, and evidence projection.
The synchronous terminal submits through that owner and observes its completion
with one stack-owned atomic wait; there is no backend-specific prepared-run
function or second statistics path. A full resident upload uses that same queue
and returns after ordered submission rather than waiting for its fence;
the claimed command slot retains both the mapped staging owner and the native
target storage until completion. It does not retain the public owner or adapter
lifetime.
The next submission on the one Vulkan queue observes the transfer through the
recorded transfer-to-compute barrier. Download and unaligned boundary-word
preservation remain synchronous observation points. A download waits for one
free command slot, submits its copy behind prior queue work, and waits its own
fence; it does not drain unrelated commands before staging allocation. The
host staging-to-caller copy then runs outside the queue-publication lock. An
adapter-local active-readback count pins one statistics epoch across the whole
logical download, including every bounded staging slice and every unlocked host
copy interval. Return waits only for completion bookkeeping that was already
pending at the download submission sequence, not commands submitted later.
Each prepared command
carries its own submit count, dispatch count, submit interval, timestamp, and
observed slot occupancy into completion. A Job submission never resets or reads
an adapter-global counter epoch and therefore never drains unrelated commands
for telemetry. Counts compose by saturating addition; immutable capacity and
observed occupancy compose by maximum. An explicit adapter-statistics reset is
the only global evidence-epoch boundary and may quiesce earlier commands; it is
not part of prepared Job execution. Resident Jobs own disjoint prepared input,
output, status, and scratch buffers, so independent submissions carry no global
compute barrier. Explicit caller-buffer execution remains synchronous under the
adapter execution claim. No mapped staging owner is reused before its fence
completes. Adding borrowed-buffer asynchronous Jobs would require a separate
buffer-range hazard owner; it may not reintroduce a global barrier.

Backend-local Kernel and Pipeline submissions use one common concrete
`submission::State<Owner>` transition. The non-null owner pointer is the sole
active-state authority; there is no mirrored boolean. `Begin` takes the one
mutex, rejects a busy owner or null completion, and publishes owner,
completion, and user together. `Take` clears all three under that mutex before
native finish and user completion, so a duplicate backend callback observes no
claim and cannot publish twice. `Cancel` is the only failed-publication
rollback. Metal and Vulkan instantiate this template with their concrete
prepared owner. The transition adds no virtual call, type erasure, extra
function-pointer layer, atomic, allocation, copy, or lock.

Prepared step status folding is one common increasing-index operation. The
first failed-batch metadata is retained from the first nonzero row,
failed-batch counts accumulate in that order, and the first non-success check
terminates the fold. Backend finish functions retain only their native step
observer and last-error publication. Pipeline control telemetry likewise has
one common field projection into `RuntimeStats`; Metal and Vulkan own only how
the 128-byte control is observed.

The common prepared implementation is physically owned by
`kernel/prepared/{run,batch,pipeline,completion,evidence}.cpp`, with immutable
run and Pipeline state in `model.hpp`. Preparation performs the complete kernel
admission once. A warm submit compares the supplied Context with that frozen
admission in constant time and does not re-admit or traverse the Kernel graph.
The prepared state retains the admitted kernel owner through
`KernelExecution`; it does not keep a second `AccelKernel` owner mirror.
`completion.cpp` is the only common submit and completion authority for both
Jobs and Pipelines. Release/acquire ordering on the stack-owned completion flag
makes all projected evidence visible to the synchronous waiter even when a
backend completes inline; testing the flag before blocking prevents a lost
wakeup.

Metal and Vulkan Scan lowerings share one fixed physical width of 128 lanes.
For logical block size `B`, lane `l` owns the contiguous interval

```text
q     = ceil(B / 128)
begin = block_begin + l*q
end   = min(begin + q, block_end)
```

Each lane performs one increasing-index local sweep, the 128 lane totals pass
through the fixed prefix tree, and each lane replays its interval with that
prefix. No element-count threshold selects another width or algorithm. For
U32, Metal threadgroup scan storage is `2*128*4 = 1024` bytes; for
U64 it is `128*8 = 1024` bytes. Resident width is fixed at 128 lanes per
workgroup.
Modulo addition reconstructs the same stored prefix bits; overflow is evaluated
at each element from the reconstructed predecessor, input, and successor, so
reducing physical lanes does not weaken signed or unsigned failure evidence.

## Prepared Batch

The public Compute Batch projects one bounded native submission over 1 to 64
already prepared Jobs on the exact same Device. Metal encodes the ordered Jobs
into one command buffer and commits once. Vulkan executes each Job's recorded
secondary command from one existing primary ring slot and calls `vkQueueSubmit`
once. Every Job may retain a different
graph, Program, signature, shape, and pipeline; this boundary amortizes only
command publication and never claims graph or dispatch fusion.

All backend resources are validated before submission. The common prepared
owner then claims every Job-local submission gate before marking any active.
A duplicate, invalid, or busy prepared owner rejects the complete Batch without
execution. After submission, backends finish Jobs in admission order and
preserve each exact status, overflow reason, and graph/output evidence.
For `N <= 64` Jobs, one `Theta(N)` admission pass validates Context and backend
identity, builds the ordered backend rows, and saturating-accumulates immutable
planning evidence. The required `Theta(N)` Job-gate claim remains, but there is
no second evidence scan, workload-sized allocation, or additional warm lock.
The backend then owns one native submission. Thus host work remains
`Theta(N)`: one prepared-state/evidence traversal plus the separately required
gate claim and result projection, rather than two prepared-state traversals.
Auxiliary storage remains the fixed 64-Job envelope.

One shared evidence snapshot owns execution submit count, queue pressure,
kernel timing, and submit-wait timing. The backend writes that snapshot from
the one submitted command; Batch does not reset or subtract adapter-global
counters. The common prepared owner derives shared dispatch count from the
already admitted Job plans. Per-Job evidence immediately after the Batch
deliberately leaves shared execution fields zero; it retains only Job-local
planning and result facts. A later explicit read may own a distinct transfer command.
The complete product
admission and observation law is owned by the
[Compute Batch contract](../../compute/batch.md).

## Memory and Telemetry

Resident, staging, device, and transfer bytes are separate categories. Current,
peak, cumulative, reused, and budget counters saturate at `UINT64_MAX`. A
prepared owner records the physical extents it owns; `Job::memory()` aggregates
those owners directly rather than subtracting global snapshots.

Vulkan has one memory-placement authority. `Resident` buffers and
device-internal collective scratch require `DEVICE_LOCAL` memory and are never
mapped. `Staging` buffers require `HOST_VISIBLE | HOST_COHERENT` memory, remain
mapped for their lease, and are the only buffers charged to the staging meter.
The reusable pool key includes both effective Vulkan usage and memory class, so
a host-visible allocation can never satisfy a resident or device-local request.
There is no memory-class fallback. Physical-device memory properties are
immutable for an admitted adapter, so admission snapshots them once and every
allocation reuses that snapshot instead of repeating the driver query.

Vulkan also snapshots `minStorageBufferOffsetAlignment` and
`maxStorageBufferRange` once. One descriptor admission function rejects a null
buffer, zero or whole-size range, unaligned offset, device-range overflow, or
typed owner overflow before `vkUpdateDescriptorSets`. Every descriptor writer
consumes that authority and propagates its typed failure; no primitive can
bypass it with a raw update.

A strided dense-primitive View whose complete physical span exceeds that
per-binding limit is normalized through canonical descriptor pages. For page
start ordinal `b`, authored offset `o`, stride `s`, element width `w`,
alignment `a`, and limit `L`, the sole page planner computes

```text
q = o + b*s
base = floor(q/a)*a
prefix = q - base
capacity = 1 + floor((L - prefix - w)/s)
count = min(remaining, capacity)
```

Admission proves `prefix + (count - 1)*s + w <= L` and backing containment
before allocating a descriptor. Pages consume increasing `b`, are disjoint in
logical ordinal space, and cover `[0, count_total)` exactly once. Each page
writes the same dense ordinal `b + i`; only the descriptor base changes, so
the copied bits and downstream graph are invariant under the page count.

The same immutable snapshot owns both Vulkan compute-grid limits
`X = maxComputeWorkGroupCount[0]` and
`Y = maxComputeWorkGroupCount[1]`. View normalization, terminal publication,
and recurrence-window copies use one overflow-free law for `N` work items and
fixed width `W`:

```text
G = 1 + floor((N - 1) / W)
x = min(G, X)
y = 1 + floor((G - 1) / x)
accept iff N > 0, W > 0, x <= UINT32_MAX, y <= Y, y <= UINT32_MAX
```

The subtraction-first ceiling formulas cannot overflow. Grid planning happens
once during cold preparation and the retained `(x, y)` is encoded on every
warm run. Publish and window preparation consume the retained device limits;
View execution consumes the retained grid without division or repeated count
validation. The View shader linearizes `(workgroup.x, workgroup.y, lane)` back
to the same increasing logical index, so bit order and failure precedence do
not change.

Logical resources suballocated from one arena are compared as half-open byte
ranges, not by resident owner identity. For two views on the same owner,
overlap is exactly `a.begin < b.end && b.begin < a.end`; disjoint offsets are
independent bindings. Transform admission and recurrence use this same range
law. Producer-consumer telemetry counts logical view bytes rather than the
whole physical arena, so packing cannot inflate traffic evidence.

Map binds each canonical dispatch window rather than one whole logical value.
It aligns the descriptor base down and specializes the binding's one frozen
base constant to the alignment byte prefix. Every direct read/write and
indexed `ReadAt` address is emitted from the same form

```text
binding_base + logical_index * binding_stride
```

where `logical_index` is the dispatch lane for a direct binding and the
validated index value for an indexed source. Thus native descriptor base plus
the specialized address equals the authored byte address bit for bit. Runtime
specialization replaces one base declaration and one stride declaration per
binding; it never searches for backend-source use patterns. This covers
ordinary and wide writes as well as both the index and source sides of
`ReadAt`, while every bound range remains within the device limit. No copy,
extra dispatch, runtime size strategy, or backend graph is introduced. A
dense Map freezes one common `InputWindowPlan` per input from the admitted
execution metadata. Direct and `ReadAt`-index inputs retain the canonical
dispatch window. Uniform-only and indexed-source-only inputs instead retain a
base-anchored source-identity span beginning at zero with the exact
`RequiredInputCount`; compatible uniform plus indexed-source use takes their
maximum. Metal and Vulkan consume the same plan for resident descriptors and
staged packing, so a count-one broadcast transfers and binds one element for
every native window without materializing a broadcast Buffer. Mixing either
base-anchored use with direct or index use in one binding fails preparation;
there is no descriptor-base heuristic or backend-specific reinterpretation.
A dense primitive whose shader has no prefix operand uses the existing
device-local View normalization when its authored offset is not a legal
descriptor base. The copy is reported as internal round-trip traffic and never
becomes a host transfer or fallback.

Reset consumes that same View-lowering decision instead of deriving staging
from stride a second time. An external binding resets its unique dense
replacement when View lowering created one; an internal binding resets its
exact arena interval. Two reset intervals backed by the same resident owner
remain two routes unless their logical byte intervals are identical, which
admission rejects as duplicate authority. Owner identity alone never drops a
route and no reset widens across alignment padding or another suballocation.
Vulkan plans each reset descriptor as

```text
base  = floor(offset / storage_alignment) * storage_alignment
span  = (count - 1) * stride + element
range = offset + span - base
```

and admits it only when `range <= maxStorageBufferRange`. The shader receives
`offset - base`, so its selected byte addresses are exactly the authored
addresses. The descriptor never widens to the physical arena and never binds
an unaligned logical offset.

`DeviceInfo::storage_alignment` and `storage_bytes` expose the immutable
alignment and per-binding storage limit used by this admission law.

Vulkan Sort parameter rows use
`ceil(sizeof(SortParams) / alignment) * alignment` bytes. Descriptor offsets
therefore satisfy the same device law for every radix pass; padding is cold
parameter storage and neither changes parameter bits nor adds a warm command.

Public resident transfers cross one explicit staging boundary. Vulkan buffer
copies have four-byte alignment, so a semantic byte interval `[o, o + n)` is
partitioned by the frozen staging budget `S`, where `S` is rounded down to a
four-byte multiple and is at least four bytes. The slices are consumed in
strictly increasing semantic-byte order. Their exact count is

```text
K = 0                                      when n = 0
K = ceil((n + (o mod 4)) / S)             otherwise
```

Each slice is independently normalized to

```text
begin = floor(slice_begin / 4) * 4
end   = ceil(slice_end / 4) * 4
```

and `end - begin <= S`. General and batch upload/download call the same
partition, range, preservation, copy, and evidence implementation; there is no
whole-request staging path. A one-slice nonoverlapping upload transfers staging
and target-storage ownership to the command slot and performs no host fence
wait. A multi-slice upload completes slices in canonical order and reuses one
bounded staging allocation. A partial unaligned slice first preserves only the
intersected boundary words, patches the mapped staging interval, and copies the
normalized interval back. At most two four-byte words are read back, and two
edges in the same word are preserved by one copy region. Those regions share
one copy command, one source barrier command, and one host-visibility barrier
command. An aligned partial upload has no preservation readback. A full
semantic upload does not read the padding word and clears only its zero to
three staging padding bytes; it never clears bytes that the caller copy
overwrites. Downloads expose only the requested semantic bytes. Hashing, when
requested, consumes those bytes in the same increasing order across slice
boundaries. Transfer telemetry counts `n`, never padding or the internal
boundary preservation copy, and reports `staging_peak_bytes <= S`.

The internal batch route carries one explicit completion policy. General
resident upload requests `Queued`, preserving the one-slice ownership-transfer
rule above. A caller that cannot publish state before observing execution may
request `Complete`; Vulkan then waits for every batch command and reports a
completion failure synchronously. Pipeline checkpoint restore is that caller.
Metal shared-memory upload is complete when its copy returns under either
policy, so the policy adds no second Metal transfer implementation.

Vulkan cold buffer reuse has two independent memory-class pools: temporary
execution buffers and public resident storage. Each uses the same deterministic
FIFO retention law, a count cap of 32, and byte cap `P = 32S`. A returned
buffer larger than `P` is destroyed; otherwise the oldest retained buffers are
destroyed until both caps admit it. Best-fit acquisition changes only which
physical allocation serves a request, never semantic bytes or ordering. The
temporary pool's mapped staging bytes remain in physical
`MemoryStats::staging.current` after lease return; `peak` is the maximum of
active plus pooled physical staging, while `cumulative` and `reused` remain
lease traffic. Thus an idle cached allocation cannot disappear from telemetry,
and neither pool can retain a workload-sized multi-gigabyte buffer.

Metal Pipeline status metadata separates canonical entry order from status
source policy. For `Q` canonical status entries and `C` source bindings, one
eight-byte entry row stores only `(source ordinal, absolute raw word)`, while one
32-byte source row stores encoding, declared step, four packed 16-bit
`(reason, priority)` policies, and limit. The entry's observed word is resolved
to its absolute packed-arena word during preparation. Retained metadata is
exactly `8Q + 32C` bytes before parameter-arena alignment. Since every source
contributes at least one entry, `Q >= C`. Preparation first collects all `C`
sources, allocates the `Q` entry rows exactly once, and fills them in canonical
declaration order. The device reducer still selects the first nonzero canonical
entry and projects its source's declared step; the packed control ABI and
failure ordering do not change.

Sort key/value ping-pong storage, radix tables and indirect arguments; scan
totals; reduction partials; partition masks, offsets, and totals; segmented
collective index storage; and every internal accelerator status range use the
device class. Parameters remain staging because the host writes them at the
prepare boundary. A standalone status owner pairs its device-local shader range
with one mapped readback range bounded by four U32 words. It copies
`min(status_bytes, 16)` bytes, enough for the reason, first failing ordinal,
and bounded telemetry header without making a large primitive status range
host-visible. A Pipeline-private Vulkan status owner omits that range: the
Pipeline's single 128-byte terminal control observation is its only
host-visible status allocation. The `C` private one-word sources use zero
mapped bytes and zero staging allocations. Standalone finish records

```text
shader write -> transfer read -> copy bounded status prefix -> host read
```

and the CPU reads only the words owned by the operation evidence interface
after fence completion. The first failing prepared step is the sole failure
projection authority; its operation may publish a first-invalid ordinal into
the common runtime statistics. Later step status cannot overwrite that
evidence. A segmented status
range keeps all per-block atomics device-local while transferring the one
reduced reason. On a discrete adapter the workgroup path performs no
host-coherent or PCIe writes; unified memory uses the same explicit ordering.
There is no memory-class fallback and no second semantic status value.

A Pipeline-private Vulkan numeric status range follows the same ownership law,
but its standalone observation spans every batch word. For `Q` numeric status
words it omits one mapped staging allocation of exactly `4Q` semantic bytes,
plus its buffer-to-buffer copy and transfer-to-host barrier. Its device-local
status range remains bound directly to Pipeline canonicalization. Standalone
numeric preparation retains that full readback range and finish behavior.

Every status value that accumulates errors with an atomic operation is reset by
the same recorded command immediately before its first shader use. One central
Vulkan reset owner emits
`compute/read-write -> transfer/write -> fill ->
transfer/write -> compute/read-write`. The first edge is a required WAR
dependency, not defensive synchronization: a later Program may reuse a public
status range only after Pipeline canonicalization has read the preceding
Program's value. Each Program reset span applies the same edge at its exact
first-write frontier, including shader-based strided resets. Preparation
does not clear those words. Status-producing shaders that overwrite their
complete status range retain that single shader authority instead. This keeps
`failure -> write(valid input) -> success` reproducible on one prepared Job,
prevents a later clear from erasing earlier semantic evidence, and makes
secondary-command reuse independent of retained result state.

Explicit upload, write, download, and read boundaries own transfer evidence.
Kernel execution does not manufacture user-visible traffic. Pipeline compiles,
cache hits, descriptor work, buffer allocation/reuse, dispatches, submissions,
kernel samples, and readback time remain diagnostic and do not enter semantic
identity.

Vulkan projects the immutable `command_capacity`, observed
`command_inflight_peak`, and saturating `command_capacity_rejections` through
the same RuntimeStats, AccelEvidence, and public `compute::Stats` path.
`telemetry::Profile::command_pressure()` derives peak/capacity without storing
a second ratio. A zero rejection count at or below capacity is the warm-path
contract; a nonzero count identifies submission-envelope pressure directly.
For one Job, these values are captured from its command lease at publication and
completion. For an envelope of independent Jobs, totals use saturating addition
for submits, dispatches, and rejections and maximum for capacity and occupancy;
no global snapshot subtraction or reset participates in that projection.

Explicit Metal and Vulkan statistics reads and resets wait for the
active-host-readback count to reach zero. Vulkan reset then quiesces submitted
commands as the documented global epoch boundary. This keeps the
dependency-bound payload copy out of the publication critical section without
allowing its bytes or `readback_ns` to cross a reset epoch.

## Verification

`accel.backend-runtime` verifies ring capacity, non-wrapping sequence order,
counter saturation, quiescent epoch reset, host-readback/reset exclusion,
readback-safe adapter destruction, Metal and Vulkan weak-registry release,
resident pool reuse, disjoint staging/device memory properties, the
device-local-status/mapped-U32-readback pair, transfer
flags, rounded physical storage, and exact unaligned subrange preservation.
`runtime.compute-accel` verifies eight independent Vulkan Jobs, precise ninth
rejection, per-command evidence, allocation-free warm admission, completion,
and graph/output hash parity through both synchronous and asynchronous
terminals. `compute.batch` verifies one-submit Metal/Vulkan execution, atomic
bounded admission, ordered heterogeneous results, transfer backpressure beyond
the Vulkan command envelope, and evidence ownership.
`compute.resident-write` verifies that scan, reduce, and gather recover from a
failed status on the same prepared Metal and Vulkan Job without recompilation
or a second status authority.
`compute.memory` verifies Program and Job aggregation and warm
stability. Backend availability is proved separately by required selection and
installed product contracts.
