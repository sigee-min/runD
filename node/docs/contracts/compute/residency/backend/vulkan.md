# Vulkan Residency

Vulkan cold preparation keeps two capabilities distinct. The reusable rolling
selection owner retains the normal control prefix, exact selected locals, and
control suffix for both direct and indirect selected dispatches. The bounded
owner is admitted only when every selected payload dispatch can be captured by
the GPU-visible indirect admission gate; Graph and other ineligible shapes may
therefore keep rolling selection without falsely becoming native windows. The
raw bounded adapter accepts `1<=Q<=4`, validates every fixed batch and adjacent native
owner before mutation, reserves four independent Host-ready timeline cells and
one ordered Device-done timeline, and calls `vkQueueSubmit` once with `Q`
`VkSubmitInfo` records. Each record waits its own ready cell at
`DRAW_INDIRECT|COMPUTE_SHADER`, executes the retained selection through the
GPU-visible indirect gate, and signals its exact done value. The caller returns
after that queue call; a cold residency completion lane waits terminals and
emits fixed Releases plus one Final. Main-thread per-epoch submit/wait is not
part of this route.

The timeline source is direct-owned under
`node/src/accel/vulkan/timeline/owner/`: `validation.cpp` contains shared
capability/error/proc/value checks, `lifecycle.cpp` contains feature query and
semaphore create/destroy reset, `generation.cpp` contains capability and
generation reserve/cancel/close, `point.cpp` contains point and ready
preflight/signal, `submit.cpp` contains the single-batch queue path,
`batch.cpp` contains window/stream transient `VkSubmitInfo` lowering, and
`observe.cpp` contains done waits and counter reads. `internal.hpp` is
declarations-only; `owner.hpp` remains the sole mutable timeline-state owner.

Ready cells have independent monotonics, so Host may signal epoch one before
epoch zero and a long sequence of Q=1 windows does not create an unbounded
value jump when Q later becomes four. The global done sequence retains queue
order. Duplicate, stale, wrong-bank, adjacent-alias, and premature same-bank
reuse signals fail before their GPU-visible admission cell changes. Normal
admission seeds the exact stride-aware terminal generation only after those
checks. Failed admission zeros the indirect arguments and records a Known,
completed, dispatched, non-writing receipt; selected payload and publication
therefore remain suppressed while the accepted queue suffix drains.

The exact-generation emergency abort is drain, not cancellation. The cold
service opens every still-unsignalled gate in bank-safe order, waits its done
point before reusing the same gate storage, and publishes every accepted batch
conservatively as `UnknownMayWrite`. One Unknown Final quarantines the adapter
and retained prepared owners, and raw callback/user pointers are cleared. The
ordinary fence-completion lane is separate, so residency Host service cannot
head-of-line block unrelated command retirement. Worker lifetime is published
before thread start, retained only around dequeued work, and final-owner
destruction from a worker is transferred to a neutral cold deleter; an idle
worker cannot pin the Device or self-join.

The explicit/resident Direct Virtual Q=2..4 DeviceVsm route is admitted only
when both fixed VSM
Input and Output banks expose authenticated, full-buffer, stable
`HOST_VISIBLE|HOST_COHERENT` views. That allocation policy is carried by a
source-private VSM-only `HostVisiblePreferred` intent. Ordinary public Buffer,
Pipeline resource, control, intermediate, workspace, and scratch allocations
remain Device-local and unmapped; Vulkan resident reuse never crosses the two
allocation classes. When an allowed coherent Device-local type exists,
including the exercised MoltenVK unified target, it enables direct Host
fill/readback, exactly one window queue call, zero H2D/D2H queue submissions,
and exact output. A discrete or otherwise noncoherent adapter receives no
native window submit and keeps the established rolling transfer route until a
distinct transfer-queue timeline DAG exists. Issuing a blocking compute-queue
upload or download behind an unopened ready gate is forbidden.

Focused evidence covers raw Q=1..4, out-of-order first-bank readiness, Q4
suppression/no-write, public Direct MoltenVK Q=2..4 output, exact one queue
call, four native batches, actual inflight peak, abort/Unknown quarantine, and
60 warm raw windows with unchanged producer-backed Vulkan
Pipeline/descriptor/buffer allocation counters. The separately reported
global `operator new` diagnostic includes MoltenVK's opaque work inside
`vkQueueSubmit` and is not relabeled as a runD-owned allocation count.

Each raw Vulkan chunk remains bounded to four and uses one `vkQueueSubmit`, but
coherent Direct pointwise Q>4 production now prefers one whole Schedule. The
lowering materializes Q transient `VkSubmitInfo`/timeline records, cycles four
independent ready cells over reusable simultaneous-use commands, and submits
all Q batches with one actual `vkQueueSubmit`. The records are explicitly
reported O(Q) transient Host storage; retained Plan, role, timeline, and
Authority state stay O(1) in Q. Public evidence is one handoff, Q batches, and
one queue call, including Q=17.

The legacy `OneSubmit` persistent service uses the same one-call `VkSubmitInfo`
stream for Q=5/9/257 and performs no per-coordinate native encode, submit, or
callback. The `BackendChunked` product instead accepts two-coordinate chunks,
uses two physical payload slots, and performs `ceil(Q/2)` queue submissions;
Vulkan may additionally retain four fixed authentication metadata cells.
The persistent submit implementation is physically divided under
`pipeline/residency/persistent/submission/`: the parent `submission.cpp` owns
request admission and Control-to-Run-to-Adapter lock ordering,
`continuation.cpp` owns later chunk timeline reservation and submission,
`initial.cpp` owns first-run materialization and native acceptance, and
`lifecycle.cpp` owns attempt cleanup and control reset. `internal.hpp` exposes
only their private declarations; the run and control models remain the sole
mutable authorities.
The serialized Window fixture remains OneSubmit with one native submission;
the Pointwise R2 Q=2/3/5 product is the distinct chunked path. Neither mode
claims same-submit GPU ownership of nonresident backing data.
On Apple, MoltenVK expands that one Vulkan call into driver-private Metal
command buffers. During `VkInstance` creation runD therefore enables the
enumerated standard `VK_EXT_layer_settings` extension and installs
`MoltenVK/MVK_CONFIG_MAX_ACTIVE_METAL_COMMAND_BUFFERS_PER_QUEUE=1024`. The
persistent capability is limited to that configured bound; if the extension
is absent, its capacity is zero and the route fails closed. The reported O(Q)
transient byte count covers runD-owned `VkSubmitInfo`, timeline-value, and
stage-mask arrays only. MoltenVK's opaque native command-buffer memory is not
observable or relabeled as runD-owned memory. Raising the ceiling changes the
instance-wide maximum, but ordinary routes allocate native command buffers
only on demand and retain their existing queue semantics.

The whole-run Vulkan schedule is physically direct-owned under
`node/src/accel/vulkan/kernel/pipeline/residency/schedule/`:
`entry.cpp` owns request validation, role arithmetic, and the cold capability
projection; `submission.cpp` owns role materialization, generation/point
reservation, and the one `vkQueueSubmit`; `signal.cpp` owns authenticated ready
gate selection and Host timeline signaling; `terminal.cpp` owns the service
wait, no-write drain, ordered Release delivery, and the one Known/Unknown
Final; and `cleanup.cpp` owns active-slot clearing and exact-generation abort.
`internal.hpp` contains only helper declarations and the fixed schedule
protocol views. The shared residency model is itself divided under
`pipeline/residency/model/`: `base.hpp` owns window/schedule/persistent run
state, `admission.hpp` owns candidate and diagnostic snapshots, `graph.hpp`
owns backend graph proofs, `sliding.hpp` owns the gate payload and retained
gate resources, and `selection.hpp` alone composes the cold selection owner.
The adjacent `model.hpp` is only the stable include surface.
`VulkanResidencyScheduleRun` remains the sole mutable schedule-run authority;
no Q-entry state mirror is introduced.

The Vulkan sliding-gate contract is likewise divided by proof ownership:
`pipeline/vulkan/sliding_gate.cpp` owns coordinate progression, stale-gate
rejection, and cold-versus-warm allocation stability, while
`pipeline/vulkan/sliding_gate/terminal.cpp` owns the paused terminal-frontier,
peer-submit, device-loss, and adapter-quarantine proof. Their `local.hpp`
contains only the callback result carrier and cross-owner declarations; it
does not duplicate gate, pipeline, or adapter state.

The bounded Vulkan residency Window submission is physically direct-owned under
`node/src/accel/vulkan/kernel/pipeline/residency/submit/`:
`entry.cpp` owns request validation, batch materialization, active-window claims,
timeline reservation, and the one queue submission; `signal.cpp` owns
authenticated ready admission and timeline signaling; `terminal.cpp` owns
suppression, timeline waits, ordered Release delivery, and the one Known/Unknown
Final; and `abort.cpp` owns the abort/quarantine transition.
`VulkanResidencyWindowRun` remains the sole mutable bounded-window authority;
the split introduces no second callback, receipt, or timeline state.

Vulkan prepared-pipeline recording is direct-owned under
`node/src/accel/vulkan/kernel/pipeline/prepare/record/`: `describe.cpp` owns
route dispatch-count projection, `recipe.cpp` owns immutable record-recipe
construction and capacity failure mapping, and `encode.cpp` owns command
recording lifetime, capture/replay selection, and final publication.
`encode/entry.cpp` owns ordered occurrence admission, shared-scratch barriers,
dispatch scopes, and nested-window advancement; `encode/evidence.cpp` owns
per-step canonical status folding and telemetry visibility. Their private
encoding context borrows the same pipeline, recipe, and capture; it allocates
no state and cannot outlive the recording call. Timestamp and dispatch scopes
still surround the same native commands, and
`accounting.cpp` owns the checked and retained recipe host-byte bounds.
Pipeline teardown destroys recorded native commands under the adapter lock,
then releases the record recipe after unlocking. A recipe can retain the last
prepared-kernel owner, whose independent resource deleter must reacquire that
same non-recursive adapter lock; dropping it inside the pipeline lock would
self-deadlock. The native persistent lifecycle contract exercises this final
owner release, not only cases with an externally retained kernel owner.
Preparation failure likewise keeps its cleaned, non-published pipeline alive
until the enclosing preparation lock has unwound; it must not destroy the
pipeline's locked submission mutex inside the failure helper.
`compute.pipeline-vulkan-graph-admission` checks normal and failed preparation
release with a final-owner deleter: a separate observer must acquire the adapter
lock, and failure cleanup must retain the mutex-owning object until unlock.
`VulkanPipelineRecordRecipe` remains the only record model; no capacity,
dispatch, or capture authority is duplicated in this split.

Vulkan separately has a genuine fixed-structure Direct Map recurrence. Common
recurrence lowering rewrites an admitted pointwise Map into one shader whose
eight-byte `RundDispatch` push-constant row carries `tile_count` and
`iterations`; the shader owns the `iterations` loop. The Vulkan route retains
one descriptor-set lease and cold-records one `vkCmdDispatch` into one reusable
primary `VkCommandBuffer`. Dedicated actual Q=5/9/257 evidence requires exact
output, one queue submit, one dispatch, one command buffer, one descriptor set,
and unchanged exact runD-owned parameter-buffer and recurrence-route Host
bytes. `VkCommandBuffer` implementation allocation bytes are opaque in Vulkan,
so the fixed-native-storage claim is limited to those fixed native object
counts and exact runD-owned allocations; it is not an opaque driver-byte
claim.

That fused Direct primitive is not the persistent service lowering. Its shader
has no Authority descriptor/service ring ABI and cannot consume Host
ready/wait/ack turns. The distinct Authority-proved service-free Direct ABI now
connects the public terminal-output and `write_each` History
`Pipeline::repeat<Q>::run()` routes to this primitive. Actual Q=5/9/257
product tests require one queue submit, one payload dispatch, no
per-iteration Host service/callback, one Final, exact terminal output or exact
History slices, Known retry, and sticky Unknown quarantine. Backend diagnostics
publish the native Terminal/History mode, and capability opens only when it
equals the proof retention. The common owner now compacts its repeated
semantic rows after backend preparation, and the actual contract requires the
complete current common Host/Device/Staging memory tuple to match across
Q=5/9/257. The recurrence record recipe does not retain the cold
occurrence/barrier arrays that its fused `vkCmdDispatch` never reads; the
current common tuple is `105920/0/132` for each tested Q. This resident
service-free History shape is not general backing-serviced VSM, total
scratch/storage, or a large-Q product claim.

Timeline-semaphore waits and signals remain queue submission operations, not
shader operations, and this backend has no proved Host/GPU system-scope atomic
polling contract with forward progress. Consequently the separate
service-aware persistent capability keeps `device_generated_recurrence=false`
and `fixed_native_storage=false`, while common keeps
`fixed_common_storage=false`; its O(Q) `VkSubmitInfo` and authored occurrence
streams are not promoted by the service-free fused-Map evidence.

The separate explicit/resident/required DeviceVsm lowering owns Virtual page
coordinates rather than
repeat iterations of one resident state. For Direct pointwise and centered
Clamp Window, common emits a Vulkan compute shader whose W workgroups execute
`page = worker; page < Q; page += W`; Clamp preserves Direct/SharedHalo.
Centered Clip submits the same fixed grid but worker zero owns the whole-run
passes and all other workers return. Clip Sum executes its frozen PrefixDifference
passes in the same shader; Clip Min/Max execute BlockPrefixSuffix and preserve
backward scratch by writing final output in descending coordinate order.
These Clip paths reuse the private whole-run input/output staging and add
neither a native submit nor a Host epoch callback. Pure unsigned U32/U64
Sum/CountNonzero/Min/Max Reduce uses one 256-lane workgroup and one scalar
output. Sum uses a two-word accumulator; CountNonzero adds one bit per admitted
element; Min/Max use unsigned identity trees. For the exact two-stage canonical
total U64 Map-to-Reduce Graph, one 256-lane workgroup traverses Q pages,
evaluates the retained typed Map IR, and applies the sealed Sum, CountNonzero,
Min, or Max reduction. Sum alone carries an additional
high word so public overflow remains checked. Product
submission records and submits one command buffer with one `vkCmdDispatch`; no
epoch callback or Host service operation exists in the request. Actual
public-run Q=5/9/257 evidence requires Q GPU phase counters, exact output, one
queue submit, and one aggregate Authority/Pipeline/backing Final. Graph Sum and
pure Sum Reduce overflow return Known `ReduceSumOverflow` and publish nothing.
The preparation retains a fixed Owner plus parameter/result buffers; whole
input/output backing allocations scale with logical bytes and are not fixed
controller storage.

Known Host-service failure drains the already accepted suffix through
no-write gates. Each suffix wait reports a Known completed,
undispatched/non-writing observation and its exact negative service
acknowledgement advances the same linear frontier. Only the Qth acknowledgement
closes the timeline generation, releases the active claims, and emits one
aggregate Known failure Final with
the first service-failure coordinate, failed-admission suffix, and suppressed
suffix. The dedicated persistent actual contract proves that behavior with a
Q=5 `OneSubmit` failure at coordinate two while retaining one native queue call, zero
epoch native submits/callbacks, Q GPU completions, and no quarantine. A
missing terminal publishes the known prefix plus one submitted-but-unknown
logical batch, emits one Unknown Final, quarantines the adapter and retained
owners, and makes same-owner capability return `DeviceLost`. Product contracts
require exact output, zero H2D/D2H submissions, Known Q5 retry, and Q5
terminal-loss sticky quarantine. If coherent views or whole-Schedule
preparation are unavailable, the fixed-chunk Stream/rolling route remains the
truthful fallback.

The dedicated Q=5 Unknown service-failure case additionally requires one
accepted native queue call, one `UnknownMayWrite` Final, zero mapped GPU-result
reads and backing publication, quarantine of every active role gate and the
adapter, false persistent capability after quarantine, and rejection of both
same-lowering retry and late ready signal. The retained run clears its raw
callback/user and borrowed service-control pointer before the Final callback,
while the sticky claims keep any driver-private blocked command and its native
owners alive. Aggregate `completed_ns` is the elapsed interval from the one
native submit attempt to Final, matching the Metal persistent evidence unit.
Non-admitted Graph/Reduce shapes, Scan, missing coherent views, and a
fixed-local permutation miss retain the rolling route. A service-aware
persistent GPU controller, native
device-generated-command lowering for non-fusible bodies, and the separate
transfer-queue DAG remain future capability.

The persistent dispatcher uses an explicit execution disposition for this
boundary: only a preserved `compute_backend_unsupported` from cold persistent
lowering that is followed by a successful Authority abandon before native
acceptance may select rolling. The common prepared seam carries native failure
reasons through this boundary, so stream/pipeline capacity and memory-budget
failures remain terminal rather than being relabelled as unsupported.
Later unsupported statuses, including service or Final outcomes, are terminal
even when cleanup leaves the Pipeline unpoisoned; status alone does not
authorize a second route.

Vulkan's prepared product query is owned by
`QueryVulkanPreparedPersistentSlidingCapability` in the persistent capability
TU. It translates authenticated prepared owners into native roles and performs
the one native readiness, generated-map, quarantine, adapter, and stream
capacity proof. The Vulkan operations table points directly to that query; the
recurrence operations TU does not maintain a second capability algorithm.

## Sliding status

Persistent Sliding Vulkan keeps one sealed static-key lowering owner and its
bounded per-coordinate authentication storage across matching Known runs. A
private stage/commit/abort rearm validates fresh snapshot-derived roles before
queue acceptance, then replaces only the current token, generation, callback,
and timeline credentials; stale or unknown state is quarantined. The
`BackendChunked` service uses two fixed command/timeline slots and accepts at
most two coordinates per submit: Q=2,3,5 therefore use one, two, and three
native submissions while retaining one logical handoff, zero Host epoch
callbacks, one aggregate Final, and one backing publication. The OneSubmit
path may use transient Q-sized driver descriptions; those opaque allocations
are not relabeled as fixed-native storage. This is not a GPU-owned-recurrence
claim.

## Bounded WindowRing native exception

The finite `WindowRingFused` exception is bounded by checked I32/U32 footprint
and two-bank storage, with a typed proof selecting Resident or fully staged
input/output. The proof accepts bare forms and exact authenticated,
parameter-free `CanonicalTotalU32` Map chains of symmetric depth 1–3 before
and after the Window. Its one `vkQueueSubmit` contains one physical dispatch whose GPU
loop owns the `Q` seed and `Q` compute epochs plus `2Q` internal phases; the
sealed configuration and scratch accesses retain the required compute and
host-read visibility barriers. No epoch queue submit, Host epoch callback, or
transfer submission/bytes is introduced, and common Final owns one
publication/version transition. This bounded Window proof does not widen
Vulkan Persistent into a dynamic Forecast/Promote/Drain/Persist scheduler and
does not include I32 Map fusion.

## Static GraphStageDirect status

The Vulkan residency selection has a private `GraphStageDirect` mode for the
current static U64 Map or Reduce stage of the public Map-to-Reduce Graph.
Preparation admits one sealed direct operation per stage, with one or more
ordered U64 read ports and exactly one U64 write for Map (the existing static
U64 Reduce proof remains available), bound one, no
recurrence/window/transducer/profile/publication, direct dispatches, and
`W<=PreparedPipelineStepCapacity`. The retained prefix, one reusable command
owner per local, suffix, argument owner, and their bounded host/device charge
are prepared cold; each arbitrary Authority-selected local is checked against
its preserved graph identity, ordered data bindings, resident handles, aliases,
and page geometry before it can submit. For Map, D is the metadata binding
access count and only G[0..D) is interpreted as ordered data reads/writes;
GraphControl count/predicate bindings are authenticated separately as canonical
read-only control ports. Controls are not folded into the data-port order.

GraphStageGeneratedIndirect and GraphStageSequence use the canonical private status contract
`generation_stride == 1`. Seeding public generation `g` records a separate
private expected control generation `g+1`; observed control must be present,
valid, and equal to that expected value before an accepted sequence can
rearm. A successful rearm checked-increments the private expected value
exactly once; overflow or mismatch quarantines the sequence. This state is
independent of `preparation_generation`, which identifies cold preparation
rather than a queue-cycle control observation.
This non-wrapping expectation belongs only to those generated graph modes.
Ordinary transactional Pipeline preparation seeds the selected native bank
with `UINT32_MAX` at public generation zero; its stride-two Open advances the
32-bit control modulo `2^32` to generation one. That bootstrap is not a
generated-graph control receipt and must not be rejected by its stricter
terminal-generation guard.

Map and Reduce stages remain separate sealed resources and submissions. This
mode is a per-stage Host submit/wait command-retention proof: it is not a
GPU-owned whole-graph lowering, does not claim one submit for the whole Graph,
and is not the `GeneratedIndirectMap` mode. A pre-submit admission/owner
failure is Known and makes no device write; an accepted submit whose terminal
evidence is invalid is Unknown, quarantines the retained owners, and cannot be
retried. Mixed checked-Map plus Reduce routes and any dynamic, indirect,
recurrence, or non-U64 shape remain rejected or on their existing truthful
route.

The static GraphStageDirect proof coordinator and identity match are exposed by
`node/src/accel/vulkan/kernel/pipeline/residency/graph_direct.{hpp,cpp}`.
`graph_direct/local.hpp` is declarations-only: `graph_direct/bindings.cpp`
owns resident range and canonical identity predicates,
`graph_direct/execution.cpp` owns graph admission roles, aliases, and control
validation, and `graph_direct/map.cpp` plus `reduce.cpp` own the ordered
operation-specific binding proofs. The root `graph_direct.cpp` retains only
aggregate proof construction/matching and dispatches through that seam;
`materialize.cpp` retains selection, materialization, status, and submission
orchestration. No proof or mutable state is copied by the split.

## Pipeline source ownership

The backend entry is an assembly point, not an implementation authority:
`vulkan/ops.cpp` owns only the immutable `BackendOps` table and backend entry;
`ops/buffer.cpp` owns resident buffer I/O adaptation, `ops/control.cpp` owns
runtime observation and diagnostic injection, and `ops/recurrence.cpp` owns
Persistent-sliding and service-free recurrence capability validation. Their
private `ops/internal.hpp` contains declarations only.

Vulkan generated-indirect command recording is split under
`generated_indirect/record/`: `graph.cpp` owns per-stage generated
command validation, temporary command construction, and atomic publication;
`sequence.cpp` owns the two-stage paired command proof, map-state rollback, and
publication. The root `record.cpp` only selects the already-admitted mode and
checks terminal command cardinality; it owns no recording implementation. Its
plan construction and graph active-row projection are owned by
`submit_plan.cpp`/`submit_plan.hpp`, while
`submit_commit.cpp` owns the pre-submit state transition. The shared lifecycle
owner in `lifecycle.cpp` owns quarantine, known-close, and rearm transitions;
`status/terminal.cpp` owns ordered post-submit classification and per-cycle
diagnostics, delegating lifecycle mutations to that owner. Terminal status and
pre-submit commit failures therefore share one lifecycle authority.

Cold pipeline preparation keeps orchestration in `pipeline/prepare.cpp` and
assigns capacity, allocation, immutable description, occurrence dispatch
accounting, publication, command capture, and residency-memory accounting to
the matching `prepare/{capacity,allocation,description,dispatch,publication,
capture,residency}.cpp` leaves. Route dispatch demand is implemented by the
record owner declared in `prepare/record.hpp`.

Strided View normalization is also phase-owned. `kernel/view.cpp` selects and
materializes dense replacements, `kernel/view/source.cpp` owns the sole GLSL
copy recipe and pipeline identity, `kernel/view/commands.cpp` owns descriptor
publication and input/output command encoding, and
`kernel/view/accounting.cpp` owns retained-memory, traffic, and dispatch
projection. The private `view/internal.hpp` carries only the shared push-row
layout and fixed Vulkan limits; it is not another source or execution owner.

Pipeline publication is phase-owned as well: `kernel/publish/source.cpp` owns
the sole GLSL publication recipe and pipeline identity, `kernel/publish.cpp`
owns resident-route resolution and descriptor materialization,
`kernel/publish/encode.cpp` owns terminal, canonicalization, and Window command
encoding, and `kernel/publish/accounting.cpp` owns retained Host-byte
projection. The private `publish/internal.hpp` carries declarations/constants
only.

The Vulkan Map-recurrence pipeline has one owner per phase:
`pipeline/recurrence/validation.cpp` owns runtime/prepared-plan and structural
equivalence proofs, `template.cpp` owns collision-safe variant identity, cache
validation, and immutable template acquisition, and `prepare.cpp` owns route
resource binding and public recurrence/transducer preparation. These leaves
share the existing common recurrence plan and preserve its exact source,
descriptor order, generation-stride, and fail-closed status reasons; no second
template or terminal-state authority is introduced.

Vulkan has a source-private `GeneratedIndirectMap` admission and ownership
implementation. It is limited to fixed, non-history Map records with one
generated-check resource, rejects dynamic controls, and authenticates the
control row, run claim, full/tail role lifetime, and Unknown quarantine. The
ordinary Map layout remains four-binding/64-byte and the checked variant is
six-binding/128-byte; the mode retains O(Q) HostCoherent intermediates and
keeps `device_generated_recurrence=false` and `fixed_native_storage=false`.

The Vulkan cache/source contract is physically split under
`tests/contract/compute/vulkan/cache/`: `support.hpp` is the sole fake-handle
support owner; `descriptor.cpp` covers descriptor-range/storage-page bounds;
`map_bias.cpp` covers canonical source bias specialization; `collective.cpp`
covers collective source identity and pipeline-key/cache-hit accounting;
`shader.cpp` owns validated-SPIR-V cache bounds, concurrency, and tool/validator
probes; and `dispatcher.cpp` is the only case entry and ordering coordinator.
All embedded GLSL remains a host-side string consumed by the Vulkan compiler
probe; no test leaf is compiled as a shader translation unit.

This private lowering has no natural callback-backed public product evidence.
The public Gather semantics are whole-buffer, while the fixed unary page-frame
owner cannot safely reinterpret them as page-local indices; the removed test
therefore must not be counted as an E2E success, mixed-route proof, or Known
failure proof. A future product requires one of: an explicit PageLocalIndex
proof, bounded authenticated multi-page Forecast/Promote/page-table ownership,
or a whole-resident source. Until then the public route remains fallback/legacy
and the generated mode is not a public Persistent claim.

The separate `GraphStageGeneratedIndirect` owner is a narrower Host Graph
stage contract, not an expansion of that Persistent mode. It admits a sealed
U64 Map at each graph occurrence, with one through seven ordered resident data
reads and one resident write (at most eight ordered data bindings), a nonzero
authenticated Graph identity, a resident count and optional predicate source,
and bounded selected locals. The count and predicate are separate canonical
read-only control bindings and are not included in that data-binding limit.
The record has `E` entries and the common selection has `W` locals; admission
requires `E == W <= PreparedPipelineStepCapacity`, with template, occurrence,
and local indices equal. Each fixed `GraphFrame[local]` retains its own
immutable proof, resident binding/alias and handle tuple, count/predicate
descriptor row, status/generation, and exactly one reusable command recorded
for `{first=local,count=1}`. Distinct rows need only preserve the same
semantic graph/kernel/operator identity; they must not share execution or
resident-handle identity. The row identity is one-frame: `stride=1`,
`slot=0`, `turn=local`, and `coordinate=local`; its control generation is
`first_control + local * control_stride`, while its nonzero descriptor
generation is frame-authenticated (`local + 1`) with no Persistent descriptor
stride. The common `residency_pipeline_locals` selection
remains authoritative for stage, bank, page, token, and local identity.
Reduce, recurrence, Window, transducer, profile, check, fused, mixed, and
malformed routes remain rejected or on their existing truthful route.

The cold GraphGenerated owner retains `W` unframed frame commands plus a
zero-row open prefix and zero-row close suffix; its status command count stays
`W`. Each Authority lease span `L[0:A]`, `1 <= A <= W`, is submitted in that
order as `[prefix, frame[L0..A-1], suffix]`, so the one queue submission has
`A+2` command buffers and `A` active frame rows. Inactive frames remain cold
and untouched. This is still a per-stage Host submit/wait envelope, not
whole-graph GPU ownership.

The descriptor binds the resolved resident count and predicate buffers
themselves, never generic `control_args`. Host-write-to-compute visibility
precedes canonical control and each private Graph gate; every selected frame
is checked and its status is aggregated at terminal. Predicate mismatch is a
Known successful no-op, count overflow is a Known failure with zero indirect
words, and proof/identity/status/visibility/owner/entry/generation mismatch
is `UnknownMayWrite` with every affected frame and the adapter quarantined.
Rows remain live through terminal. This remains a per-stage Host submit/wait
route with common Graph Authority terminal and publication ownership;
whole-graph GPU ownership is not claimed.

The GraphStageGeneratedIndirect evidence separates three private lifetimes:
the cold graph generation and its `record_generation`, the queue-cycle
`submit_seq`/`done_seq`, and cumulative frame-row totals for submitted, done,
accepted, known-failure, and unknown rows. A successful accepted cycle is the
only transition back to `Recorded`; rearm clears current submitted and
terminal counters while retaining the cold record count and monotonic
sequence/totals. A `build_plan` rejection, or a fence/queue rejection before
an accepted submit, is Known/no-write and leaves selected rows and counters
unchanged; cold owner construction has its separate admission/record
lifecycle. A canonical per-frame failure closes as Known, records its first
reason, and cannot rearm. Any post-submit evidence gap, sequence mismatch, or
aggregate invariant failure classifies every row in that plan as
`UnknownMayWrite`, preserves the reason, and quarantines the rows and adapter.
Frame generation is checked against its local row identity before record,
submit, and terminal rearm; these are per-stage Host submit/wait and frame-row
laws, not whole-graph GPU ownership.

The proof implementation follows those boundaries physically:
`generated_indirect/admission/entry.cpp` owns GeneratedIndirect entry and
graph eligibility, `sequence.cpp` owns the two-stage sequence eligibility,
`match.{hpp,cpp}` owns the shared storage/semantic/frame comparison and replay
checks, and `graph.cpp` retains only candidate precedence and mode dispatch.
These leaves are stateless proof code; lifecycle, status, quarantine, and
recording remain owned by their existing generated-indirect modules.
Generated-indirect recording is physically split under
`generated_indirect/record/`: `graph.cpp` owns per-stage generated command
validation, temporary command construction, and atomic publication;
`sequence.cpp` owns the two-stage paired command proof, map-state rollback, and
publication. The root `record.cpp` only selects the already-admitted mode and
checks its terminal command cardinality; it owns no recording implementation.

The virtual Reduce sequence contract keeps authored two-stage construction,
execution, and its arithmetic oracle in `virtual/product/reduce/sequence.cpp`.
Vulkan-only admission, native-status, and generated-indirect lifecycle
diagnostics are read-only projections owned by
`virtual/product/reduce/sequence/diagnostics.cpp`; its `local.hpp` is a
declarations-only seam. The split introduces no second route observation or
residency state.

Generated-indirect status is physically split under
`generated_indirect/status/`: `diagnostic.cpp` owns the mapped gate result
authentication and canonical Known-reason projection, `terminal.cpp` owns the
ordered Graph/Sequence post-submit classification and accepted/known/unknown
totals before delegating lifecycle transitions, and `projection.cpp` owns
read-only shape/frame/projection queries plus public prepared-pipeline
inspection. `generated_indirect/lifecycle.cpp` remains the sole owner of
quarantine, known-close, and rearm mutations; no status leaf introduces a
second lifecycle or diagnostic counter state.

Generated-indirect resource construction and teardown are split by resource
authority under `generated_indirect/resources/`: `map.cpp` owns Map access,
ready state, and the generated control gate; `graph.cpp` owns Graph frame proof,
gate seeding, preparation, and destruction; `sequence.cpp` owns the two-stage
sequence gate seed, preparation, and destruction; and `lifecycle.cpp` owns
public mode dispatch plus generated-role teardown. The root
`generated_indirect/lifecycle.cpp` remains the sole terminal/quarantine/rearm
owner. These resource leaves preserve the existing lease scope, generation
checks, fail/quarantine transitions, and SDK conditional boundary without a
second mutable lifecycle authority.

The private Metal/Vulkan `GraphResident` mode is admitted for a fully resident
or fully all-staged, bounded U64 Graph with `input_count>=2` and a nonterminal
stage containing at least two distinct Internal/Intermediate/Transient reads
plus one Internal/Intermediate/Transient write. Its native owner table binds each
internal physical class and bank; external endpoints use separate
authenticated endpoint slots. For the branch/fan-in contract, the seven physical classes are three
internal banked owners plus four external endpoint classes (three inputs and
one output), not seven owner bindings. The shader scans the authenticated
ready wavefront in deterministic order. Internal X/W alias reuse waits for
`DispatchComplete`; only a planner-marked `ExternalOutput` predecessor uses
`ReleaseComplete`. One queue submit records one physical GraphResident GPU
controller dispatch; that controller completes the fixed 15 `(batch,stage)` steps,
and one logical payload dispatch produces one aggregate
terminal/publication; in the resident form Host epoch service,
callbacks, and runtime backing I/O remain zero. The product shape is
`A→X, B→Y, X,Y→Z, C→W, Z,W→O`, with its tail, warm lifecycle, output, and
version invariants. The bounded contract is `S<=8`, `R<=9`, `P<=16`, `O<=9`,
`Q>0`, `C>0`, `C<=Q`, `B=ceil(Q/C)`, and `S*B<64`; all-staged evidence
covers the existing S=5,Q=5,C=2,B=3 case and S=5,Q=6,C=2,B=3, each with
15 GPU wavefront steps, one submit, one physical controller dispatch, one
logical payload dispatch, one aggregate Final/publication, and no Host epoch
service or runtime backing I/O. GraphPointwise remains a separate
route. Metal uses its independent
private native implementation; strict native allocation-free storage and
dynamic nonresident GPU-native I/O remain unclaimed. All-staged endpoints are
authenticated against fallback buffer references, native handles, exact byte
extents, and proof rows before mutation; the complete input set is staged
before the native recurrence and output after Final. The GraphResident compute
submit records one physical controller dispatch; the controller executes its
fixed 15 `(batch,stage)` steps and the logical payload dispatch remains one.
Multi-input staged admission uses the public `VirtualBackingReadCohort` member side:
each input returns the same shared `VirtualReadCohort` provider with a valid
nonzero ID and lane limit 1..2. The provider synchronously joins canonical
descriptors, retains no descriptor or callback, and is reauthenticated before
and after the join. Results use `joined=true` only after joining; successful
results report exact `completed_bytes` with `UINT64_MAX` failure sentinels,
while joined known failures report their first member/page and completed
prefix. Single-input staged endpoints use the existing synchronous scalar
`read_pages` contract and do not require a cohort. For multi-input staged
endpoints, missing capability declines before preparation; post-preparation
identity uncertainty is terminal. Resident endpoints bypass the provider.
Generic callback-backed cohorts without this side contract remain outside this
proof. This compute
submit is distinct from any H2D/D2H staging submissions.

GraphResident cold preparation also records one immutable secondary dispatch
with one stable collective descriptor-set lease. The lease and secondary stay
owned through warm rearm, quarantine, and owner destruction only while the
proof digest, endpoint references/handles, pipeline, and layout identity all
match. Every execution records a fresh primary, executes the secondary, adds
the existing compute-to-Host barrier, submits, waits, and runs its own Final;
the primary, queue submit, wait, and terminal/publication are never cached.
DeviceVsm preparation is split by ownership under
`node/src/accel/vulkan/kernel/pipeline/residency/device_vsm/`: `prepare/owner.cpp`
owns Owner lifetime and aliasing, `prepare/validation.cpp` owns proof-derived
shape and resident admission, `prepare/graph.cpp` owns graph proof-table,
endpoint bindings, and immutable secondary recording, and `prepare.cpp` owns
descriptor/storage allocation and capability projection. `prepare/local.hpp`
is a declarations-only seam; `internal.hpp` remains the sole Owner/model,
descriptor-cardinality, and binding-identity authority.
All Vulkan DeviceVsm descriptor cardinalities are checked by the single
backend-private formula in `device_vsm/internal.hpp`; cold allocation and warm
descriptor publication consume the same value.

This does not add dynamic
Forecast/Promote/Drain/Persist continuation.

Resident GraphResident binds its proof-owned PageMap as the exact 3296-byte
storage row at descriptor binding 0 when `param_bytes == 0`. The shader uses
the sealed external-resource row, batch base `F`, active tail `q`, and
Begin/End source origin to resolve a source page while preserving the ordinal
staging read once per page. Host preparation, execution, and warm rearm compare
all serialized words and capacity before binding; a stale or malformed map is
rejected before mutation. The map is specific to the resource-wide `S=1`
resident GraphResident route. Nonresident GraphPointwise may use the same
proof-owned image when `param_bytes == 0`: binding 0 changes to the map while
the descriptor count and push-constant dispatch row remain unchanged. Its
source decomposes each target gid into batch base, target-local page, and
intra-page element; an absent row uses the ordinal endpoint, while a present
row applies the sealed Begin/End source. Cold upload/flush and warm full-word
rearm authentication precede mutation. Partial-byte, multi-source, and
internal/output/transient remaps retain their existing routes.

## Fixed GraphStageSequence

`GraphStageSequence` is a fixed private S=2 per-stage Host route. Each selected
local owns one command containing two authenticated Map rows: stage 0 is
1R→2W and stage 1 is 2R→1W. The stage-0 writes must be distinct owners and
must alias the stage-1 reads by complete physical storage identity (id, extent,
offset, element geometry, count, and handle, independent of usage); each local
retains both rows, descriptors, gates, and intermediate pins through terminal.
The separate per-step Read/Write roles and graph-alias representatives remain
explicitly validated and are not inferred from that physical identity.

The cold owner has immutable capacity `W=2`; each Authority submission carries
an ordered unique lease span `L[0:A]`, with `1 <= A <= W`. One queue submit
records the existing zero-row open prefix, the `A` frame commands in lease
order, and the zero-row close suffix: exactly `A+2` command buffers. The frame
commands remain open=false/close=false, and the envelope owns no rows. The
terminal result still has `2A` sequence rows (four only when `A=W=2`), while
inactive frames remain Recorded and are not read or mutated. Accepted replay
rearms only the active span. An all-known failure closes the owner, while any
mixed, stale, or ambiguous active row quarantines the affected span as
`UnknownMayWrite`. Cold generation, queue-cycle sequence, and cumulative
active-row totals remain separate identities; the route is per-stage Host
submit/wait, not whole-graph GPU ownership and not arbitrary fusion.
