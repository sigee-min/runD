# Metal Residency

This page owns Metal command construction, terminal quarantine, coherent
views, the bounded native window, and whole-Schedule lowering. Backend-neutral
admission belongs to [Common](./common.md); cache policy belongs to
[State](../state.md).

Metal is the current native arbitrary-local implementation. Cold preparation
materializes a command allocator/buffer, shared event/listener, watchdog,
residency set, and exact selected ICB closure within the existing Device
Pipeline budget.

Generic Pipeline residency materialization is physically split under
`node/src/accel/metal/kernel/pipeline/residency/materialize/`:
`stage.mm` owns candidate allocation, watchdog/resource construction, and
retained-capacity accounting; `commit.mm` transfers one candidate into the
retained `MetalSequence`; `cleanup.mm` owns candidate and sequence teardown;
and `query.mm` owns support/readiness observation. `internal.hpp` carries only
the candidate layout shared by stage and commit, so `MetalSequence` remains
the sole retained mutable authority.

## Pipeline source ownership

The backend entry is an assembly point, not an implementation authority:
`metal/ops.cpp` owns only the immutable `BackendOps` table and `MetalEntry`;
`ops/buffer.cpp` owns resident-buffer I/O and coherent Host-view adaptation,
`ops/control.cpp` owns runtime observation and fault-injection projection, and
`ops/recurrence.cpp` owns the service-free direct proof projection. The private
`ops/internal.hpp` seam contains declarations only. This split preserves the
table's field order and leaves the existing kernel, persistent-sliding, and Map
owners as the sole authorities for their retained state and native behavior.

DeviceVSM Metal preparation is physically owned by
`residency/device_vsm/prepare/entry.mm`: it preserves the ordered admission
and final capability handoff. `prepare/validation.mm` owns proof-shape and
resident binding admission, `prepare/resources.mm` owns native pipeline,
buffer, table, and ring allocation, `prepare/record.mm` owns GraphResident
ICB recording, and `prepare/owner.mm` owns the retained owner lifecycle.
`device_vsm/internal.hpp` remains the single retained-state model; the
`prepare/internal.hpp` seam carries only bounded per-call derived facts.
Map preparation follows the same compiled ownership boundary under
`metal/runtime/map/prepare/`: `binding.mm` owns immutable template/binding
identity, `template.mm` owns specialization and native Pipeline freezing,
`resources.mm` owns per-route native resources, and `route.mm` owns ordinary
and already-proved route admission. The adjacent `prepare.mm` only composes a
template plus route; Map implementation is no longer textually included
through `map.mm`.
The exposed submission charge is:

```text
M_submit = submission_allocator.allocatedSize
         + sum(window_allocator[0..1].allocatedSize)
         + residency_set.allocatedSize
```

Metal also has the private `GraphResident` topology with resident and
all-staged endpoint modes; admission requires `input_count>=2` and a
nonterminal stage with at least two distinct Internal/Intermediate/Transient
reads plus one Internal/Intermediate/Transient write. Its
bounded dynamic geometry uses three internal banked owners and four external
endpoint classes (`S<=8`, `R<=9`, `P<=16`, `O<=9`), with one native submit, one
physical GPU controller dispatch, one logical payload dispatch, and one
aggregate Final/publication per execution. The resident form performs no Host
epoch service, callback, or runtime backing I/O; all-staged endpoints
authenticate fallback buffer references, native handles, exact byte extents,
and proof rows before mutation; inputs are staged before recurrence and output
after Final. The product evidence covers the original S=5,Q=5,C=2 case and an
all-staged S=5,Q=6,C=2,B=3 case; both retain 15 sealed wavefront steps. The
one GraphResident compute submit is distinct from any H2D/D2H staging
submissions. Multi-input staged admission uses the public
`VirtualBackingReadCohort` member side: every input must return the same
shared `VirtualReadCohort` provider, with a valid nonzero ID and lane limit
1..2. The provider joins the canonical input descriptors synchronously,
retains no descriptor or callback, and is reauthenticated before and after
the join. `joined=true` marks a completed join; success reports exact
`completed_bytes` and `UINT64_MAX` failure sentinels, while a joined known
failure reports its first member/page and completed prefix. Single-input staged
endpoints use the existing synchronous scalar `read_pages` contract and do not
require a cohort. For multi-input staged endpoints, missing capability declines
to Host before preparation; post-preparation identity uncertainty is terminal.
Resident endpoints bypass this provider. Generic callback-backed
cohorts without these guarantees remain outside this proof.
Metal one-shot native command-buffer storage remains opaque, so strict native
allocation-free evidence is not claimed.

GraphResident cold preparation records one dispatch into a retained private
ICB using the existing `MetalIcbChunk`/`AllocateMetalPipelineIcb` contract and
keeps its pipeline and immutable buffer bindings with the owner. Each run
creates a fresh one-shot command buffer, executes that ICB, commits and waits,
then performs the normal result visibility, Final, and publication path. The
ICB is not a cached submit or terminal; in-flight, device-loss, and
`UnknownMayWrite` owners are not rearmed.

For a mapped resident GraphResident, binding 0 is the exact 824-word /
3296-byte proof-owned PageMap because the GraphResident plan has zero parameter
bytes. The MSL source resolves each external input's batch-local page using
the sealed `K`, tail `q`, and Begin/End origin; input staging remains ordinal
and occurs once per page. Preparation, execution, and warm rearm authenticate
the complete map image and capacity before mutation. This is only the
resource-wide `S=1` full-page map for the resident route. Nonresident
GraphPointwise also accepts the proof-owned 824-word image at binding 0 when
parameter bytes are zero; config, ring, and result bindings remain unchanged,
and MSL resolves batch base, target-local page, tail, and Begin/End source
before the ordinal alias read. Partial-byte, multi-source,
internal/output/transient, or other resident DeviceVSM remaps remain outside
the Metal route.

Candidate construction and checked plan arithmetic precede admission commit;
failure publishes no owner. Admission uses the actual candidate's retained
bytes, not a measurement from another preparation. The adapter-owned queue is
shared, while each bank has an independent exact event generation. Submission is begin, residency,
selected ranges, end, commit, queue signal, event callback, reset. Optional
Metal commit feedback is not a correctness fence.

## Bounded WindowRing preparation

The nonresident WindowRing candidate performs a checked byte preflight before
constructing its native owner or resizing its `Coordinate` vector:
`coordinate_count <= SIZE_MAX / sizeof(Coordinate)`. Overflow returns the
explicit `compute_pipeline_capacity` reason and leaves no partial owner;
invalid requests retain the ordinary invalid-request result. This bound is
the Metal-specific exception for the two-scratch-slot, one-submit
`WindowRingFused` contract. The typed proof covers bare centered I32/U32
Window semantics and exact authenticated, parameter-free `CanonicalTotalU32`
Map chains of symmetric depth 1–3 before and after the Window on Resident or
fully pre-staged endpoints. One physical dispatch/native submit owns Q seed and Q compute epochs plus 2Q internal
phases; Host epoch service/callbacks and transfer submissions/bytes remain
zero, and common Final owns the single publication/version transition. This
does not claim strict allocation-free native storage or I32 Map fusion.

Each submit arms one allocation-free `TerminalCell` whose packed atomic binds
the exact generation and Idle/Armed/Known/UnknownMayWrite state. The shared
event callback can resolve only its generation; a late callback cannot
terminalize a later submit and two terminal producers cannot both win. The
watchdog also publishes the generation's monotonic deadline. A queued event
from a prior timer observes zero or the later deadline and returns or rearms;
it cannot expire a new generation early. The normal deadline is 30 seconds. If
it expires before the exact event, the Host waiter receives
`UnknownMayWrite`, not fabricated completion.

`UnknownMayWrite` permanently quarantines the adapter's VSM capability and the
complete self-retained prepared owner tree: Pipeline lifetime, Device, Pool,
frames, command owner, and allocator remain alive because the lost generation
may still write them. The allocator is not reset and no output or backing
publication follows. Same-owner retry, peer runs, and new preparation on that
Device return `DeviceLost`; peer output callbacks stay unchanged and newly
prepared backings receive no callbacks. The lost run may already have consumed
input callbacks before its native terminal became unknowable, but its output
callback count does not advance. The
injected 10-millisecond terminal-loss deadline proves this bounded quarantine
contract, not the behavior or classification of every real driver failure.
An adapter-local terminal gate linearizes accepted native commit with this
sticky quarantine: a submit is either committed and counted before quarantine,
or observes `DeviceLost` without commit. The accepted-submit producer
increments one atomic active-command count and peak. Known terminal decrements
active; `UnknownMayWrite` deliberately leaves its command active/unknown. Each
receipt seals the active depth observed at its own accepted submit, so the
public `command_inflight_peak` is actual two-bank concurrency evidence rather
than a planned bank count.

The retained ICB/MTL4 path supplies raw selected-local evidence to the first
Q=1 Compute execution consumer. It reports the actual accepted native command
count, observed inflight peak, GPU-written control, and exact terminal
certainty; it does not self-assert Pipeline success or a Host-service terminal.
Compute publishes NativeEvidence only after the common Pipeline finisher
accepts all of that evidence. One cold real submission may initialize backend
state. The following 60 Q=1 submissions reuse the prepared owner and fixed
caller Control with zero tracked C++ allocation. Tampered control and a
synchronous submit rejection leave the Pipeline generation unchanged. An
injected missing terminal returns one submitted Dispatch, `UnknownMayWrite`,
zero completion, and exact may-write failure evidence.

Metal also implements the raw bounded window for `1<=Q<=4` Direct pointwise
selections. One cold owner retains two command slots per bank owner plus shared
ready/done events and exact-generation watchdog cells. One public handoff
copies the fixed request and queues, for every batch, `wait ready(e)`, one MTL4
command commit, and `signal done(e)` before returning. The main run thread does
not submit or wait per epoch. Metal event/watchdog listeners never execute a
Host backing callback: each completed contiguous Release bundle is transferred
through `dispatch_async_f` to one cold-owned serial residency-service queue.
That fixed lane performs common Input/Output service and signals readiness only
after the prior-bank terminal laws are satisfied. Its callback packets are
cold-owned fixed command slots, so the transfer allocates no C++ warm owner.
Consequently Metal reports `native_batches=Q`, `queue_calls=Q`, and
`public_handoffs=1`; queue-call count is backend evidence, not a restatement of
the public handoff count.

The raw adapter accepts only guarded, non-aggregate, zero-ResidentState
command streams. Each bank must retain one prepared stream for the complete
window and adjacent banks may not alias the same no-write gate. If Host service
or common Pipeline admission fails after the handoff, the coordinator signals
the already queued suffix with a failed `admission`. The command prefix reads
the cold-owned gate, performs no selected payload write, and still produces a
Known dispatched/completed Release with `may_write=false`; this drains the
queue without inventing a no-dispatch terminal. Unknown terminal loss alone
keeps the sticky adapter quarantine. Exact Release order and one Final are
stored in fixed four-entry arrays, and the warm window path allocates no C++
owner storage.

The implementation is physically split by that protocol boundary:
`node/src/accel/metal/kernel/pipeline/residency/encode.{hpp,mm}` is the single
ICB/range/selection encoder (including the Sliding and Schedule encode
adapters); `submit.mm` owns only the single warm prepare/generation/terminal
submit; and `window_submit/{prepare,submit,signal,abort}.mm` own bounded
Window command preparation/cancel, Submit, Signal, and Abort respectively.
The `window_submit.hpp` declarations remain the public boundary. The pipeline
run leaves are direct owners:
`run/entry.mm` owns sequence validation, warm command encoding, generation
seeding, and public submission; `run/terminal.mm` owns sequence
control/profile observation and the single sequence/residency terminal
projection; and `run/window.mm` owns ordered Window release/final delivery.
No aggregate `run.mm` facade remains.

Pipeline preparation finalization is likewise physically direct-owned under
`node/src/accel/metal/kernel/pipeline/prepare/finalize/`:
`capture.mm` owns final control/publication capture and exact dispatch
membership; `projection/identity.mm` owns pointer-identity dedupe and initial
capacity projection; `projection/windows.mm` owns direct/spatial proof
materialization; `projection/ranges.mm` owns residency ranges and route
selectability; and `projection/entry.mm` sequences those projection phases.
`native.mm` owns parameter-buffer and calibrated ICB allocation/encoding;
`owner.mm` owns retained/cold-workspace accounting and the final prepared
handoff; and `entry.mm` sequences the complete finalization. `internal.hpp`
carries the transient projection and cross-leaf declarations; it is not a
second Pipeline owner. The former aggregate finalization source is deleted.

The whole Schedule has the same direct-owner rule:
`residency/schedule/entry.mm` owns role validation, cold command preparation,
and the one public queueing handoff; `schedule/signal.mm` owns gate admission
and ready-event signaling; `schedule/terminal.mm` owns timeout, native terminal
classification, ordered Release delivery, and the one Final; and
`schedule/cleanup.mm` owns allocator teardown and exact-generation abort. The
private `schedule/internal.hpp` contains only the one shared owner/command
layout and cross-leaf declarations, so `MetalResidencyScheduleOwner` remains
the sole mutable schedule authority.

The Metal Persistent product has a separate spatial-Window admission. It
derives a backend-private proof from the existing bound execution steps:
exactly one `operation::Window` in the scheduled `Map -> Window -> Map` order,
SharedHalo centered Clamp stride-one geometry, descriptor-sized bounds, and no temporal
state, transducer, history, aggregate, or dynamic publication. The prepared
owner retains one immutable complete ICB range per local and authenticates the
proof, owner, generation, and selected local before executing that range. The
temporal `MetalWindow`/`ResidentState` recurrence path is distinct, remains a
ServiceFree Direct/history concern, and is rejected by Persistent Sliding. The
actual Q=2, N=48, F=16, R=2,
P=12 Clamp contract proves one Persistent batch, one native submit/queue call,
192 logical backing-read bytes, 256 expanded-frame H2D bytes, no duplicate
canonical-overlap read, exact output, and one Final/publication. This remains
O(Q) preencoded HostCoherent execution with `device_generated_recurrence` and
`fixed_native_storage` false; unsupported or temporal Window forms keep the
existing rolling/fallback routes. No Vulkan claim is made here.

The stateless spatial-Window proof model is owned by
`node/src/accel/metal/kernel/pipeline/prepare/spatial_window/proof.hpp`;
its exact equivalence, key, range, descriptor, and plan algorithms are owned by
the facets under `spatial_window/equality/`: `base.hpp` owns physical refs and
canonical key/metadata comparisons, `artifact.hpp` owns parsed IR and lowering
artifact equality, `range.hpp` owns range/window plan equality, and
`execution.hpp` owns bound-step, admission, and complete execution equality.
Each admitted proof cold-allocates exactly its declared local rows; an
unadmitted proof retains no row allocation. The prepared sequence moves that
single owner into place and never grows it during execution. The backend host
reservation bounds the rows by the canonical physical step-occurrence count:
every admitted local supplies its Map, Window, and Map steps, so this count
bounds the number of retained local rows. Checked multiplication reserves that
bound before materialization; retained-memory observation counts the actual
row capacity exactly once. The global Pipeline step limit remains an admission
limit, not an eagerly allocated array in every ordinary Pipeline.

The adjacent `equality.hpp` is only their stable include surface. The compiled
proof owners under `spatial_window/proof/` are direct and stateless:
`diagnostics.mm` owns first-false diagnostics, `geometry.mm` owns execution,
topology, range, and SharedHalo identity capture, `bindings.mm` owns resident
edge authentication and alias proof, `admission.mm` sequences the immutable
proof construction, and `validation.mm` owns the final prepared-sequence
cross-check. No owner retains a second execution or residency state.
The focused Metal execution contract uses the same cut:
`pipeline/metal/residency/execution/single.cpp` owns Q=1 tamper rejection,
synchronous rejection, cold initialization, and allocation-free warm evidence;
`execution.cpp` owns the Q=4 Window/stream readiness, release, abort, and
quarantine protocol. `local.hpp` exposes only the bounded test types and
function declarations, never either execution body.
Metal admission is physically split under
`node/src/accel/metal/kernel/pipeline/prepare/admit/`: `validation.mm` owns
entry/context admission, `aggregate.mm` owns the nested-aggregate route,
`resources.mm` owns recurrence/window proof and resident-resource projection,
`recurrence.mm` owns the cached recurrence template/source route,
`routes.mm` owns prepared sequence assembly and recurrence/transducer routes,
and `entry.mm` sequences those phases as the sole `MetalPipelineBuild::Admit`
owner. `internal.hpp` carries declarations only; no leaf owns a second
pipeline state or terminal authority.

Metal Pipeline description is physically split under
`node/src/accel/metal/kernel/pipeline/prepare/describe/`:
`aggregate.mm` owns the direct nested-aggregate status projection;
`capacity.mm` owns template-step and status/telemetry reservation;
`templates.mm` owns per-template context, telemetry, and status description;
`occurrences.mm` owns physical dispatch/reset and compact-step evidence;
`profile.mm` owns transducer occurrence compensation; and `packing.mm` owns
private/public raw-offset packing and status-entry materialization. The root
`describe.mm` only sequences these phases, while `internal.hpp` carries the
derived capacity record and declarations. Every phase mutates the same cold
`MetalPipelineBuild`; no source, ABI, lock, error, or retained-state authority
is mirrored by a leaf.

The ordered Metal program encoder is owned under
`node/src/accel/metal/kernel/pipeline/prepare/program/`: `recurrence.mm`
encodes the cached recurrence variant, `entry_prepare.mm` validates one
occurrence and captures its borrowed slices, `window.mm` records nested-window
control transitions, `body.mm` emits the ordinary or transduced program body,
`status.mm` imports and reduces status, and `publication.mm` emits nested
publication/canonicalization. `entry.mm` retains only the ordered
`EncodePrograms` composition and profiling boundary. Every phase mutates the
same cold `MetalPipelineBuild` capture; no template, source, cache, or command
state is mirrored by a leaf.

An accepted window also has one exact-generation emergency abort seam. It is
not cancellation. If both Execute admission and its ordinary Suppress retry
fail, common authenticates the active Plan/token/generation and asks the raw
owner to open every still-unsignalled gate in no-write mode. Metal marks those
commands conservatively `UnknownMayWrite`, opens all ready values, drains to
one Final, and permanently quarantines the adapter; it never fabricates a
success or an undispatched suffix. Unknown Final converts the caller-owned raw
control pointer into a strong prepared-submission quarantine, and raw retained
requests clear callback/user pointers before the caller control can die.

Before Authority admission, the common mutation-free readiness query validates
one through four prepared owners against the exact Context, one BackendOps
table, raw submit/signal/abort hooks, idle common submissions, and each backend's
cold native readiness. It takes no claim and allocates no owner. A false result
is therefore a pre-admission capability decision; a raw submit rejection after
Authority begins is a run failure and cannot be relabeled fallback.

For coherent Direct pointwise Q>4, production first attempts one whole
Schedule. Cold preparation creates Q distinct allocator/command pairs because
the admitted MTL4 API rejects concurrent reuse of one command buffer. It
freezes the remaining Device Pipeline budget, commits the exact tracked
pending-command charge, and does so before Authority admission. One public
handoff queues Q `wait/commit/signal` operations; no later Host callback
submits a Dispatch. Independent Metal event listeners only mark completion.
One Schedule-owned serial delivery lane emits the global contiguous Release
prefix, preventing cross-bank listener reorder from corrupting Authority.

A Known Host-service failure opens every accepted suffix gate in no-write mode
and permits immediate same-owner retry. A lost terminal emits one Unknown
Final, self-retains the possibly writable owner, and makes same/peer/new-owner
capability checks return `DeviceLost`. The public contracts execute Q5/Q9,
verify `handoffs/batches/queue_calls=1/Q/Q`, zero physical transfers, exact
output, Known failure retry, and terminal-loss quarantine. A
`retained_bytes-1` contract proves a schedule budget miss selects the existing
four-epoch Stream before Authority mutation; refunding that reservation admits
the same Schedule. Non-admitted Graph/Reduce shapes, Scan, missing coherent
views, and a warm fixed-local permutation miss retain their documented
fallbacks. This is not
persistent device scheduling or an asymptotic Metal queue-operation reduction.

After a Known exact terminal, DeviceOps may authenticate a stable coherent
read-only Host view of the complete output Buffer. Direct non-Scan,
non-reduction execution then keeps the Device page under its Writeback token,
hashes/writes backing from the view, and reports no D2H. This is capability
driven; Compute contains no Metal branch. If capability is absent or invalid,
or for Scan and Graph reduction, D2H targets registered Host output and
Authority atomically migrates dirty ownership. Telemetry counts only the path
that happened.

Before execution, DeviceOps may independently authenticate a stable coherent
writable Host view of the complete private Input Buffer. Direct non-Scan,
non-reduction execution uses it only when the frozen Host and Device frame
capacities are equal. The backing worker first acquires the compound
Input/Output transform lease, then fills the exact Device Input bindings
through that view; the execution consumer resumes the same live token. There
is no physical upload, H2D byte count, or H2D interval. This is a CPU write into
coherent Shared storage, not GPU DMA and not a native transfer node.

View denial is a capability miss and falls back to the ordinary private upload
on the same prepared owner. It does not consume or mask injected transfer
faults. If Host capacity exceeds Device capacity, the existing reusable Host
tier remains authoritative for supply so future-use hits are not discarded for
copy elision. Scan, reduction, and Graph keep their existing supply route. The
coherent equal-capacity mode currently retains the already planned Host frames;
it removes the physical copy, not that cold memory commitment.

Native GPU timestamps and deterministic OS queue-error classification remain
unavailable on this selected path. A normally terminal DeviceLost is projected
after exact retirement; unknown-terminal loss follows the quarantine contract
above. Neither is native driver-failure evidence.

## Sliding status

The Persistent Sliding Metal owner is cold-shaped once and may be rebound for
a Known quiescent run only when the sealed static key still matches. Rebinding
stages fresh snapshot-derived roles and changes only the per-run Plan token,
generation, callback, and service identities; a shape change or unknown
terminal retains the owner in quarantine. Shared-event values are monotonic
across rearmed command buffers. The retained lowering and coordinate table
are runD-owned; Metal one-shot command-buffer objects remain opaque native
state and are not counted as fixed storage, so strict native allocation-free
warm execution is not claimed. This warm contract does not claim GPU-owned
recurrence.

Persistent Sliding cold preparation is physically split under
`node/src/accel/metal/kernel/pipeline/residency/persistent/prepare/`:
`validation.mm` is only the issue-to-`AccelCheck` coordinator;
`validation/{issue,role,request,identity,capability}.mm` respectively own the
issue key, native-role/sequence proof, request/range proof, prepared-request
identity, and capability projection. `rearm.mm` owns coordinate refresh and
the pending/rollback ticket transaction; and `entry.mm` owns public owner
allocation, shared-event/command construction, encoding, and capability
assembly. The existing `internal.hpp` carries declarations and the sole
`Owner` layout; no leaf creates a second request or lifecycle authority.

Metal now cold-materializes a source-private shared/unified-memory sliding
gate beside each prepared residency owner. Its fixed 480-byte payload contains
expected and published copies of the complete owner, Plan, token, run,
coordinate, turn, stride, slot, descriptor generation, terminal control
generation, read/write masks, and immutable selected locals, followed by a
GPU-written accepted/reason/observed-generation result. The Host never
self-authenticates by rereading that buffer.

Before each accepted coordinate the Host sets the existing Pipeline device
guard to fail closed, writes the shared payload, performs a release fence, and
signals a backend-owned monotonically increasing `MTLSharedEvent` value. The
MTL4 queue waits that exact value before committing one reused command buffer.
Its first unguarded one-command ICB dispatch compares both rows, validates
every
identity and structural relation, proves
`coordinate=turn*stride+slot`, checks mask containment and distinct in-range
locals, and requires
`control[0]=terminal_generation-generation_stride`. Only an exact match writes
zero to the guard. A Dispatch-to-Dispatch device-visibility barrier then
precedes the existing guarded prefix, selected ICB ranges, and suffix in the
same command. The selected ranges consume the immutable expected-row local
snapshot used to build the payload; the encoder does not reread caller
storage after authentication. A mismatch leaves the guard closed, writes
`CompletionInvalid`, and cannot execute stale selected payload.

The implementation is physically split under
`node/src/accel/metal/kernel/pipeline/residency/sliding/`: `shader.mm` owns
only MSL, `descriptor.mm` owns Host row validation and publication,
`prepare.mm` owns cold gate construction, `capability.mm` owns admission
reporting, and `submit.mm` owns only warm-submit ordering. Its adjacent
`submit/` directory separates admission, context capture, cancellation,
descriptor build, command encoding, terminal arming, descriptor publication,
queue commit, and watchdog arming. `finalization.mm`
owns acquired GPU terminal classification, and `diagnostics.mm` owns focused
fault and inspection hooks. No file reconstructs page or victim policy.

The exact completion event is also the CPU acquire point for the GPU result.
The callback requires accepted, zero reason, and an observed descriptor
generation equal to the expected generation. Any mismatch or unknowable
native terminal returns `UnknownMayWrite`, quarantines the adapter and gate,
and strongly retains the prepared owner. `UnknownMayWrite` never maps or reads
the still-GPU-owned result words. Result classification and sticky quarantine
publication run under the same adapter terminal gate that admits a peer
commit, so no peer can cross the fault frontier between physical retirement
and semantic mismatch classification. A Known exact terminal clears all
gate state before the user callback; no post-callback cleanup can overwrite a
reentrant submission. The gate counts one actual queue operation at accepted
commit, including a later watchdog-Unknown command, and reports one native
inflight command per accepted coordinate. Its retained charge is the
sum of the descriptor buffer, gate pipeline, and one-command ICB
`allocatedSize`; its transient charge is the fixed 480-byte stack payload. The
already admitted command
allocator, command buffer, guard, control, and residency set are not charged a
second time.

The persistent route cold-encodes one legacy `MTLCommandBuffer` containing Q
`wait ready -> fixed gate/ICB selection -> signal done` epochs and commits it
exactly once. Its capability `retained_bytes` is the logical incremental Host
charge for the persistent Owner and Q-entry Coordinate vector, while
`transient_bytes` is the fixed payload copied by one service operation. Metal
does not expose an allocation-size query for the retained legacy command buffer
or shared events, so their driver-private memory is opaque and is neither
included nor relabelled as an exact runD-owned byte count. A Known Host-service
failure records the exact failing coordinate, then every remaining ready signal
publishes a zero-mask row. The gate authenticates that row but keeps the payload
guard closed. Because a closed payload does not advance the role's control
generation, each suffix signal seeds the exact predecessor generation before
opening its retained event; this keeps slot reuse authenticated for suffixes
longer than W. Every suffix done is therefore Known, completed, and
undispatched/non-writing at the payload seam. Only the Qth exact acknowledgement
releases the claims and emits the one aggregate Known failure Final. Unknown
terminal or row evidence retains the existing sticky adapter quarantine
instead. Before the one Unknown Final callback, the self-retained owner clears
its raw callback/user target; no late native terminal can rediscover
caller-owned storage.

This O(Q) command stream is not the Metal O(1) recurrence primitive. Metal's
proved Direct Map recurrence instead rewrites one admitted pointwise Map into
one MSL kernel whose device code executes
`for (rund_iteration=0; rund_iteration<rund_iterations; ...)`. The runtime
passes Q as one four-byte kernel argument and retains one Map route and one ICB
Map dispatch plus a fixed set of Pipeline control/status ICB commands.
`InspectMetalFusedDirectRecurrence` accepts only that exact terminal or History
Map form: a History owner must pass the exact Q-slice binding and pitch checks;
dynamic Map control, transducer, ResidentState, aggregate, or second payload
dispatch are rejected. It reports the actual ICB
`allocatedSize`, Map parameter-buffer allocation, route Host bytes, and
complete retained Metal sequence bytes. The dedicated actual contract executes
Q=5, Q=9, and Q=257, checks the numerical Q-fold result, requires one command
submit and one payload dispatch, and requires the total ICB command count and
the native ICB/parameter allocation and recurrence-route Host bytes to be
identical across those three independently prepared runs. No per-epoch native
encode, submit, or callback exists in this fused path.

The common service-free handoff now compacts the repeated semantic-owner table
to one row after backend preparation. The actual Q=5/9/257 product contract
therefore requires equality of `MetalSequence::retained_bytes`, the 2240-byte
recurrence route, the 16385-byte native allocation, and the complete current
common Host/Device/Staging memory tuple. The terminal route currently retains
32101 bytes and the History route 37229 bytes; the native allocation is 16385
bytes and the common tuple is `60576/16629/0` for each tested Q.
Q-dependent authored-occurrence work is cold transient preparation and is not
retained by the admitted request.

The current persistent service ABI cannot safely substitute that fused kernel
for the O(Q) stream. It hands Metal only opaque prepared `MetalSequence` roles
and selected-local/generation identities; it does not hand over the canonical
Map artifact, recurrence binding proof, or a declaration that inter-coordinate
Host backing service is absent. More importantly, its order is
`signal_ready(e) -> GPU done(e) -> Host ack(e)` before the next ready signal.
`MTLCommandBuffer`/`MTL4CommandQueue` event waits and signals are encoded Host
commands, while an indirect compute command can encode Dispatch but not an
event wait or signal. Therefore one arbitrary retained ICB execution per Host
epoch requires Q event/execute records. Moving those waits into a running MSL
kernel would require a documented CPU/GPU system-scope atomic wait and GPU
forward-progress guarantee; Metal Shared storage supplies neither contract.

The common, Authority-proved service-free Direct recurrence handoff now
contains the canonical Map artifact, exact bindings/windows, Q, and one
aggregate terminal/publication contract. The public terminal-output
`Pipeline::repeat<Q>::run()` route selects the fused Metal primitive without
reconstructing semantics from an ICB. Its actual Q=5/9/257 product contract
requires one native submit, one payload dispatch, no per-iteration Host
service/callback, one Final, exact terminal output or exact `write_each`
history slices, Known retry, and sticky Unknown quarantine. Backend
diagnostics publish the native Terminal/History mode, and capability opens
only when it equals the proof retention. Retained common and native route
storage are fixed for both admitted resident shapes; this is not a claim about
total user output/scratch storage or general backing-serviced VSM. A
service-aware handoff remains on the O(Q) lowering until Metal exposes a
GPU-visible Host-event primitive with the required memory scope and progress
guarantee. Large-Q evidence remains open, and the legacy route is deliberately
not relabeled O(1); its capability/storage evidence remains separate.

The separate explicit/resident/required DeviceVsm lowering owns Virtual page
coordinates rather than
repeat iterations of one resident state. For Direct pointwise and centered
Clamp Window, common emits a Metal kernel whose W threadgroups execute
`page = worker; page < Q; page += W`; Clamp preserves Direct/SharedHalo.
Centered Clip submits the same fixed grid but worker zero owns the whole-run
passes and all other workers return. Clip Sum executes its frozen PrefixDifference
passes in the same kernel; Clip Min/Max execute BlockPrefixSuffix and preserve
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
submission commits one command buffer and one payload dispatch; no epoch
callback or service operation exists in the request. Actual public-run
Q=5/9/257 evidence requires Q GPU phase counters, exact output, one queue
submit, and one aggregate Authority/Pipeline/backing Final. Graph Sum and pure
Sum Reduce overflow return Known `ReduceSumOverflow` and publish nothing.
Preparation retains a fixed Owner plus parameter/config/result buffers; whole input/output
backing allocations scale with logical bytes and are not counted as fixed
controller storage.

The dedicated persistent actual contract covers bounded `BackendChunked`
recurrences at Q=2, Q=3, and Q=5. Each accepted chunk has at most two
coordinates, so Metal performs respectively one, two, and three native
submissions while the product still has one logical handoff, zero Host epoch
callbacks, one aggregate Final, and one backing publication. A Known failure
drains only the accepted prefix and leaves the unsubmitted suffix to cleanup;
an Unknown service failure quarantines the accepted native owner and publishes
nothing. Inputs without the two-lane capability retain the OneSubmit/fixed-W
contract. The serialized Window fixture is one such OneSubmit case; the
Pointwise R2 product is the separate Q=2/3/5 chunked evidence. Neither claims
same-submit GPU ownership of nonresident backing data.

The dedicated legacy `OneSubmit` Metal contract executes Q=5, Q=9, and Q=257 with strides
one, two, and three, then injects a stale published row. Every successful
coordinate reports one queue call, one inflight command, and the exact GPU
control generation. The stale row returns `UnknownMayWrite` and closes future
capability. Across all 273 actual commits, descriptor/gate-pipeline/gate-ICB/
event/submission-command/allocator identities and tracked pipeline/buffer
allocation counters remain unchanged after cold preparation.
An independent fresh-adapter terminal-loss row proves exactly one callback,
zero GPU-result reads without an acquire edge, one accepted queue-call delta,
one deliberately retained unknown inflight command, and sticky capability
closure. The dedicated binary passes twenty consecutive actual runs on the
checked Apple unified-memory path.

Product admission nevertheless remains disabled
(`descriptor_release_acquire=false`). The source-private contract supplies
synthetic raw rows and therefore does not yet prove that common Authority
creates and preserves the exact authenticated row across concurrent W=2..4
roles, or that common Release/Final and publication handle a GPU-rejected row
without stale output. Non-unified/Managed storage and generation strides other
than one are also unavailable rather than silently weakening the visibility
or terminal-generation checks. Before enabling the capability, common
admission must also own the already-staged gate charge rather than reserving
the same retained bytes a second time, and the product contract must add exact
stale-output, reentrant-callback, and W=2..4 Release/Final evidence.
