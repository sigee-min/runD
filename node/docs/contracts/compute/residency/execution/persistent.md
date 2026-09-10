# Persistent GPU-Owned VSM Recurrence

This page owns the product acceptance boundary for a GPU-owned VSM
recurrence.  [Sliding](./sliding.md) owns the fixed-`W` logical and physical
transaction model.  The Host-driven raw Sliding transport remains a bounded
fallback and is not evidence for this page.

## Chunked Persistent service

For an eligible nonresident Pointwise Persistent input that exposes at least
two parallel-read lanes, `BackendChunked` is selected for `Q>=2`. `width` remains the two-bank
role width; it is not a chunk size. The common service selects a half-open
`accepted_end` in chunks of at most two coordinates and advances it only after
the backend accepts the chunk. Metal and Vulkan therefore perform
`ceil(Q/2)` physical native submissions while retaining one logical Pipeline
handoff, one Host-free aggregate Final, and one backing publication. `Q=1`
and inputs without that capability retain the existing `OneSubmit`/fixed-W
route. A Known failure drains only the accepted prefix/suffix through
`accepted_end`; unsubmitted coordinates are never signalled, waited, or
acknowledged and remain owned by the existing cleanup path. Unknown terminals
quarantine every reachable owner without publication or version change.
The chunked owner retains two physical payload slots; Vulkan may also retain
four fixed authentication metadata cells, which are not payload slots.

Unknown closes its control with `active=false` and `quarantined=true`, retains
the native owner, and suppresses publication, fallback, and retry.

Window is not a generic Persistent `OneSubmit` product. Unsupported spatial
Window declines before lease to the dedicated bounded Window (`Q<=4`) or
Stream (`Q>4`) fallback, which owns one logical handoff, one Final, and one
output version; the current Metal fallback may make `Q` physical queue calls.
The backend-neutral finite Q=5/9/257 checks exercise the legacy `OneSubmit` seam only,
while public Pointwise Q=2/3/5 cases are `BackendChunked` product evidence.
Explicit/resident DeviceVsm Window remains separate. The finite centered
I32/U32 `WindowRingFused` proof is another explicit DeviceVSM route: Resident
or fully pre-staged input/output is authenticated before lease. It accepts
bare forms and exact authenticated, parameter-free `CanonicalTotalU32` Map
chains of symmetric depth 1–3 before and after the Window. One physical
dispatch/native submit owns Q seed, Q compute, and 2Q internal phases with no
Host epoch service/callback or transfer submission and one Final/publication/
version. It does not include I32 Map fusion. Neither this bounded route nor
the generic service claims same-submit GPU ownership of arbitrary nonresident
backing data.
`UINT64_MAX` is reserved as the no-coordinate sentinel and is rejected before
Metal owner allocation; each active role's final turn must also fit the uint32
control and uint64 descriptor generation ranges. These checks are shared by
common request validation and native preparation.

## Transaction-capable aggregate Scan boundary

The optional `VirtualBackingTransaction` provider is atomic only for the
ordinary accelerator, non-Graph/non-Reduce/non-DeviceVSM aggregate Scan whose
output extent is the complete custom backing. The provider is discovered by
dynamic cast and begins before epoch zero; every epoch uses `stage` into its
shadow, while canonical backing version publication is deferred to one Final.
Final preflights provider coverage, both deferred Pipeline banks, and the
fixed physical output-cache regions before one `commit`. Success rekeys or
retires those rows and applies the per-bank deltas. Known no-write aborts
empty provisional rows and restore private control with unchanged public
bytes/version/generation; unknown outcomes quarantine and poison without
public publication. Resident, non-provider, partial-extent, Graph, Reduce,
and DeviceVSM routes remain legacy or blocked. A reader that bypasses the
backing/provider authority is outside this atomicity guarantee.

## Persistent Sliding owner cache

The admitted Host-driven Persistent Sliding product seals one value-only
static key over the device/adapter, pipeline-owner identity, Plan and ordered
physical roles/ranges, route geometry, backing identities and physical
extents, Q/tail topology, and memory mode. Snapshot generation/parity,
prepared handles, locals, visible bytes/versions, Authority credentials,
callbacks, and terminal fields are excluded. A matching Virtual pipeline
reuses its committed owner, persistent memory charge, native lowering, and
the bounded two-slot backend storage used by `BackendChunked`; the older
`OneSubmit` lowering may retain transient Q-sized native descriptors. Each run
stages a fresh snapshot, complete roles,
handles, locals, and control/descriptor bases before queue acceptance;
the cached common Sliding controller is retained across Known quiescent runs,
while its per-attempt credentials and fixed cells are rebound fresh.
the cached run retains the common `PersistentResidencySlidingPreparation`, its
`RequestCell`, ticket, and native lowering; per-attempt Authority credentials,
callbacks, control, evidence, and result remain transient. A Known quiescent
terminal rebinds fresh token/generation credentials before the next submit.
Shape mismatch, an unknown or writable terminal, authentication failure, or
failed rearm
quarantines the retained owner and cannot fall through to another route.

Authority and the retained Sliding State form one paired lifecycle. Their
gates are always acquired Authority-first. Exact plan/token/generation/owner
and non-wrapping owner-nonce authentication is required before a Known
pre-submit abandon can clear the Authority slot and State attempt together.
Any related mismatch, issued/native/callback work, or UnknownMayWrite evidence
keeps the pair and its credentials pinned in sticky quarantine; it is never
rearmed or released as Known. Warm admission observes one coherent idle
snapshot (run, backend ticket/control, prepared lease, and Sliding
quiescence) and rechecks it immediately before staging. Metal one-shot native
command objects remain outside the runD-owned allocation-free claim.

The cache is runD-owned and allocation-free on the warm Host path. Backend
command/timeline objects may have platform-specific rearm requirements; Metal
and Vulkan retain their lowering and bounded per-coordinate authentication
storage, while opaque driver allocations are not counted as runD-owned
storage; Metal one-shot native command allocations remain outside the
`fixed_native_storage=false` runD claim. The common
`PersistentResidencySlidingPreparation` owns the single
pending ticket and its cold `RequestCell`: backend owners may retain only a
native shadow. Vulkan commits that ticket only after the native timeline queue
accepts; Metal commits it only after the command buffer is committed. Every
pre-accept idle failure calls the common abort and restores the prior Known
owner. An accepted or otherwise validated `UnknownMayWrite` terminal instead
quarantines the ticket/cell and retained roles idempotently; cleanup never
aborts or rearms that owner. This is not a fixed-native-storage or GPU-owned
recurrence claim.

## Bounded Window/Stream fallback

Unsupported Persistent spatial Window declines before lease. The dedicated
bounded Window fallback owns `Q<=4`, while Stream owns `Q>4`; together they
retain one logical handoff, one Final, and one output version. The current
Metal fallback may make `Q` physical queue calls. Its centered-Clamp
`Map(+3) -> Window -> Map(*2)` proof, exact frame/tail authentication, and
cold/warm ownership rules remain where the fallback admits them. The old
`OneSubmit` checks are lower-level backend evidence only, and explicit/resident
DeviceVsm Window remains separate. A bounded may-write or identity-uncertain
terminal is `UnknownMayWrite`: it quarantines the owner, emits no publication,
and cannot fall through. The general dynamic nonresident
Forecast/Promote/Drain/Persist continuation remains blocked.

The ticket is authenticated before any backend trace or state mutation. An
inactive ticket must have no pending request and must match the exact prepared
identity; an active ticket must have one pending request equal to the submitted
request. Fresh credentials are accepted only after Authority stages that exact
pending request. Any mismatch is rejected without mutation.

## GPU-owned nonresident continuation — blocked

The current Metal/Vulkan O(Q) preencoded bounded-W Host service is
an intermediate product path. It is not a GPU-owned nonresident
Forecast/Promote/Drain/Persist implementation. That continuation needs one
fixed-R ring independent of Q with private demand/ack rows carrying
`op`, `ticket`, `coordinate`, `turn`, `page`, `next_use`, `bank`, `frame`,
`resource`, `offset`, `bytes`, `generation`, and `plan_digest`; acknowledgements
carry `result` and `may_write`. The ring law is release/acquire visibility,
contiguous tickets, exact full/empty distinction, a Host-only owner-nonce
shadow, explicit page-ownership transfers, one aggregate Final, and
`UnknownMayWrite` retention of every affected owner.

The missing primitive is a device-to-Host demand wake plus a Host-to-device ack
wait with system-scope visibility and dynamic continuation without another
queue submit. Vulkan still needs a correct timeline/host-event bridge; Metal
needs a correct SharedEvent/ICB/MTLIO continuation, and neither backend yet
provides a truthful noncoherent transfer DAG for this contract. No unused ABI,
diagnostic row, or static proof is treated as that implementation.

A recurrence that needs no inter-iteration backing service uses the distinct
[Service-Free Direct](./service-free.md) contract. It is not represented by a
persistent Sliding request with empty callbacks.

The public default first probes `StagedLoop` for an ordinary all-staged unary
nonresident Pointwise run with `Q>=2`. Exact mapped Host-visible/coherent
input and output views are required; the route enum carries this decision into
preparation without recomputing shape, tier, or max-read policy. StagedLoop
owns one native recurrence submit, `Q` GPU epochs, zero transfer submits/bytes,
zero Host epoch submit/service/callbacks, and one Final/Authority/output
publication/version. A structural `BackendUnsupported` before owner, rearm,
stage, lease, or native acceptance is a clean decline to the legacy route;
owner mutation, native acceptance, or any non-capability failure is terminal.
The legacy nonresident Persistent Pointwise R2 path is `BackendChunked` with
two-coordinate chunks and `ceil(Q/2)` native submissions; old `OneSubmit`
evidence remains a lower-level fallback seam.

Persistent spatial Window is not the generic `OneSubmit` product. Unsupported
Window declines pre-lease to dedicated bounded Window (`Q<=4`) or Stream
(`Q>4`) fallback, which owns one logical handoff, one Final, and one output
version; current Metal fallback queue calls may be `Q`. Explicit/resident
DeviceVsm, Q1, Clip, scan, reduction, shared external rows, same-stage
external fan-in, malformed/unsupported Graph shapes, and parallel-read shapes
retain their existing routes. An ordinary accelerator nonresident,
non-required `GraphPointwise` with Q>=2 first passes the hard bounded-plan
check: every external input is nonresident with serialized reads, each stage
has exactly one external read, and the sealed topology is valid.
Host deferral then requires either the exact `graph_wavefront_pair_eligible`
proof or an output backing that advertises `VirtualWriteLanes::write_lanes() >= 2`.
The pair branch issues both bounded Host Forecasts before nonblocking callback
polling in deterministic lane order and delays cross-batch prefetch until the
pair retires. The output-capability branch uses the ordinary fixed two-slot
Drain-to-Persist ring. Cross-stage reuse of one external input row is permitted
when each stage has exactly one external read; same-stage external fan-in
remains rejected. A serial-output GraphPointwise remains on the DeviceVsm
probe. The public one-input three-stage branch/join product exercises the
two-slot ring at Q=5 with two simultaneous output callbacks, exact logical I/O
bytes, exact output, and one backing version/publication. Selected stages still
use Host-owned native submit/wait; this does not claim GPU-generated or
fixed-storage recurrence.

The natural public ring evidence above is success-only. Known/Unknown
Forecast, Drain, and Persist retry/quarantine lifecycle coverage remains in
the lower-level contracts and dedicated service-aware fixtures; this ring
fixture does not claim a public natural-run failure/retry proof.

For Metal, the Window portion of this default is currently limited to the
truthful centered Clamp stride-one SharedHalo HostCoherent subset. Its prepared
owner retains an immutable proof of the three-step spatial `Map -> Window -> Map`
schedule, exact SharedHalo candidate and centered-stride-one shape, and complete
per-local ICB ranges. This is separate from the temporal `MetalWindow`/`ResidentState`
recurrence binding, which remains a ServiceFree Direct/history concern and is
not admitted by Persistent Sliding. The actual Q=2, N=48,
F=16, R=2, P=12 Clamp evidence is one native submit and queue call, 192
logical backing-read bytes, 256 expanded-frame promotion bytes, zero duplicate
canonical-overlap reads, exact output, and one aggregate Final and publication.
It is still O(Q) preencoding with HostCoherent callbacks; it does not claim
GPU-owned or fixed-native recurrence, and this evidence is Metal-only.

The Vulkan `GeneratedIndirectMap` variant is a source-private admission and
ownership implementation, not a public product claim. It retains fixed Map
control/indirect records, authenticates run and full/tail role lifetime, and
keeps ordinary four-binding/64-byte versus checked six-binding/128-byte layouts.
Dynamic controls, history, malformed rows, and mixed non-Map schedules remain
fallbacks; O(Q) intermediates, `device_generated_recurrence=false`, and
`fixed_native_storage=false` remain explicit.

No natural callback-backed public E2E evidence is currently valid: Gather is
whole-buffer, whereas the fixed unary page-frame owner requires page-local
indices. The removed fixture was therefore not evidence of Persistent mode,
mixed E2E, or Known failure. A future public route needs an explicit
PageLocalIndex proof, bounded authenticated multi-page Forecast/Promote/page
tables, or a whole-resident source; until then this mode remains private.

Status: **implemented for the explicit/resident/required Direct pointwise,
centered Clamp U32 Window Sum/Min/Max in generic Persistent Sliding; Clip is
explicit DeviceVsm-only, with its exact parameter-free canonical-total
`Map DAG{1,2,3} -> Window -> Map DAG{1,2,3}`
composition, exact sole-input/sole-output pure total U32/U64 pointwise Map DAG,
exact two-through-seven-input/one-output same-type pure total U32/U64
pointwise Map,
exact one-input two-through-eight-stage, two-through-seven-input/two-stage,
six-input/three-stage, five-input/four-stage, four-input/five-stage,
three-input/six-stage, or two-input/seven-stage
parameter-free canonical-total acyclic U64
`GraphPointwise`,
pure unsigned U32/U64 Inclusive/Exclusive Scan, exact canonical
total element-local one- or two-read/one-value-write U64
`Map expression DAG -> Scan` actual boundary plus one-through-seven-read
common admission/source generation, pure unsigned U32/U64
Sum/CountNonzero/Min/Max Reduce, and exact service-free canonical total
U64 one-through-six-input Map-DAG-to-{Sum, CountNonzero, Min, Max} `DeviceVsm`
boundaries; partial for the general
persistent product**.
The older `OneSubmit` service-aware request,
fixed Control, typed data-service rows, aggregate evidence, validation, and
finite Q=5/9/257 fake contract are implemented. Separate Metal and Vulkan
actual-adapter finite Q=5/9/257 checks wrap the exact coordinator slots in a
test-local `DeviceOps` copy; they are direct backend evidence, not public
`VirtualPipeline::run()` product evidence. They prove one native queue submit,
zero epoch native submits/callbacks, Q exact ready/wait/ack service turns and
GPU coordinates, one Final, exact output, and lower-level coordinator
publication facts. The separate service-aware public-run
fixture proves a Known
first-unsent backing failure drains Q failed-ready no-dispatch gates, aborts
without publication, and permits a same-pipeline retry. Its Unknown case
proves one aggregate Unknown Final, zero publication, retained Authority and
backend lifetimes, sticky quarantine, and retry rejection without another
queue submission. The fixture also requires both original production
`DeviceOps` product slots to be non-null and delegates through those exact
functions. Those results prove the `OneSubmit` pre-encoded intermediate; they
do not prove that this older service-aware lowering owns recurrence
generation.

The distinct `DeviceVsm` product path now closes the stronger success boundary
for explicit/resident/required admitted Direct pointwise runs, statically composed sole-input/sole-output
pure total U32/U64 pointwise Map DAGs, one centered Clamp/Clip U32 Window
Sum/Min/Max stage, its exact parameter-free canonical-total U32 one- or
two-Map-DAG prefix/suffix
composition, one pure
unsigned U32/U64 Inclusive/Exclusive Scan stage, the exact canonical total
element-local one-through-seven-input U64 `Map expression DAG -> Scan` composition,
one pure unsigned
U32/U64 Sum/CountNonzero/Min/Max Reduce stage, and the exact service-free Graph
shapes `one external input -> two through eight parameter-free U64 Map stages
with bounded branch/fan-in -> one U64 backing output`,
`two through seven external inputs -> two parameter-free U64 Map stages -> one
U64 backing output`, `six external inputs -> three parameter-free U64 Map
stages -> one U64 backing output`, `five external inputs -> four
parameter-free U64 Map stages -> one U64 backing output`, `four external inputs
-> five parameter-free U64 Map stages -> one U64 backing output`, `three
external inputs -> six parameter-free U64 Map stages -> one U64 backing
output`, `two external inputs -> seven parameter-free U64 Map stages -> one
U64 backing output`, and
`one through six U64 inputs + canonical total Map DAG -> {Sum, CountNonzero,
Min, Max}`. The pointwise stage-DAG shape exists specifically for individually valid
stage artifacts whose composed Program exceeds the static expression
capacity. It authenticates every producer/consumer physical view and the
complete planner wavefront, substitutes the exact external input or earlier
stage register at each bounded Read, and creates no intermediate allocation or
Host epoch edge.
Actual Metal and Vulkan `DeviceVsm` Q=5,9,257 product runs require one handoff, one queue
submit, one payload dispatch, zero Host epoch submit/service/callback control,
exact short-tail output, every stage generation, and one aggregate publication;
Q=5 additionally proves Known partial-write recovery and same-state retry.
The Final holds every stage Pipeline state/publication gate through the one
backing commit. This admission is limited to two through eight total U64
stages, one through eight Reads and one Write per stage, at most seven public
external inputs, and the parameter-free canonical-total scalar operations
owned by the typed-Map validator. Parameterized, non-total, or unsupported
operations such as integer division remain rejected. More
than one public input is proved through seven inputs at two stages, six inputs
at three stages, five inputs at four stages, four inputs at five stages, three
inputs at six stages, and two inputs at seven stages.
Every
nonterminal stage output must feed a later stage. The actual three-stage case
proves two independent branches and a one-public-input fan-in. The distinct
two-stage actual binds two external resident rows, proves `2Q` exact backing
reads at Q=5,9,257, and preserves the same one-submit/zero-Host-epoch/one-Final
law. Whole-run private inputs use their authenticated coherent Host write view
when present, including legacy Direct Graph collectives; a route label is not
a second authority over mapping availability. Only an absent view takes the
existing native-upload path. This does not permit Host writes to public
resident inputs, which remain separately authenticated and skip staging.
The maximum-width two-stage actual binds seven rows at Q=5 and proves
thirty-five exact reads with that same law. The deep actual binds six rows to
three stages at Q=5 and proves thirty exact reads totaling 3,504 bytes,
three stage generations inside the same one-submit/zero-Host-epoch/one-Final
transaction. The four-stage actual binds five rows at Q=5 and proves
twenty-five exact reads totaling 2,920 bytes, four stage generations inside
that one transaction, and exact XOR/multiply/add continuation results. The
five-stage
actual binds four rows, proves twenty exact reads totaling 2,336 bytes,
commits five stage generations inside one aggregate publication transaction,
and uses nine planner resources: four inputs, four intermediate values, and
one output. The six-stage actual binds three rows, proves fifteen exact reads
totaling 1,752 bytes, and commits six stage generations. The seven-stage actual
binds two rows, proves ten exact reads totaling 1,168 bytes, and commits seven
stage generations. Both retain the same nine-resource and one-submit boundary.
These
cases are not evidence for more than seven external inputs, multi-input graphs
beyond seven stages or other unproved combinations,
cyclic/runtime-selected Graph recurrence, or runtime backing service.
A maximal linear prefix uses its
compiled Map stage. An admitted public-input
fanout/fan-in graph instead owns one separately compiled semantic Pipeline
whose typed expression is equivalent to the pure Map DAG; the collective is
the second publication owner. The original physical Graph stages remain
available to the Host-wavefront fallback. DeviceVsm now authenticates their
complete ResidencyPlan identity, fixed frame/batch geometry, and
phase-separated same/prior predecessor masks. The submitted GPU workgroup
deterministically traverses those masks, opens the exact lowered Map and
collective payload owners at their selected positions in every batch, and
emits an exact step count and ordered trace that common Final validation
compares to the proof. It counts
Virtual backing pages rather than iterations of one resident state, owns exact
page/payload/frame geometry, requires whole-run GPU-addressable input and
output backing, and exposes only one aggregate submit and Final. Host
ready/wait/ack, Project, Release, Returned, and wake entries are absent from
that request type. Actual Metal and Vulkan public-run Q=5/9/257 contracts prove
one queue submit, one payload dispatch, device-generated progress through all
Q pages, zero per-epoch Host submit/service/callback control, exact output,
one exact GPU wavefront trace, and one Authority/Pipeline/backing Final
transaction. This is a static service-free Graph traversal: the fused semantic
payload is attached to its selected Map owner and the reduction to its selected
collective owner, but canonically eliminated intermediate Maps have no separate
payload and arbitrary physical stages remain unsupported. It does not consume
runtime backing-service readiness. The composed Scan proof
retains the canonical typed Map artifact and complete typed ParsedIR authority,
copies its exact parameter bytes when present, and emits every authenticated
total expression node into the same Scan shader; it does not materialize a Map
intermediate or create a second dispatch. Compile-time literals legitimately
produce a zero-byte parameter buffer because their values are already part of
the retained artifact and generated source identity. Scan and Reduce use one
256-lane workgroup. Sum and CountNonzero fold strided lanes through a two-word
tree; Min and Max use an unsigned identity tree; Scan owns its carry and
advances ordered 256-element chunks. Sum and Scan first prove the mathematical
sum fits the U32 or U64 destination width without exposing a Host carry. U32
Sum uses an exact U64 precheck accumulator; U64 Sum uses a two-word
accumulator. CountNonzero cannot exceed the admitted U32 logical-element
bound. A
checked overflow reports the exact first failed page as Known/no-write,
publishes no backing version, and permits the same Pipeline state to retry.
The Scan actuals include the one-input fanout/fan-in DAG `(x+1) + (2*x)` and
the two-input `a+b` DAG for both Inclusive and Exclusive Sum at Q=5/9/257.
Common lowering owns the composition into one Map step before the Scan; the
proof authenticates two lowered stages, the exact ordered resident input set,
zero parameter bytes for those embedded literals, one native submit, zero
Host epoch control, exact one- or two-input backing bytes, and one aggregate
publication.
The admitted common limit is now seven ordered inputs plus one output. A
focused source oracle generates that maximum-width expression for both backend
APIs and both Scan operations, and an eighth input fails cold. Actual Metal and
Vulkan Q=5 maximum-width cases bind every row and prove 35 input pages, one
native submit, zero Host epoch control, and one aggregate publication for both
Inclusive and Exclusive Sum; one/two-input Q=5,9,257 cases independently prove
depth.
The U32 and U64 pointwise DAG actuals use the same public-input fanout/fan-in expression
without a collective. Common lowering composes its three authored Map nodes
into one unbounded canonical Map Program before accelerator preparation.
Metal and Vulkan Q=5/9/257 contracts require Pointwise proof identity, exact
logical output, one native submit and payload dispatch, zero Host epoch
service/callback control, one aggregate publication, zero parameter bytes for
the embedded literals, and identical retained controller bytes across Q.
This is a static semantic fusion boundary, not runtime GPU selection of Graph
ready cells.
The Graph cases include one authored Map plus every supported terminal
reduction, two authored Maps plus Sum, and a four-authored-stage
public-input branch `(x+1) + (2*x) -> Sum`. The linear fused case
retains `authored_nodes=3` while lowering to `lowered_nodes=2`, so its removed
Map intermediate has no VSM stage. Both use one 256-lane workgroup to traverse
and reduce Q pages inside the shader, publish the Map and Sum Pipeline stages
together, and return checked
`ReduceSumOverflow` without output publication when the mathematical U64 sum
does not fit. The U32 Window Sum/Min/Max cases also authenticate the frozen
Range-plan family and exact eliminated physical overlap
`(Q-1)*(prefix+suffix)`. The exact composed shape admits either the specialized
U32 wrapping add/multiply immediate pair or up to three independent
parameter-free canonical-total single-read/value-write U32 Map DAG stages
before and after the Window. The generic proof retains every typed admission
and exact artifact identity; source text is never reparsed for authority.
Actual Q=5/9/257 cases use `+3`/`*2`, branch expressions
`(x+1)+(2*x)` / `(x+5)+(3*x)`, a five-stage chain, and the maximum
seven-stage `Map -> Map -> Map -> Window -> Map -> Map -> Map` program whose
adjacent Map expressions do not fit the ordinary composition envelope. Every
operation and boundary proves exact output with the same one submit, zero Host
epoch control, overlap, and aggregate publication laws.
For the explicit DeviceVsm owner, Clamp preserves the frozen Direct or
SharedHalo execution. Its Clip path preserves the frozen PrefixDifference Sum
or BlockPrefixSuffix Min/Max selection but executes all Range stages inside
the same device-owned whole-run kernel.
PrefixDifference reuses private input staging as its prefix row;
BlockPrefixSuffix reuses input for forward aggregates and output for backward
aggregates, then writes final results in descending coordinate order so no
live suffix row is overwritten. Neither path allocates another Q-sized
scratch buffer or introduces a Host stage callback. The backend retained
controller bytes are identical across Q=5/9/257 for each prepared topology.
The explicit DeviceVsm Direct pointwise backend contract additionally executes Q=100,000 with
the same retained controller bytes as Q=5/9/257. Its explicit DeviceVsm public Virtual
public product contract uses Q=100,001 because that fixture deliberately seals an
odd page count with a physically short final page. It still performs one
queue submission, zero Host epoch service/callback control, device-generates
and completes all 100,001 coordinates, and commits one aggregate
Authority/Pipeline/backing Final transaction.

This is not a claim for every Virtual shape. Current generic Persistent Sliding
admission is nontransactional Direct pointwise, including the exact statically
composed sole-input/sole-output pure total U32/U64 Map-DAG subset above and
one centered stride-one Clamp U32
Window Sum/Min/Max whose input/output frame and payload geometry are identical and
whose prefix and suffix equal the authored radius, optionally surrounded by
the exact one- or two-stage parameter-free canonical-total U32 Map-DAG chains
above, one pure unsigned U32/U64
Inclusive/Exclusive Scan, exact canonical total element-local one- or
two-input U64 Map expression DAG followed by U64 Inclusive/Exclusive Scan,
one pure unsigned
U32/U64 Sum/CountNonzero/Min/Max Reduce, or
the exact U64 Map-DAG-to-Reduce Graph subset above: one through six inputs with any
of those four terminal reductions. Its Map semantic owner is the
complete retained canonical typed IR with the exact one through six public U64 reads
admitted by the route and one terminal U64 value write; all nodes must be total.
The actual suite covers wrapping Add and two-input `xor/multiply/add` for all
four reductions, plus non-summary one- and two-Map Sum prefixes. The Graph input is
whole-run resident and its output is one U64 scalar;
no typed Forecast/Promote/Drain service turn occurs during native execution.
The frozen Range planner may select Direct or SharedHalo for the generic
centered Clamp path. PrefixDifference for Clip Sum and BlockPrefixSuffix for
Clip Min/Max belong to the explicit DeviceVsm proof and execute in one payload
dispatch. A Clip Window with a prefix Map is marked
DeviceVsm-required: if the exact proof is unavailable, preparation returns
`BackendUnsupported` before the service-aware fallback because padding a
mapped Clip input with the Window identity would change its semantics.
Signed/fixed, segmented,
or otherwise non-U32/U64 Scan/Reduce
shapes, other Graph shapes, more than three Map stages on either side of Window,
other fusion across Window or a collective,
stateful, profile, or publication boundaries, asymmetric Window,
and other prefix/block multi-pass Range families remain outside this route. It
allocates whole-run GPU-addressable
input and output backing, so those dataset bytes scale with the logical extent
even though retained controller/native state is fixed in Q. Deterministic
DeviceVsm Unknown/device-loss injection is implemented on the public Metal and
Vulkan pointwise route. The injected command is still submitted once, but its
aggregate result is not acquired: the common terminal authority emits one
`UnknownMayWrite` Final, output backing writes and version publication remain
zero, recovery stays live, the Device residency frontier is quarantined, and a
strong self-cycle retains the exact prepared buffers, Pipeline states, Pool,
memory reservation, and Authority proof after public wrapper destruction. A
same-owner retry returns `DeviceLost` without another native submission.
Same-workload performance remains open. The Graph checked-overflow Known
failure is implemented and tested independently; it is not used as Unknown
evidence.
Warm preparation is closed for an exact active shape: one VirtualPipeline-owned
cold proof, pair of GPU-addressable buffers, exact input/output fallback staging,
backend lowering, Authority resident registration, and their committed logical
memory reservation survive successful runs. Every run still mints a fresh
Pipeline snapshot and Authority token/generation/nonce, while the backend only
clears its aggregate result cell and re-arms the one-submit owner. A changed
active extent or proof identity replaces the cold owner before admission.
Producer counters and invariant retained capacity prove that this runD-owned
warm state does not allocate. Process-global interception still observes
driver-private work records when Metal creates its one-shot command buffer and
when MoltenVK submits; those opaque native allocations are not reclassified as
runD ownership and remain a measured performance cost. Those explicit limits
keep the general persistent status partial without retracting the implemented
one-submit success path.

## Explicit DeviceVsm product boundary

For W fixed GPU workers and Q distinct backing pages, the explicit/resident
DeviceVsm native controller is
the device loop:

```text
Host: whole-run backing read -> submit once -> wait aggregate Final
      -> whole-run backing write -> publish once
GPU: topology-owned loop authenticates and completes all e in [0,Q)
```

The Host read and write are whole-run data-service boundaries outside native
execution. Neither creates an epoch scheduler callback or a native submit.
The product request contains the immutable 128-bit proof, one canonical
fixed-capacity resident set, Authority token/generation/nonce, lowering owner,
and one Final callback. The resident set has at most eight input-major then
output-major rows. Every row carries one exact `ResidentBufferRef` and one
strong backing handle; its input/output counts must equal the sealed
`ComputePlan`, resource ids and handles must be pairwise distinct, and unused
rows must remain empty. The proof, Authority registration, and Metal/Vulkan
descriptor projection consume this same ordered set instead of reconstructing
first-input/final-output scalar bindings. It contains no per-page control
operation.

This resident-set ABI removes the prior two-scalar authority and is also the
physical authority for the implemented public multi-input pointwise route.
That route admits two through seven distinct same-type U32 or U64 inputs and
one distinct output, all with the same element count and page geometry. Input
rows remain in Program-port order, every input and output backing is locked as
one address-ordered set, and admission rejects any typed-count, identity,
extent, alias, or unused-row mismatch before physical preparation.

The common `VirtualPipelineState` cold owner follows the same authority shape:
it stores up to seven public inputs in canonical Program-port order and one
public output, with no scalar input mirror. Unary and Graph-reduce preparation
populate only input row zero. Multi-pointwise preparation populates every
public input row, validates every unused row is empty, stages each exact backing
once before submission, and binds the ordered resident set to one immutable
DeviceVsm proof. The public `virtual_pipeline(program, inputs..., output)`
surface selects this route for the admitted subset; it does not fall back to a
per-page Host scheduler.

Warm-owner authentication is physically partitioned under
`virtual/run/device_vsm/prepare/warm/`: `identity.cpp` owns immutable
identity/topology/geometry/ring matching, `pipeline.cpp` owns selected Pipeline
and backend-binding continuity, `pointwise.cpp` owns the refreshed wavefront
and page-map proof, `resident.cpp` owns the complete resident proof comparison,
and `wavefront.cpp` owns wavefront equality. The adjacent `warm.cpp` only
orders these checks and assigns the established failure reason; its
declarations-only local seam carries no cached proof or second route authority.

The DeviceVSM admission boundary follows the same ownership cut under
`virtual/run/device_vsm/admission/`: `shape.cpp` owns the route-specific
semantic shape predicates and graph-stage projection, `pipeline.cpp` owns
Pipeline selection and shared Authority pairing, `validate.cpp` owns binding
and device/type checks, and `owner.cpp` owns cached-owner authentication.
The parent `admission.cpp` is only the ordered coordinator: it verifies the
identity proof, preserves the established capability/shape/binding failure
precedence, and delegates to those leaves. `identity.cpp` is the sole route
stamp/proof-matching owner. The accelerator source manifest lists each leaf
once; no route or admission implementation redefines the identity recipe.

The product and backend contracts prove that this physical foundation is not a
scalar mirror. A common, fully validated DeviceVsm proof with two canonical
Input rows and one Output row lowers `output = first + second` and runs Q=5,
Q=9, and Q=257 on actual Metal and Vulkan adapters. Each case enters through
the public product coordinator and sealed `PrepareDeviceVsm` token router,
registers all three rows with Authority in input-major/output-major order,
binds all three native descriptor rows, returns exact output, and records one
native submit, zero epoch submits, zero Host service turns or epoch callbacks,
Q generated and completed pages, and one aggregate Final. The public typed API
case independently proves two distinct input backings are read (`2Q` physical
pages and twice the logical input bytes), one output backing is written, and
both shared Pipeline banks plus the backing version publish atomically once.
The maximum-width public case separately binds seven ordered Input rows and
one Output row, executes Q=5 on both actual backends, and proves `7Q` physical
input pages, seven times the logical read bytes, one native submit and payload
dispatch, exact output, and one backing publication. Thus the two-through-seven
surface and fixed eight-row resident-set bound are execution evidence, not only
a type or admission claim.
This does not implement multi-input Graph Forecast/Promote, multiple outputs,
or service-backed input turns during native execution.

Common preparation authenticates both fixed Pipeline banks against either the
same canonical pointwise Map, or the same exact centered Clamp Window for
generic Persistent Sliding. Clip Window preparation belongs to the explicit
DeviceVsm owner and is not generic Persistent Sliding evidence; that owner
authenticates the centered Clamp/Clip Window authority, the
same exact pure unsigned U32/U64 Scan authority or exact canonical total
one-through-seven-read/one-value-write U64 Map-DAG-plus-Scan authority, the same
exact pure
unsigned U32/U64 Sum/CountNonzero/Min/Max Reduce authority, or the exact Graph Map and
collective
pair. A
pointwise Map is rewritten once into the page controller. A generic admitted
Clamp Window is lowered from its already-selected Range plan into a distinct
controller artifact that reads the whole-run resident input, writes only
logical core outputs, and preserves Clamp Direct/SharedHalo semantics. The
explicit DeviceVsm Clip owner separately preserves PrefixDifference or
BlockPrefixSuffix semantics. Scan and Reduce dispatch
one workgroup: their wide precheck, and Scan's carry, stay device-owned until
the aggregate terminal. Preparation owns the copied windows and parameters and
registers the complete canonical resident set with Authority. Pointwise and Window
Direct/SharedHalo dispatch W threadgroups on Metal and W workgroups on Vulkan;
each workgroup advances its own `e += W` sequence in device code. Clip
multipass dispatches the same fixed grid but only worker zero owns the
whole-run prefix/forward/backward loops; the other workers return without
touching state. The Window proof additionally owns a fixed-size canonical
page-footprint authority. Each GPU page iteration contributes its exact
adjacent-boundary transition and a checksum of the epoch, first/last canonical
input page, and active core width; Final compares both aggregate result words
to the sealed proof. The aggregate result words are written by the GPU and
accepted only when their topology-specific terminal algebra closes.

Pointwise and `GraphPointwise` additionally own a physical fixed-W native ring
rather than accepting only a completed-page count. Each backend allocates
exactly `4W` bytes of turn state and
`resident_count*W*payload_bytes` of page scratch, where `resident_count` is
the number of exact input rows plus the output row. This storage is independent
of Q. For page `e`, the GPU atomically requires the prior state of slot
`e mod W` to equal `floor(e/W)`, then increments that physical slot. Every
resident input page is first copied into its distinct slot scratch frame. The
Map payload binds only those scratch inputs and the output scratch frame, not
the whole-run resident rows. The computed output is then copied from scratch
to the resident output. Exact output therefore depends on one physical
input/output scratch round-trip per resident row and page. The GPU also
contributes page ordinal, slot, one-based turn, and active element count to a
commutative modulo-2^32 checksum. Common Final derives the expected checksum
and storage in O(1) from triangular sums and requires exactly
`Q*resident_count` round-trips and `Q-W` reuse transitions. Actual Metal and
Vulkan product contracts close this receipt at Q=5,9,257 and the long Q cohort
while retaining Q-invariant native storage, one `DeviceVsm` native submit, zero epoch
submits/service turns/callbacks, and one aggregate Final. Whole-run input and
output backing rows are still GPU-addressable residents used as the source and
sink of those device-side copies; this does not prove a running-GPU/Host
readiness ring or permit per-epoch backing callbacks inside the native
recurrence.

Product evidence distinguishes the native proof from the public backing.
`gpu_addressable_backing` says the backend proof binds GPU buffers;
`public_gpu_addressable_backing` says those exact buffers are the public
Virtual backings. Generic callback backings instead record one
`whole_run_staging` run. `bounded_page_io` is the fixed-W resident-to-scratch
GPU access above, while `bounded_external_page_service` remains false. The
latter is the explicit unimplemented boundary for runtime bounded
Forecast/Promote/Drain/Persist service; none of the other bits may substitute
for it.

Production selection occurs before the Host-service Sliding fallback. A
missing capability rejects this route before an Authority lease exists. After
native submit, any malformed aggregate receipt becomes `UnknownMayWrite`; it
cannot fall back to a Host epoch loop.

The prepared capability query is the sole pre-admission validator and returns
a typed capability result. An explicitly absent product hook may report
`compute_backend_unsupported`; malformed role/state, mode or memory
disagreement, and native invalid or unknown diagnostics remain terminal. The
query preserves the backend reason so the execution boundary can project it
without laundering a failure into a clean decline.

The execution result carries a typed disposition, defaulting to `Terminal`.
Only `prepare_persistent_lowering` returning a preserved
`compute_backend_unsupported` capability result, followed by a successful
`abandon_bound_execution_sliding`, may mark `CleanPreNativeDecline`; that
boundary is before pipeline initialization, native acceptance, and Final, so
the dispatcher may continue with rolling. Backend preparation preserves its
native failure reason through the common prepared seam: pipeline and device
capacity (including stream-capacity) and memory-budget failures remain
terminal and never become a rolling fallback. Prepared-owner authentication,
begin/start/submit failures, service or Final statuses, and any result with
uncertain cleanup remain terminal. A `BackendUnsupported` status or
`poison_pipeline=false` alone is never fallback evidence.

## Definition

For an invocation with `Q` logical coordinates and `2<=W<=4` reusable roles,
the accepted product shape is:

```text
Host:  prepare -> submit once -> signal ready(e) -> wait done(e)
       -> Drain/Persist(e) -> acknowledge done(e) -> Final once
GPU:   for e in [0,Q): wait ready(e) -> authenticate -> execute -> signal done(e)
Final: aggregate terminal -> Authority close -> one public publication
```

The native schedule is complete before the one submit becomes visible.  After
that point the Host may fill or drain an Authority-minted physical ring and
signal an already-recorded dependency.  It may not encode, seed, submit, or
select a Dispatch for coordinate `e`.  A backend terminal may notify only the
aggregate Final.  Per-coordinate `Project`, `Release`, and `Returned`
callbacks are structurally absent from the product request.

The fixed mutable state is independent of `Q`:

```text
RunHeader = plan + token + generation + owner + Q + W
Role[W]   = bank/parity + control base/stride + prepared owner
Cell[W]   = coordinate + turn + descriptor generation + ready/done state
Frontier  = ready + completed + persisted + first failure
```

A backend may retain an `O(Q)` encoded native command stream only as an
explicit intermediate lowering.  It must report that storage before
Authority admission and may not call it a fixed-storage persistent kernel.
Replacing that stream with device-generated commands or a persistent kernel
is a separate, stronger acceptance item.

## Product API law

The Host-driven `PreparedResidencySlidingRequest` remains fallback-only.  The
persistent product uses a distinct typed request with:

- one immutable whole-run credential and role table;
- exact final `tail_local_count` in `1..Role[(Q-1)%W].local_count`;
- one fixed GPU-visible service ring;
- one aggregate Final callback;
- no coordinate projection, terminal, returned, wake, or native-submit
  callback;
- capability facts for `whole_run_preencoded`, and the `mode` submit-shape
  authority (`OneSubmit` = 1 native submit; `BackendChunked` = `ceil(Q/2)`
  native submits), plus `host_epoch_callbacks_zero`, memory visibility, and
  exact retained/transient bytes;
- separate `device_generated_recurrence`, `fixed_native_storage`, and
  `fixed_common_storage` facts. All three are required before the O(Q)
  intermediate may be called a genuinely GPU-driven fixed-storage
  recurrence. Fixed backend command storage does not hide Q-sized common
  occurrence or control rows.

Preparation is mutation-free.  A missing capability selects an older route
before an Authority token exists.  The sole post-preparation capability
decline is the typed, clean pre-native lowering disposition described above.
A failure after initialization, acceptance, service, or Final is a run
failure or `UnknownMayWrite`; it is never relabeled as fallback.

## Data-service boundary

External backing is not a GPU scheduler.  The Host service lane may perform
the exact Fetch, Promote, Drain, and Persist operations authorized by the
fixed ring.  It publishes data readiness and consumes completion, but it
cannot create or enqueue native work.  The distinction is observable:

```text
native_submit_count       = 1
epoch_native_submit_count = 0
epoch_scheduler_callbacks = 0
backing_wait_count         = Q on a completed run
backing_signal_count       = Q on a completed run
backing_ack_count          = Q only after Drain/Persist terminals
backing_service_turns     = workload/cache dependent
final_callback_count      = 1
```

`resident_virtual_backing<T>(device, count)` is the explicit exception to the
whole-run callback copy. It allocates one runD-owned Buffer, keeps that exact
physical owner in the backing's private authority state, and exposes ordinary
byte `read`/`write` only for initialization and observation. DeviceVsm cold
preparation may bind that Buffer directly when its Device and Type, physical
owner, and resident handle/identity all match. The allocation extent is a
capacity, so it may be larger than the current run: the active logical input
and output byte/count prefixes must be nonzero and fully covered by that
extent/count, while the cached proof retains the exact physical owner and
handle identity. A later run cannot substitute another resident allocation
under the same logical backing identity. The backend binds the full physical
view, but GPU accesses and proof geometry remain bounded to the active
prefix; this is a prefix binding, not a dynamic-paging policy.

On this admitted route, input staging and output writeback vectors are empty,
`backing_read_bytes=backing_write_bytes=0`, and the one submitted GPU
controller reads and writes the resident rows directly. Thus successful
resident execution has zero Host backing read/write, while GPU page-I/O
evidence still reports the exact active logical bytes traversed. The final Host
output-hash observation is not an epoch scheduler, data-service callback, or
native submit.
The first successful resident run observes that hash once from the stable Host
view. A warm rearm with the same prepared proof and the exact same sealed input
backing versions reuses the committed hash instead of rescanning the output in
O(N) Host work. Input `invalidate()` or any projected input-version change
forces a fresh observation. A fresh hash is not made reusable until the
Authority, Pipeline, and backing publication transaction commits successfully;
the cache is telemetry evidence and never recurrence or publication authority.
Clip Window input remains on the private fallback because its proved native
algorithm mutates input scratch; generic callback backings and resident
outputs without a stable Host observation view retain the established
whole-run staging boundary.

The backend-neutral service surface has four exact identity-bearing rows plus
one immutable Control. `ReadySignal` publishes
`(Plan, token, generation, coordinate, turn, slot, control generation,
descriptor generation, admission)` after Host input service.
`DoneWait` synchronously waits from the Compute service thread and returns a
`DoneObservation`; it is not a callback. After the exact Drain/Persist terminal,
Compute sends `AcknowledgeDone` with the service result. `ServiceFailure`
records a Known or Unknown backing failure without creating native work.
The function table therefore separates `submit`, `signal_ready`, `wait_done`,
`ack_done`, and `fail_service`; only `submit` may be a native submission.

Control validates every service coordinate monotonically. Slot reuse
`signal_ready(e+W)` is rejected until `ack_done(e)`, so GPU completion cannot
overwrite a Host output still being drained or persisted. The aggregate Final
is rejected until the Qth acknowledgement; the last GPU done signal alone is
not publication authority. A failed ready admission must carry a nonempty
reason. It seals the first failed coordinate, and every subsequently opened
Known suffix admission must remain failed so the already-submitted GPU suffix
passes only through its no-write gate. Successful Final requires no failed
admission or service failure.

Running-kernel polling of ordinary shared memory is forbidden unless the
backend proves a CPU/GPU system-scope atomic and forward-progress contract.
Metal shared storage requires explicit synchronization between processors;
Vulkan host-set events may technically participate in a command-buffer wait,
but the Vulkan specification calls waiting for a future Host-set event
defunct because the allowed stall period is undefined.  Product lowerings
therefore use queue/command-buffer event or timeline operations, not an
unbounded shader spin loop.

## Vulkan lowering

The safe intermediate Vulkan lowering records the complete recurrence as
wait/execute/signal batches over `W` fixed role commands and submits all
batches in one `vkQueueSubmit`.  Timeline semaphores carry Host-ready and
Device-done values.  A cold Host service owner may wait and signal those
semaphores for backing I/O, but no service turn calls `vkQueueSubmit` or a
backend epoch callback.

This lowering has fixed role, descriptor, page, and Authority state but
`O(Q)` transient `VkSubmitInfo`/timeline records.  It is one-submit GPU-owned
execution evidence, not device-generated-command or `O(1)` native-stream
evidence.  Noncoherent memory additionally requires a distinct transfer DAG;
the compute queue must never wait on work that can only be submitted behind
that wait.

## Metal lowering

The safe intermediate Metal lowering records one command buffer containing
the complete sequence of event wait, GPU descriptor gate, fixed-role ICB
execution, and done signal operations, then commits that command buffer once.
The existing MTL4 path that calls `waitForEvent`, `commit`, and `signalEvent`
for every coordinate is not eligible.  A cold Host service owner may observe
done values and signal already-recorded ready dependencies for backing I/O;
it may not encode or commit a coordinate.

This first lowering has fixed ICB roles and page rings but an `O(Q)` encoded
primary stream.  A later Direct fused/device-generated kernel may remove that
stream only after it preserves exact Authority descriptors, failure
suppression, and service visibility.  Merely unrolling `Q` ICB calls and
renaming the result a persistent kernel is forbidden.

## Failure and publication

The GPU publishes one aggregate terminal:

```text
success = completed Q, no failure
Known   = exact completed prefix + first failure + accepted no-write suffix
Unknown = known prefix + one UnknownMayWrite boundary
```

Known failure drains only already-recorded work through authenticated
no-write gates.  Unknown retains every reachable role, ring, prepared owner,
Authority generation, and backing recovery generation.  Neither outcome
invents per-coordinate callbacks.

Successful visibility is one transaction across both Pipeline banks,
Authority/Sliding Final, and the output backing.  All validation and native
rebase work happens before the first public mutation.  The final commit is
allocation-free and infallible and exposes both Pipeline generations, one
backing version increment, and recovery clear under one run-publication
authority.  No observer may see a successful Pipeline generation paired with
the prior or poisoned backing generation.

## Acceptance

Actual Metal and Vulkan product tests must run the bounded chunked `Q=2,3,5`
cases and prove:

1. one public handoff, `ceil(Q/2)` physical native queue submissions, and one
   aggregate native terminal;
2. zero epoch native submits and zero epoch scheduler callbacks;
3. `Q` GPU-authenticated coordinates and exact output;
4. one Authority accept, one Pipeline terminal transaction, one backing
   version publication, and one Final callback;
5. stale descriptor and terminal-loss `UnknownMayWrite` with no publication
   and sticky owner quarantine;
6. Known failure with an exact prefix/suppressed suffix and same-owner retry;
7. unchanged warm allocations and exact incremental retained/transient
   accounting;
8. CPU objects, hot branches, public sizes, semantics, and same-workload
   performance unchanged.

The backend-neutral fake `OneSubmit` contract already proves the typed portion of items 1
and 2 at Q=5,9,257: one native submit, zero epoch native submits, zero backend
epoch callbacks, Q exact service signals/waits/acknowledgements, Q simulated
GPU completions, and one Final after—not before—the last acknowledgement. It
also rejects stale descriptor identity, out-of-order service operations,
missing failed-admission reasons, success after the first failed admission,
and `tail_local_count` values of zero or wider than the final role. It does not
prove Metal/Vulkan synchronization, visibility, output, or publication.

The actual Metal and Vulkan service-aware contracts now prove items 1 through 6
through the public Virtual runner on the bounded Pointwise `BackendChunked`
Q=2,3,5 cases. Direct Q=5/9/257 `DeviceOps`-copy checks remain separate
backend evidence and are not substituted for this public product evidence.
Failure uses exact focused
coordinates. Every success case also ends in a physically short pointwise
page, proving that the sealed `ZeroInactiveTail` service completes the Host
frame without widening logical output or backing publication. The retained
coordinator Final additionally exposes the closed Authority-owned Sliding
receipt; together with the two exact Pipeline generation deltas and one
backing version delta, that is the non-mirrored exactly-once success oracle.
The test-local route requires, copies, and wraps the exact non-null production
product operations, delegates through them, then restores them before
returning. Success counters alone are intentionally separate from the
adversarial observer-atomic publication proof.

The separate DeviceVsm public-run contract proves the requested success subset
of items 1 through 4 at Q=5,9,257 without using the service-aware request. Its
centered Clamp/Clip U32 Window Sum/Min/Max cases prove the same one-submit,
zero-Host-service boundary at Q=5,9,257, exact output, and exact eliminated
physical overlap for every operation. They also prove the GPU-returned `Q-1`
canonical boundary transitions and exact footprint checksum for the short-tail
page geometry. The corresponding fused cases prove
the exact authored `Map(+3) -> Window {Sum, Min, Max} -> Map(*2)` semantics,
the independent canonical five-stage and maximum seven-stage
`Map -> Map -> Window -> Map -> Map` and
`Map -> Map -> Map -> Window -> Map -> Map -> Map` semantics, three-, five-,
or seven-stage proof identity, exact output, fixed retained controller storage,
and the same one-submit/one-Final boundary. Clip Sum authenticates its frozen
PrefixDifference plan; Clip Min/Max authenticate BlockPrefixSuffix. All three
execute their internal passes in the one DeviceVsm dispatch and reuse the
already admitted whole-run input/output staging. Its
pure unsigned U32/U64 Inclusive and Exclusive Scan and
Sum/CountNonzero/Min/Max Reduce cases prove
one native submit and payload dispatch, zero Host epoch submit/service/callback
control, an exact whole-run result, and a single publication. Its Scan and Sum
checked-overflow cases prove
an exact Known/no-write failed page, zero publication, and same-state retry.
The Scan suite additionally executes optimized
`Map(+7) -> {Inclusive,Exclusive}Scan`, canonical
`Map((x xor 0x55) * 3 + 7) -> {Inclusive,Exclusive}Scan`, and two-input
`Map(a+b) -> {Inclusive,Exclusive}Scan` at Q=5,9,257 on both actual backends.
It authenticates `stage_count=2`, the exact retained typed Map artifact and
parameter-byte contract, one through seven ordered resident inputs, one dispatch,
zero Host epoch control, exact output, and one aggregate Final. A focused
source contract also builds the exact seven-read maximum for Metal and Vulkan,
rejects a non-total recurrence and a mismatched scalar domain, and keeps
Vulkan workgroup-shared Scan state at module scope.
Its source proof is physically divided as well: `device_vsm/source.cpp` owns
canonical Map transformation plus direct/shared/fused Window source checks,
while `device_vsm/source/graph_map_reduce.cpp` owns typed U64 Graph Map/Reduce
artifact identity and rejection proofs. Their declarations-only `local.hpp`
seam reuses the one canonical artifact constructor and creates no shader-source
or semantic-model mirror.
Its exact Graph cases prove one-Map and fused two-Map canonical U64 prefixes
plus the public-input branch/fan-in semantic DAG.
Wrapping Add covers all four terminal reductions; non-summary
`xor/multiply/add` covers one- and two-Map Sum. The fused cases prove three
authored operations become two physical stages without another native submit,
Host service turn, or VSM intermediate;
both prove one aggregate publication of both stages, and checked overflow with
zero publication. Every case also proves the planner-projected frame capacity,
batch count, exact Map/terminal payload-stage ownership, phase-separated
dependency masks, exact GPU wavefront step count, and ordered trace; a trace
mismatch is not success evidence. It
requires the non-null production DeviceOps entries and fails if execution
falls back. Its exact oracle is `native_submit=1`,
`epoch_native_submit=0`, `payload_dispatch=1`,
`host_service_turn=0`, `host_epoch_callback=0`, `Final=1`, Q generated and
completed pages, exact output, two Pipeline terminals, one Authority accept,
and one backing version publication. The DeviceVsm-specific public Unknown
contract proves one native submission, one aggregate Unknown Final, zero
backing publication, retained recovery, and sticky same-owner retry rejection
on both actual backends. Performance remains open as stated above. The
same-state warm contract first
conditions one Q=9 run, then proves the next run uses `cold_prepare=1`,
`warm_rearm=1`, advances both Pipeline banks and backing version exactly once,
and is accepted as one producer-reported allocation-free sample. Process-wide
warm allocation and same-workload performance remain open evidence items.

The Direct pointwise Q=100,000 backend run and Q=100,001 public product run
additionally prove bounded retained controller state and a large-Q aggregate
transaction for that topology. They do not widen Window, Scan, Reduce, or
Graph admission. If an intermediate backend still retains `O(Q)` encoded
native records, that fact remains partial and separately metered. A 100x claim
requires independent same-workload measurement and is not implied by any
state-machine test.

## External architecture evidence

- Apple documents that Shared resources require the application to
  synchronize CPU/GPU access and that CPU writes must finish before the
  command buffer consuming them is committed:
  [MTLStorageMode.shared](https://developer.apple.com/documentation/metal/mtlstoragemode/shared),
  [Synchronizing CPU and GPU work](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work).
- Apple command buffers expose encoded event waits/signals between GPU passes,
  and Shared events provide the CPU/GPU synchronization boundary:
  [MTLCommandBuffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer),
  [About synchronization events](https://developer.apple.com/documentation/metal/about-synchronization-events).
- The Vulkan synchronization specification defines timeline Host
  signal/wait operations and explicitly warns that waiting in a command buffer
  for a future Host-set `VkEvent` has no portable stall period and is defunct:
  [Synchronization and Cache Control](https://docs.vulkan.org/spec/latest/chapters/synchronization.html),
  [Command Buffers](https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html).
