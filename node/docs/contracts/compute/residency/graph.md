# Residency Graph

This page owns recurrent Graph demand, multi-port epoch leases, stage
execution, collective ordering, and the exact implemented/partial graph scope.
Physical extent compatibility belongs to [Pool](./pool.md).

Graph epoch admission is one Authority-gated transaction. The public lock,
CPU retry/pending handoff, state preconditions, and final token/generation
publication remain in `registry/graph_epoch/admission.cpp`. Its bounded
request/remap/materialization projection is owned by
`registry/graph_epoch/validation.cpp` (with declarations in its narrow
`internal.hpp`), deterministic hit/replacement
scoring and rectangular assignment by `assignment.{hpp,cpp}`, and
rollback-safe frame relocation/binding staging by `relocation.{hpp,cpp}`.
These phase records are fixed-capacity invocation scratch only; Authority
continues to own the sole frame table, epoch slots, gate, and retry journal.

Virtual Graph preparation is one ordered coordinator in
`virtual/graph/prepare.cpp`. Its compiled phase owners under
`virtual/graph/prepare/` are `validation.cpp` for request/page-map and route
contracts, `slices.cpp` for canonical graph-slice compilation checks,
`bytes.cpp` for page arithmetic and prefetch policy, `materialize.cpp` for
resource/remap/stage projection, `topology.cpp` for the dry physical plan,
`budget.cpp` for frame and Host-ring projection, and `assemble.cpp` for Pool,
Pipeline, CPU receipt, and `VirtualPipelineState` assembly. The private
`internal.hpp` carries only invocation-local value results; it is not a second
planner, cache, or publication authority.

## GraphResident source ownership

The public `source/graph_resident.hpp` boundary declares the artifact builder;
`graph_resident/model.hpp` keeps only artifact and stage layouts. The compiled
`graph_resident/build.cpp` owner performs proof admission and artifact assembly,
while `validation.cpp`, `digest.cpp`, and `helpers.cpp` own endpoint/stage
checks, canonical key digest, and shared source facts. Vulkan and Metal source
construction is compiled under their backend directories: `loads.cpp` emits
resource fragments, `stage.cpp` emits stage bodies, and `entry.cpp` assembles
the complete source. `internal.hpp` is declarations-only; no backend source or
build implementation is included through a header.

## GraphPointwise Metal source ownership

The GraphPointwise Metal source root remains the validation and composition
coordinator at `source/graph_pointwise/metal.cpp`. Its bounded compiled leaves
under `source/graph_pointwise/metal/` are disjoint: `binding.cpp` owns generic
MSL map and ring argument bindings, `fixed.cpp` owns the merged fixed-operation
helper emission, `chained.cpp` owns stage read/write value chaining, and
`page_ring.cpp` owns the PageMap and ring-page body. The local `internal.hpp`
contains declarations only; no executable source is included textually. The
coordinator preserves the original fragment order and exact emitted MSL bytes.

## GraphPointwise Vulkan source ownership

The GraphPointwise Vulkan source root remains the GLSL assembly coordinator at
`source/graph_pointwise/vulkan.cpp`. Its bounded compiled leaves under
`source/graph_pointwise/vulkan/` are disjoint: `binding.cpp` owns page-map and
ring/alias buffer declarations, `fixed.cpp` owns merged fixed-operation helper
emission, `chained.cpp` owns ordered stage-value emission, and `body.cpp` owns
the wavefront/page body, including mapped input copies, stage execution, and
output publication. The local `internal.hpp` is a declarations-only seam. The
coordinator keeps the existing validation and fragment order, so every
supported Vulkan variant retains byte-identical GLSL and artifact identity.

The backend-neutral GraphResident proof boundary remains in
`device_vsm/graph_resident.hpp` and contains only capacities, layouts, and
function declarations. Compiled proof owners under
`device_vsm/graph_resident/` are `type.cpp` for U32/U64 type validation and
codes, `identity.cpp` for shared-owner control-block identity, and `digest.cpp`
for the canonical digest order. `root.cpp`, `owners.cpp`, `resources.cpp`,
`stages.cpp`, `lifetime.cpp`, and `tails.cpp` own the ordered root/page-map,
owner-bank, resource/external-slot, stage-port, producer/lifetime, and unused
tail checks; `validation.cpp` composes them in the original fail-closed order.
Its declarations-only `internal.hpp` carries only bounded validation context;
there is no heap allocation or second proof authority.

DeviceVSM projection is physically split by phase. The public
`device_vsm/projection.cpp` is only the projection coordinator;
`projection/candidate.cpp` selects the request route and bounded topology
candidate, `projection/pipeline.cpp` owns exact pointwise/Graph-Map pipeline
matching, and `projection/validate.cpp` owns per-route geometry, chain, and
peer authentication. `projection/artifact.cpp` constructs the selected
GraphResident, GraphPointwise, GraphMapReduce, Scan, Reduce, or Window artifact,
retains the prepared owners, and seals the final `DeviceVsmProof`.
`projection/internal.hpp` contains only invocation-local candidate declarations;
the candidate is not retained as a second mutable authority.

The Authority migration/Graph contract evidence is split by ownership under
`node/tests/contract/compute/pipeline/residency/authority/`: `migration.cpp`
covers paired dirty Device-to-Host migration and overwrite exclusion;
`transform.cpp` covers role/tier, materialization, transient relocation, and
rollback; `capacity.cpp` covers concurrent frame admission; `lifetime.cpp`
covers retained output and missing-transient rejection; and `graph.cpp` covers
multi-resource Graph admission and atomic failure/recovery. The thin
`dispatcher.cpp` preserves the source order and return IDs 30–73.

## External page remap

The additive `virtual_pipeline(..., GraphPageMap, config)` overload is the
binding-view contract for nonresident `GraphPointwise`; `ResidencyConfig`
remains the two-budget descriptor. `GraphPageMap::graph_hi/graph_lo` must
match the Program. Each `GraphPageMapEntry` names a dense public input
ordinal, a target-local page, a source-local page, and `Begin` or `End`
origin. Preparation resolves that ordinal through the compiled
input-resource table and seals the canonical private resource ID; the caller
span is copied into the prepared plan and is never retained as a runtime
pointer. The plan accepts one full-page bijection per mapped
external input resource, with `K = frame_capacity <= 32` entries and an
aggregate checked bound of `TiledGraphPortCapacity * PipelineLeafCapacity =
512` entries. A mapped input must resolve to an `ExternalInput` with
`Backing` persistence, and every consuming graph port must be `Read`; mixed,
internal, output, transient, and `ReadWrite` resources are rejected.

For batch base `F = b*K`, active count `q = min(K, P-F)`, a target `t` reads
`F+s` for `Begin` and `F+(q-1-s)` for `End`. A nonempty template must cover
and uniquely name every frozen `t` and `s`; the tail is validated again before
the Authority mutation. `PageUse.key` is the resolved source page while the
sealed target order remains the binding order. All stages consuming that
resource therefore retain the same next-use, pin, prefetch, and ready facts.
Malformed, duplicate, missing, out-of-range, wrong-resource, wrong-origin, or
wrong-fingerprint descriptors fail before frame or Pool mutation. Empty maps
retain the ordinal schema and identity version 9; nonempty maps use the
canonical resource groups and fingerprint in identity version 10. Each group
has the exact serialized number order `resource, remap_count, target_local,
source_local, origin` for every entry; the public ordinal is not hashed.
Partial-byte, multi-source Assemble, internal/output/transient remaps, and
plain GraphPointwise execution without a sealed page map remain outside this
contract. Nonresident GraphPointwise now consumes the same proof-owned 824-word
map: active preparation requires zero parameter bytes, keeps the existing
descriptor count, and its Vulkan/Metal source resolves target-local pages by
the sealed row before the ordinary ordinal load. Resident
GraphResident additionally accepts the same nonempty resource-wide `S=1`
full-page map when the proof seals the map into its GPU binding.

The recurrent residency model is split along its dependency boundary under
`pipeline/residency/model/`: `base.hpp` owns page/access/transition vocabulary,
`stream.hpp` owns the one-resource stream plan, `graph.hpp` owns recurrent
graph topology and active invocation projection, and `plan.hpp` alone composes
either route into `ResidencyPlan`. The adjacent `model.hpp` is only the stable
include surface; it owns no parallel state or planning behavior.
Compiled behavior follows the same boundary: `residency/model.cpp` owns Stream
and active-invocation lifecycle/query methods, `model/project.cpp` alone
projects remapped per-stage `PageUse` records, and `model/dependency.cpp`
projects the sealed same/prior-batch predecessor list. These members operate on
the one `TiledGraphPlan` storage authority and introduce no copied graph.

Residency planning is split by its actual graph phases. The public Stream
overload is owned by `pipeline/residency/planner/stream.cpp`; the public Graph
overload in `planner/graph.cpp` orders the graph phase owners under
`planner/graph/`: `validation.cpp` normalizes resources/remaps and validates
stage ports, `liveness.cpp` seals first/last/writer and next-consumer facts,
`physical.cpp` performs interval coloring and physical-class assignment, and
`dependencies.cpp` seals same-batch and prior-batch wavefront edges. The
declarations-only `graph/internal.hpp` carries only the bounded planning
state and phase seams; the resulting `ResidencyPlan` remains the sole
planner-owned immutable artifact.

Physical coloring keeps the canonical compatibility/lifetime order and lowest
available color. Each resource visits its predecessor prefix once, marks
conflicting colors in a resource-capacity-bounded bitmap, then selects the
first unmarked color. External inputs still forbid sharing regardless of
lifetime. This changes the coloring scan bound from `O(R^3)` to `O(R^2)` without
changing physical IDs, compatible classes, or their byte charges.

Dependency sealing merges duplicate predecessor stages and their strongest
completion phase in bounded invocation-local arrays. After the combined
same/prior-batch capacity check, each immutable vector is assigned once.
Intermediate vector growth is not retained or charged, and the sorted edge
sequence and identity remain unchanged. Cold predecessor discovery itself
retains its existing liveness and alias rules.

The recurrent plan stores `O(resources + stages + ports)` cold topology and
projects a caller-owned K-bounded `PageUse` span. One forward liveness pass and
one reverse port pass are the sole next-use authority: backing pages pin only
their exact consumer epoch, while transient/output pages pin from their
producer through their last consumer. The reverse pass writes the next
consumer directly into each port; it does not scan future stages per use. The
former O(total uses) materialization and resource-by-stage-by-future-stage scan
were removed. Every stage and bank retains one cold-prepared Pipeline. One
multi-resource Authority lease binds all ports for that stage and publishes or
retires them atomically.

The same sealed liveness result also owns the Graph ready wavefront. Each
stage stores only its deduplicated resource predecessors: a transient read
joins its exact producer, the first use of a colored physical alias joins the
immediately preceding logical owner's last consumer, and the first use of a
physical owner in batch `b` joins that owner's exact last consumer in batch
`b-1`. Independent branches have no edge merely because their stage ordinals
are adjacent. The active invocation projects these sealed stage ids to
`(batch, stage, ordinal)` coordinates without scanning `PageUse`,
reconstructing liveness, or adding a Host-I/O dependency. Fan-in is bounded by
the cold port capacity, is included in residency identity schema 9 for the
ordinal plan (schema 10 when a remap template is present), and
planning fails closed if it cannot be represented. Every edge also seals its
completion phase. A transient/data dependency consumes `DispatchComplete`;
cyclic reuse of an external Output frame consumes `ReleaseComplete` and cannot
be satisfied by the producer terminal before Drain returns.

### Resident GraphResident PageMap

The resident GraphResident path may bind a nonempty, resource-wide `S=1`
permutation for each external input. Its proof-owned fixed image is 824 `u32`
words (3296 bytes): an eight-word header followed by eight 102-word rows, each
with `K <= 32` target/source/origin entries. Preparation deep-copies and seals
the canonical resource, external slot, `K`, page bytes, full bijection, and
the actual tail `q = min(K, P-F)` before any backend mutation. The mapped form
reuses GraphResident binding 0 only because its parameter bytes are zero; it
does not change descriptor count or create another topology owner. Vulkan and
Metal validate the exact serialized words, map digest, buffer capacity, and
warm proof before binding or rearming. Inactive maps retain the existing direct
endpoint path byte-for-byte. The only resident remap supported here is this
GraphResident PageMap; other resident DeviceVSM remaps remain unsupported.
Partial-byte, multi-source, and internal/output/transient GraphResident remaps
remain outside this binding. Plain GraphPointwise without a map remains the
byte-equivalent ordinal route; mapped nonresident GraphPointwise is the only
additional GPU binding-view route.
The public Graph product now has a bounded common ready-wavefront controller.
It retains exactly `2 * stage_capacity` cells for the two execution banks and
selects a ready cell by one fixed-array scan; there is no Q-sized heap or warm
allocation. Planner-sealed predecessors are the only dependency facts. A
typed Forecast terminal opens HostReady only; the cell becomes selectable
only after an exact Authority lease has either proved the binding
Device-resident or returned from H2D. Deterministic selection compares stage,
then global coordinate, then least resource id. The focused controller contract
proves that an independently ready later input is selected while an earlier
input remains blocked and that a fan-in cannot overtake either producer. A
middle Forecast is explicitly scoped to the requested current batch and to
non-prefix, nonterminal stages; its fixed `before()` ordering applies only
among those eligible cells, never to an earlier or future batch. The matching
middle Promote has the same current-batch, non-prefix, nonterminal postcondition
and never joins a future batch merely because its HostReady mask is complete.

The current source path admits up to seven public inputs for a multi-stage Host
Graph. The fixed nine-resource topology reserves one intermediate and one
output class. The service-free Graph Map-to-Reduce product
has actual six-input Sum/CountNonzero/Min/Max evidence on Metal and Vulkan in
addition to its one- and two-input cases. A given input may first appear at
the front stage or at one later stage, and a later stage may require multiple
exact backing inputs together. Same-stage external fan-in still uses sequential Host supply. Before DeviceVsm preparation, the ordinary accelerator
`GraphPointwise` route now first validates a bounded nonresident,
non-required graph with Q>=2, serialized external reads, one external read per
stage, and a sealed planner shape. Cross-stage reuse of one external backing
row is valid and covered idempotently; same-stage external fan-in, resident or
mixed rows, parallel reads, malformed plans, and unsupported topologies retain
their existing safe routes. Host deferral and the two-worker dependency-driven input window are owned by
[Forecast](./execution/forecast.md#ready-horizon). The output-capability branch
uses the ordinary fixed Drain-to-Persist banks. A graph with neither admission
proof remains on the DeviceVsm probe.
The public one-input branch/join `GraphPointwise` product proves the same
bounded ring through five pages, two simultaneous output callbacks, exact I/O
bytes, and one backing publication.
The bounded wavefront first selects the least planner-predecessor-ready exact
`(batch, stage, resource)` demand. A read-only Authority probe opens the fast
path only when every exact page is still Device-resident; the immediately
following stage lease reauthenticates those rows. On a miss, the stage
projection mints an exact typed Forecast, records HostReady only after its
terminal, and mints an exact Graph Promote against the destination stage
token. Promote release alone opens DeviceReady. A concurrent eviction
therefore fails closed; the probe remains a readiness fact, not a pin or a
second cache authority. Outside parallel Forecast admission, if the same fixed
Host bank holds a `batch+2` speculative receipt, the controller first waits
for its callback, terminals and releases that exact capability, reuses the
lane for current demand, and reschedules the displaced prefix
non-speculatively at its ready edge. It does not allocate a third lane.
The physical Authority seam now admits one aggregate Promote containing up to
seven exact sources. A successful Forecast callback is first retired into a
move-only `GraphReady`: this frees the bounded Forecast journal slot but keeps
the exact Resident Host row and planner `retain_until` pin. Promote
reauthenticates that row, pin, plan, resource, page, and destination before it
consumes any source. A failed aggregate issue consumes none of them. The
focused seven-source contract also proves that a different page cannot evict a
retained Host row before its consumer. The seam binds the disjoint Host rows
to the corresponding read ports of one still-Prepared destination lease and
moves the complete ready set into one move-only capability. Its terminal
authenticates every copied page, and its release activates the destination
exactly once. This closes the multi-source activation algebra.
The seven-input multi-stage Host
product runs on actual Metal and Vulkan with five pages, `K=2`, and a
seven-element tail. It proves thirty-five exact input-page reads, five output writes,
three two-stage batches, no DeviceVsm selection, exact output, and one backing
publication. The fixed overlap journal retains nine intervals: seven possible
source waits plus the already-admitted Drain/Persist pair. The dependency-driven middle-stage window overlaps at most two Forecast
callbacks and refills a free lane before the other lane returns. Same-stage
fan-in and arbitrary per-input worker creation remain outside that window.

This is a common Host controller. Every selected stage still performs one
Host-owned backend submit and one Host-owned completion wait/callback; general
native Graph wavefront lowering and GPU-owned multi-stage dispatch are not
implemented.

The CPU Graph epoch is the Direct/GraphPointwise Host route only. Its pending
reservation is serialized before `begin`, then moves through the exact
`Reserved -> Prepared -> Armed` receipt states; only the authenticated CPU
close can clear it. CPU-tagged leases are rejected by cycle and generic
`complete` APIs, and an unbound receipt cannot activate or resume an epoch.

The planner separately admits a final single-writer `ExternalOutput` as either
`Transient` or `Backing`. `Transient` remains the authority for the current
reduction partial: GraphDrain returns that page to Host and the deterministic
CPU fold consumes it. A `Backing` output may instead enter the fixed two-slot
Authority `GraphPersist` seam after an exact Host-output Drain. The two slots
are independent of the Device-to-Host migration lease, so Persist for one Host
bank may remain callback-live while Drain fills the other bank. Focused
contracts prove reverse callback order, terminal-before-release backpressure,
same-bank reuse only after release, absolute page/tail byte ranges, Known
no-write retry, strong plan lifetime, and sticky Unknown quarantine. A Known
success release retires the exact Host staging row because backing is again
authoritative; a Known failure restores the Dirty row for retry.

For CPU Graph persistence, that retry is represented only by the existing
Authority `CycleAuthorityState::graph_persists` LeaseSlot:
`Drain -> rollback -> RetryReady` keeps
the exact token, generation, plan, book domain, region, transition journal,
and undo rows. It is not mirrored in a Ticket or receipt slot. Before the
next CPU graph mutates Authority, one same-domain preflight changes every
RetryReady row's authenticated Dirty frames to Empty and clears all rows;
foreign domains, duplicate frames, aliases, or partial consumption return
Busy with no mutation. Known success alone retires and clears the row.
The two rows are bounded to 32 pages each (64 aggregate frame claims). A
terminal Unknown `Drain` row and a nonterminal `RetryReady` row are checked
together; cross-row duplicate frames, residual Free metadata, and any
transition/undo or identity mismatch fail closed before mutation.
The projected nonzero coordinate is a credential component shared exactly by
the Ticket, Authority row, and detached Unknown locator. Teardown snapshots
the fixed Book before taking the Authority gate, performs a pure row/claim
preflight and no-fail commit under that gate, then scrubs the Book after unlock.

That preflight uses one shared undo witness: every Drain/Unknown rollback sees
current Writeback against its exact prior Dirty frame, while RetryReady sees an
unchanged Dirty snapshot; all frame identity fields, Writeback transition
keys/ranges, Host/Output region bounds, and cross-row uniqueness are checked
before any rollback, including a partial issue journal before publication.
Direct rows carry an empty identity; CPU rows carry a valid identity whose plan
matches the slot. An authenticated Unknown with no CPU book domain clears
its ticket and remains a sticky Authority terminal; a CPU-domain Unknown keeps
the ticket as the Book's immutable locator for quarantine recovery.

The public `GraphPointwise` CPU/Host-fallback path consumes this seam for a
final Backing output. It is available after ordinary service-free pointwise
composition fails with an admitted unsupported/capacity result and every Map
stage can still be compiled as an exact bounded slice. The focused CPU product
uses three individually valid U64 Map stages whose composed expression exceeds
the 1024-node limit. For 73 elements, five pages, and `K=2`, it executes nine
stage epochs, writes five exact output pages including the 40-byte tail, then
publishes the backing once. A Known partial first-page write keeps the full
584-byte recovery extent and version unchanged; the same prepared state then
retries, retires the two cyclic Host banks, and publishes exactly once.
CPU persistence remains synchronous. The accelerator Host fallback owns two
fixed Persister workers whose request metadata is configured cold while the
payload remains in the Authority Host-output frames. `GraphDrain` release
opens the disjoint Device-output bank immediately; `GraphPersist` release
alone opens the corresponding Host-output bank. A backing is serialized by
default and may opt into at most two writes with `VirtualWriteLanes::write_lanes()`. The
actual Metal/Vulkan product contract forces five pages through the Host
fallback and proves two simultaneous callbacks, exact 40-byte tail/output,
same-bank `e+2` reuse only after callback return, and a Final join before the
single backing publication.

Metal and Vulkan have a narrower service-free accelerator alternative for an
exact acyclic graph. The one-input form admits two through eight stages. The
multi-input evidence covers two through seven external inputs at two stages
and the complete fixed nine-resource diagonal: six inputs at three stages,
five at four, four at five, three at six, and two at seven. Every stage must be a
parameter-free, canonical,
total U64 Map with one through eight Reads and one value Write. Operation
admission reuses the typed-Map authority for parameter-free canonical-total
scalar operations; GraphPointwise does not maintain a second Add-only
whitelist. The generated per-stage extension adds only parameter-free tiled U64
`AddSatUnsigned` on Metal and Vulkan. Signed saturation, U32 `AddSatUnsigned`,
Window/fixed/float forms, and whole-expression fusion remain unsupported.
Every nonterminal output must feed a later stage. Common projection
authenticates every retained stage artifact,
the exact planner-projected source of every read, each producer/consumer
physical view, the complete planner wavefront, and the whole input/output page
geometry. The generated payload keeps every stage value in a register and
substitutes the exact external input or earlier producer at each read; it
neither composes the authored IRs into the 1024-node Program nor materializes
an intermediate buffer. One
product handoff creates one native queue submit and one payload dispatch. The
submitted workgroup advances all Q pages and all authenticated stage
positions on-device, performs zero epoch Host submits, service turns, or
callbacks, and returns one aggregate Final. The Final address-orders and holds
every stage Pipeline state/publication gate while it publishes all stage
generations and the backing output once. Actual Metal and Vulkan public-product
contracts use the three-stage branch `input -> left`, `input -> right`,
`(left,right) -> output` and cover Q=5,9,257, a short final page, exact output,
every stage generation, and Known partial backing-write recovery with
same-state retry at Q=5. The same actual contracts now use a physical GPU
fixed-W ring: `4W` turn-state bytes plus
`(external_input_count+1)*W*payload_bytes` page scratch. Every page must
observe the exact prior turn in `slot=page mod W`; each external resident input
is copied into its distinct scratch frame, the fused Map-DAG binds only those
scratch inputs and its output scratch, and the final output is copied back to
the resident destination. Final checks `Q*(external_input_count+1)` physical
round-trips, the O(1) common page/slot/turn/tail checksum, exact ring bytes, and
exactly `Q-W` slot-reuse transitions. This authenticates
static service-free Graph page reuse and tail alignment with Q-invariant
native storage. Whole-run backings remain resident, so it does not add runtime
Forecast/Promote/Drain/Persist readiness to the GPU wavefront. A distinct
staggered three-stage contract binds one
new external resident row at each stage. It executes
`stage0(first) -> stage1(prior,second) -> stage2(prior,third)` and proves
Q=5,9,257, `3Q` exact backing reads, one native submit, zero Host epoch
control, exact output, and one aggregate Final/publication. The common Graph
I/O layout, rather than the first stage's authored bindings, owns all three
physical rows. Planner coloring keeps distinct external backing inputs in
distinct physical owner classes even when their stage lifetimes are disjoint;
the aggregate run freeze therefore authenticates both exact keys at once.
Transient intermediates continue to reuse disjoint physical lifetimes. A maximum-width
two-stage contract binds seven external rows at Q=5 and proves thirty-five
exact backing reads totaling 4,088 bytes, one payload dispatch, five
GPU-generated/completed page epochs,
zero Host epoch control, exact output, and the same aggregate transaction on
both backends.
A distinct deep contract binds six external rows to three sequential stages
at Q=5. It proves thirty exact backing reads totaling 3,504 bytes, one
payload dispatch, five GPU-generated/completed page epochs, zero Host epoch
control, exact output, and the one-Authority/three-Pipeline/one-backing Final
transaction on Metal and Vulkan.
A second depth contract binds five external rows to four sequential stages at
Q=5. It proves twenty-five exact backing reads totaling 2,920 bytes, one
payload dispatch, five GPU-generated/completed page epochs, zero Host epoch
control, exact output, and the one-Authority/four-Pipeline/one-backing Final
transaction on both backends. Its three continuation stages execute XOR,
unsigned wrapping multiplication, and addition, so this is actual evidence
for more than the former Add-only subset.
A third depth contract binds four external rows to five sequential stages at
Q=5, using nine planner resources: four inputs, four intermediate values,
and one output. It proves twenty exact backing reads
totaling 2,336 bytes, one payload dispatch, five GPU-generated/completed page
epochs, zero Host epoch control, exact output, and the
one-Authority/five-Pipeline/one-backing Final transaction on Metal and Vulkan.
A fourth depth contract binds three external rows to six sequential stages at
Q=5. It proves fifteen exact backing reads totaling 1,752 bytes, one payload
dispatch, five GPU-generated/completed page epochs, zero Host epoch control,
exact output, and the one-Authority/six-Pipeline/one-backing Final transaction
on Metal and Vulkan.
A fifth depth contract binds two external rows to seven sequential stages at
Q=5. It proves ten exact backing reads totaling 1,168 bytes, one payload
dispatch, five GPU-generated/completed page epochs, zero Host epoch control,
exact output, and the one-Authority/seven-Pipeline/one-backing Final transaction
on Metal and Vulkan. Both cases retain nine planner resources without adding
an intermediate allocation.

This `GraphPointwise` boundary is not a general native Graph scheduler.
More than eight stages, parameters, other scalar types, non-total or
unsupported operations such as integer division,
more than seven external inputs, multi-input shapes beyond the proved
seven-input/two-stage, six-input/three-stage, five-input/four-stage,
four-input/five-stage, three-input/six-stage, and two-input/seven-stage
boundaries, multiple outputs, cyclic or dead stage values, data-dependent
access, runtime Forecast/Promote/Drain/Persist readiness, and arbitrary
ready-wavefront selection remain on the Host path or fail closed.

The distinct DeviceVsm Scan boundary may statically compose the same class of
pure total U64 Map DAG with one through seven public inputs into one typed Map step
followed by one Inclusive or Exclusive U64 Scan step. Actual Metal and Vulkan
Q=5/9/257 contracts include the one-input fanout/fan-in expression `(x+1) +
(2*x)` and the two-input `a+b` expression, exact output and input-byte
accounting, one native submit, zero Host epoch submit/service/callback control,
and one Final/publication. The original authored DAG remains the
materialization identity; only its non-observable Map intermediates are
removed. This does not widen the common ready-wavefront into a native GPU
Graph scheduler.

The fixed maximum is seven because those inputs and the one public output
exactly occupy the eight DeviceVsm resident rows. Common source verification
builds a seven-read wrapping sum for both backend APIs and both Scan
operations; eight public inputs fail cold. The accepted actual-device cohort
still covers the one- and two-input forms, so maximum-width execution is not
claimed from source generation alone.

A second static boundary covers the same sole-public-input pure total U32/U64 Map
DAG when it has no collective. Common slicing composes the terminal expression
into one unbounded Map Program, and ordinary Pointwise DeviceVsm executes all
Q backing pages in one native submit and one payload dispatch. Actual Metal
and Vulkan Q=5/9/257 contracts cover `(x+1) + (2*x)`, exact logical output,
zero Host epoch submit/service/callback control, one aggregate
Authority/Pipeline/backing publication, and Q-invariant retained controller
bytes. The multi-input GraphPointwise boundary above is distinct
from this composed single-stage route. More than seven public inputs or more
than seven stages with multiple inputs for GraphPointwise, other unproved
multi-input stage/input combinations, more-than-seven-input or data-dependent Scan,
multiple outputs, data-dependent reads, later-stage backing service, and
runtime-ready node selection remain outside those static routes. The Graph
Map-to-Reduce route below separately admits up to six public inputs. A separate public
MultiPointwise DeviceVsm route admits two
through seven distinct same-type U32/U64 inputs and one output; it is one Map
stage, not Graph wavefront admission.

DeviceVsm proof storage, Authority registration, and Metal/Vulkan descriptor
projection now share one fixed-capacity typed resident set rather than separate
input/output scalar fields. Its canonical input-major/output-major order and
strong handles prevent an omitted intermediate or reordered descriptor row
from becoming a second physical authority. The Graph route below admits one
through six public U64 inputs and one scalar output. One-, two-, and maximum
six-input actuals cover all four supported reductions. Its
`ComputePlan` counts are checked against that exact set before preparation.

Separate MultiPointwise backend and product contracts exercise the same set
with two Input rows and one Output row at Q=5/9/257. Metal and Vulkan each bind
the three exact resident resources, compute an elementwise binary sum with one native submit,
perform zero epoch submits, Host service turns, or Host epoch callbacks, and
close one Authority aggregate Final. The public typed API independently proves
the two backing reads, exact sum, shared Pipeline-pair terminal, and one backing
publication. A maximum-width public Q=5 case additionally binds all seven Input
rows plus the Output row, proves `7Q` physical input pages and the exact
seven-input sum with one native submit/payload dispatch, and publishes the
backing once. This implements the narrow MultiPointwise product, but those
cases are not Graph evidence. The exact Graph-to-{Sum, CountNonzero, Min, Max}
boundary is proved separately below: one-, two-, and maximum-width six-input
actuals cover all four reductions. Multiple outputs and atomic multi-output
publication remain unimplemented.

A distinct service-free `DeviceVsm` route is implemented for one through six
public U64 inputs and one terminal U64
`Reduce::{Sum, CountNonzero, Min, Max}` after a canonical pure total pointwise
Map DAG. A linear Map prefix continues to own
its ordinary compiled Map Pipeline. When external-input fanout makes the
physical Host-wavefront topology wider, common Graph slicing additionally
composes the exact typed Map expressions into one strongly owned semantic
Program/Pipeline; it does not discard or relabel the physical stage table.
DeviceVsm authenticates the complete planner identity and projects the exact
physical frame capacity, batch count, and per-stage same-batch/prior-batch
`DispatchComplete` and `ReleaseComplete` predecessor masks into its immutable
proof. The CPU-common `virtual/run/device_vsm/identity.cpp` owner computes the
route stamp consumed by both route selection and accelerator admission; the
accelerator-only admission layer does not retain a second identity recipe. The
one submitted GPU workgroup performs a deterministic least-ready
traversal of that sealed topology and returns both the exact
`stage_count * batch_count` step count and an ordered trace before Final can be
accepted. The proof also names the exact lowered Map and collective stage
owners. For each batch, local lane zero writes the selected order into one
fixed eight-entry workgroup array; all lanes then execute the composed Map only
when that Map owner is selected and fold the batch partial only when the
collective owner is selected. Workgroup barriers make those selected phases,
not a separate Host loop or a precomputed ordinal loop, the actual arithmetic
order. It publishes the semantic Pipeline and terminal collective together as
the two publication owners.
The retained semantic admission owns the already-validated typed IR, not a
source-text summary. It requires the exact one-through-six Program-ordered U64
read bindings admitted by the route, one terminal U64 value write, and total
per-element operations; data-dependent reads and integer
division remain outside this route. Sum carries an additional high word so
public U64 reduction overflow remains checked; CountNonzero, Min, and Max use
their exact unsigned reduction identities. One Metal or Vulkan queue
submission dispatches one 256-lane workgroup; that workgroup traverses all Q
pages, evaluates the typed DAG value for each element, performs the
deterministic reduction, and emits one aggregate terminal.
The Host performs zero epoch submits, service turns, or epoch callbacks, then
commits both Pipeline publications and the scalar backing once. Actual
Q=5/9/257 product contracts prove a one-Map prefix with all four unary
reductions, a two-Map prefix with Sum, and a public-input fanout/fan-in
`(x+1) + (2*x) -> Sum` DAG. The distinct two-input cases stage two exact
backings and evaluate `(a xor 0x55) + 3*b` before each of Sum, CountNonzero,
Min, and Max; they prove `2Q` input pages and bytes, one native submit, zero
Host epoch service/callback control, one aggregate Final, two Pipeline
terminals, and one backing publication. The suite also binds the fixed-topology
maximum of six external inputs to each supported reduction at Q=5. Metal and
Vulkan prove thirty input pages, exact six-backing bytes and scalar output, one
native submit, zero Host epoch control, five GPU-generated/completed pages,
and the same one-Authority/two-Pipeline/one-backing publication transaction in
every case. Seven
external inputs remain valid at the lower-level single-stage Promote boundary
but cannot form Map-to-Reduce inside the fixed eight-resource Graph topology
because the intermediate and output consume the last two resource classes.
The suite covers both the former wrapping-add cases
and non-summary `xor -> multiply -> add` one-Map and fused two-Map prefixes.
The latter preserves three authored nodes while
exposing only two lowered physical stages, one native submit, zero Host epoch
service/callbacks, and one aggregate publication. A focused overflowing Sum
input proves Known `ReduceSumOverflow` with no Pipeline or backing publication.
Every Q=5/9/257 Metal and Vulkan Graph case additionally checks the projected
frame/batch geometry and the exact GPU-produced wavefront step/trace against
the same planner proof; a missing or mismatched trace is Unknown and cannot
publish.

The public `compute.virtual-residency-product` reduction contract keeps its
`CheckProductReduce` entry point in `virtual/product/contract.cpp`; its
semantic test owners are `virtual/product/reduce/{basic,sequence,lending,direct}.cpp`
with `support.cpp` holding only the shared fixture, canonical inputs, backing
model, and comparison helpers. The dispatcher preserves the original
basic → sequence → lending → direct order and all established return-code
routing; no leaf is a second graph or arithmetic authority.

The public scan contract keeps its `CheckProductScan` entry point in
`virtual/product/scan/dispatcher.cpp`. `scan/support.hpp` declares the shared
contract types, and `scan/support.cpp` owns the canonical scan oracle,
generation/control observers, and persistent backing fixture;
`basic.cpp` owns inclusive/exclusive cold and warm checks; `tiered.cpp` owns
the persistent tiered and wide-capacity checks; and `failure.cpp` owns
overflow atomicity, Unknown quarantine, and backing-poison recovery. The
dispatcher preserves the original basic → tiered → overflow → wide → Unknown
→ poison order and keeps one `DeviceVsmBypassScope` across the GPU cases,
resetting it before poison recovery.

The shared virtual-product route authority is split by responsibility under
`virtual/product/route/`: `support.cpp` owns the single observer protocol,
recording, callback lease, and restoration state; `observation.cpp` owns the
DeviceOps observation callbacks; `protocol.cpp` owns scope installation and
teardown; and `resolve.cpp` owns route classification and final observation
resolution. Existing `route.hpp` remains the public declaration surface, and
the route-selection, product, and graph-product cases all link these same
owners without a second routing policy.

The fusion semantic authority is the complete canonical `ParsedIR` already
admitted by common lowering. Graph Map-prefix compilation retains that strong
typed owner through backend preparation. DeviceVsm rechecks its exact artifact
identity, U64 shape, exact one-through-six-read/one-write binding layout, terminal
value write, and totality bit before emitting the typed nodes into the
reduction kernel. It
never reparses backend source or treats an opcode hash, an additive immediate,
or a partial IR pattern as semantic authority. The additive classifier remains
a diagnostic/optimization fact; it is no longer the admission boundary for
this DeviceVsm Graph route.

That narrow fused route bypasses the Host-ready cells and typed physical I/O
rings by requiring every whole input and the scalar output to be
GPU-addressable for the run. It consumes only immutable planner predecessor
facts. The GPU selects
their deterministic ready order, executes the two exact lowered payload owners
at their selected cells, and authenticates the aggregate trace. Canonical Map
intermediates are deliberately fused into the selected Map owner; this is not
an executor for arbitrary non-fusible physical stage payloads and it does not
consume runtime Forecast/Promote/Drain/Persist readiness. Later-stage backing
misses during native execution, more than six external inputs, multiple
outputs, stateful or non-Map DAG nodes, and independently bounded Graph Persist
remain outside this route.

The implementation keeps one authority per responsibility under
`virtual/graph/reduce/`: the thin `projection.cpp` facade preserves the
topology-to-runtime entry point, while `projection/identity.cpp` owns graph
identity/materialization and persist identity, `projection/effects.cpp` owns
lease effect capture/application, `projection/regions.cpp` owns physical
region/cache-key projection, `projection/stage.cpp` owns stage scratch
projection, and `projection/ticket.cpp` owns the ordered ticket projection.
The declarations-only `projection/internal.hpp` is the local seam; no leaf
retains a second Ticket or graph-plan authority. `authority` owns Graph
admission/discard composition, `lease` owns the exact
ticket-retained Authority views, `evidence` owns input residency counters,
`cleanup` owns per-ticket terminal disposition, `abort` owns whole-run failure
joining and close, and `transfer` owns controls
and Device-to-Device relocation. `middle/` separates ready selection, one-stage
Authority execution, and terminal-input binding. `output/` separately owns Device-to-Host request
projection, download, terminal, release, and final logical reduction consume,
`promote/`
separately owns Host-to-Device projection, copy, terminal, and release
orchestration, `collective/` owns terminal-stage Authority prepare/terminal,
and `prefetch/` owns the fixed two-lane exact external-input Forecast
lifecycle. Within `prefetch/`, front/later-stage projection, miss selection,
Authority issue and request construction, receipt acceptance, evidence,
replenishment, and cancellation have distinct implementation owners.
`timeline` owns overlap
accounting, `supply/` owns Projected-to-SupplyReady-to-PrefixReady composition,
`stage/` solely owns one-active-stage submit/wait/stats-fold serialization,
`result` owns the terminal value, and `wavefront/` owns the bounded ready-state
machine. The graph invocation seam is declarations-only in
`reduce/internal.hpp`: `coordinator.cpp` wires the borrowed owners,
`prepare.cpp` owns initial admission and the first Prefix submission,
`batch.cpp` owns one ordered Prefix/Middle/Collective/Output-or-Persist turn,
`failure.cpp` owns the single whole-run failure handoff, and `finish.cpp` owns
final retry/quiescence/publication checks. The public `reduce.cpp` file is only
the ABI-preserving reduction/pointwise facade. None of these phase owners
retains a second Ticket table or graph-plan authority. The stateless
`registry/graph_promote_owner.{hpp,cpp}` owns the
complete Graph Promote credential lifecycle: single and aggregate issue,
Forecast/GraphReady group validation, destination binding, terminal, and
release. It borrows Authority's one gate, frame/lease table, and quarantine
state without mirroring them. The Authority's `execution/graph_drain/` issue,
terminal, and release implementations are direct `GraphDrainOwner` methods;
the facet holds the exact Device-output to Host-output physical migration over
Authority's one gate. The product cannot compose unrelated source and
destination tokens. The graph coordinator is the only substantive invocation
orchestrator; the public `reduce.cpp` facade forwards to it and must not grow
duplicate projection, lease, transfer, or readiness policy.

Graph execution borrows each `EpochLease` directly from Authority's fixed
lease slot. Its bindings, transitions, ports, remaps, and relocations are
immutable from admission through activation and execution; admitting or
recycling another slot cannot invalidate them. A Ticket carries the complete
borrowed view, including its token and generation, and clears the entire view
when that epoch closes or rolls back. Effect capture and output retention must
finish before close; cleanup uses credentials and copied effect keys, never a
closed lease. Receipt/quarantine ownership remains with Authority.

Stage projection writes directly into the Ticket's bounded PageUse/request
storage and publishes active counts only after every port is valid. Inactive
capacity has no semantic meaning and is neither cleared nor read by projection.
Ticket initialization projects Terminal output identity first, then Prefix
into the same storage. After Prefix has joined and its lease has closed,
Middle reuses that storage. Terminal binding finally reprojects its stage
before admission. Each synchronous reader must finish before another stage
projection overwrites that same Ticket; Forecast workers retain their own
projected requests and do not borrow this scratch. Only epoch metadata, output
keys, and the small output projection survive a stage change.

Graph Promote groups likewise project one common invocation/batch/stage;
input sources retain only their own ranges into that projection. Source owner,
identity, coordinate, unique read port, ordered page subset, and exact physical
row authentication still apply independently to every source. Input evidence
consumes borrowed spans directly, without constructing a Ticket or copying lease tables. These changes add no heap
workspace, physical frame, or memory-budget owner. The product Graph fixture
executes public CPU/native runs and native failure/retry sequences on a 1 MiB
worker stack; fixed view/Group/Ticket bounds and cross-slot lease stability
have separate contract checks. Compiler frame measurements and the 512 KiB
Release reproduction live in the
[Graph scratch evidence](../../../../../docs/reference/performance/virtual/scratch.md).

The `registry/graph_persist_owner.{hpp,cpp}` `GraphPersistOwner` owns the
complete Host-output-to-backing Persist credential lifecycle and CPU retry
queries/recovery. Its issue, terminal, release, validation, and retry methods
borrow Authority's one gate, frame table, fixed Persist slots, and pending CPU
reservation. The quarantine source is split into snapshot, validation,
commit/scrub, retry, and bounded `RetryTxn` owners; these are direct
implementations over Authority state, not forwarding adapters or retained
mirrors.

Authority release validation has one declaration-only seam in
`registry/release_check.hpp`. Compiled owners under `registry/release_check/`
separate physical row/region validation, Graph lease and Persist validation,
execution-journal validation, and cycle/final admission. They all inspect the
same Authority state through the single friend seam; no inline header copy or
second release policy exists.

The graph-promote contract test keeps its public `CheckGraphPromote` entry in
`pipeline/residency/graph_promote/dispatcher.cpp`. The shared
`support.cpp` owner contains the typed Forecast/Promote terminal projections;
`wide.cpp` owns the maximum-source-width contract, `group.cpp` owns grouped
multi-input promotion, and `single.cpp` owns the single-resource multi-stage
lifecycle. The
dispatcher preserves the original wide → grouped → lifecycle order and the
`50 + wide` / `100 + grouped` failure offsets.

Diagnostic failure provenance has one fixed `FailLog` in the run's
`VirtualPipelineState`. It is reset only after the existing admission gate and
records the first `(epoch, batch, stage, phase, check, token, generation)` plus
any flat Authority `CloseInfo`; it owns no rows and does not alter rollback,
quarantine, or publication policy. Authority retains the physical close
snapshot, while Graph owns only the copied diagnostic record. A later
close/recovery snapshot attaches only when that first context and credential
match; it never replaces the primary failure.

The bounded common wavefront keeps separate Dispatch-terminal and Device-bank
release histories. Transient consumers may therefore advance after Authority
accepts the exact Dispatch effects, while a prior-batch external Output frame
remains unavailable until its D2H Drain callback has returned and Authority
has released that exact Device row. The independently owned Host row remains
callback-live through Persist and gates only reuse of its same Host bank. The
adversarial controller contract proves the `ReleaseComplete` dependency, and
the product Persist-ring contract proves that the other Host bank may be
written concurrently. This phase split belongs to the common Host controller;
it is not evidence for a native GPU-owned Graph scheduler.

For each epoch, the planner canonicalizes duplicate access, contiguous dirty
extent, least later use, pin interval, ready epoch, and prefetch epoch. Demand
above capacity and fragmented same-page writes fail closed. The anchor port
selects one physical local per logical page; all other ports must execute at
that same local. Execution dispatches the arbitrary locals in the lease,
including a non-prefix final batch.

The implemented public graph route is same-type U64 pointwise Map stages
followed by Sum. Its Host fallback executes actual Prefix and Collective
stages, including a four-stage public-input fanout `1R->1W`, `1R->1W`,
`2R->1W`, `1R->1W` chain. For five pages, K=2, and a short final page, the CPU
contract executes 15 physical epochs. Metal and Vulkan instead select the
service-free semantic owner, traverse the three page batches in one native
submission, perform zero Host epoch submits/service turns/callbacks, and emit
one Final/publication. All three routes read the 584-byte backing input exactly
once and publish the exact scalar once. In the Host fallback the second branch
consumes the Authority-proved resident input; a `fetch` binding at that stage
is rejected instead of being relabeled as a hit. Backing Input and Output plus
Transient Intermediate identities share the one Authority. Transient partials
are migrated where required, folded in deterministic logical order, and
discarded; they cannot become backing traffic.

The separate `GraphPointwise` Host fallback retains every non-fused Map slice
as a physical Graph stage and terminates in the Backing output described
above. It preserves exact public semantics for shapes outside the narrow
single-input two-through-eight-stage service-free DAG contract. The implemented U64
contract may still use DeviceVsm when the composed expression exceeds the
static Program bound because it retains every stage authority and passes each
exact planner-projected value in a register. Neither route turns the general Host
wavefront into a GPU scheduler.

The `graph_wavefront_host` product contract keeps one substantive public
coordinator and places its semantic owners in compiled siblings: `program.cpp`
owns the authored Map recipe and slice topology checks, `fixture.cpp` owns the
bounded values/backings and reversed callback gate, and `evidence.cpp` owns the
Vulkan admission/replay proof, output/accounting checks, and failure evidence.
`internal.hpp` carries only constants, fixture layouts, and declarations; it is
not a second workload, oracle, or state authority.

A maximal authored linear/internal prefix of pure Map steps is compiled into
one independent ProgramState before tiled Graph planning. Its internal values
never acquire a VSM resource, page identity, pin interval, or backing
materialization. A Map that rereads a public input through external fanout is
not part of that linear prefix: it remains a distinct physical stage so the
planner owns its later next-use, pin interval, and ready edge. For an internal
Map-prefix followed by one collective stage fusion changes recurrent stage
demand from `3Q` to `2Q`; it does not fuse across external fanout, a collective,
stateful, profile, or publication boundary. The DeviceVsm two-Map-to-Sum actual
contract is the product evidence for this exact `3Q -> 2Q` Graph fusion case. Both
wrapping-add and typed `xor -> multiply -> add` prefixes are covered; the
one-Map contracts independently cover Sum, CountNonzero, Min, and Max. The
separate Virtual Window topology can fuse one parameter-free canonical total
U32 single-read/value-write Map DAG before centered Clamp/Clip Sum/Min/Max and
one after it; the immediate `+3`/`*2` pair remains a smaller fast path. Both
typed Map admissions are retained and emitted into the same Window payload.
That one-dispatch Window proof is not a Graph ready-wavefront or arbitrary
multi-resource DAG fusion claim.

The sliced compiler has one physical owner per phase under
`node/src/compute/graph/compile/slice/`: `source.cpp` validates the canonical
reduce and pointwise shapes, `stage.cpp` owns controlled Map stage and prefix
construction, `reduce.cpp` owns the terminal reduction Program,
`resources.cpp` is the sole referenced-resource classifier, and `tiled.cpp`
and `pointwise.cpp` own their distinct ordered assemblies. The shared
`internal.hpp` is declarations and the immutable `SliceSource` view only; it
contains no lowering implementation or second graph state.

Focused contracts cover multi-resource producer fan-in, independent adjacent
branches, reverse Host-ready callbacks, deterministic selection, dependency
non-overtake, exact K=2 short-tail coordinates, interval-colored
physical-owner handoff, and exact prior-batch last-consumer reuse. The Forecast
contract additionally issues current and future Host-bank leases concurrently,
returns the later callback first, proves terminal-before-release cannot reuse
the earlier bank, and then proves exact tail/frame reuse after release.
The product ready-wavefront
consumes these topology/Authority facts. The exact service-free Map-to-Reduce
route above now proves a GPU traversal of the same sealed predecessor masks,
but a native lowering that executes every general physical Graph stage and its
typed I/O at those ready selections, plus its throughput, remains a separate
evidence requirement.

Two fixed execution tickets consume the plan's supply distances. Persistent
backing may prepare `e+1` and `e+2` front-stage input-zero Host pages while
Prefix/Collective work for `e` executes. A deferred first or second input is
serviced sequentially at its exact ready edge through the same bank worker.
Each worker
carries an Authority-minted move-only Forecast
capability rather than treating its raw correlation token as authority. That
capability retains the full 128-bit plan owner, exact projected use/key/range,
and physical Host frame. Callback terminal authenticates the receipt but does
not free the frame. Callback-return retirement converts a successful receipt
to `GraphReady`, frees its fixed Forecast slot, and leaves the exact Host row
Resident and pinned through its consumer. The product then moves the complete
ready set into one
Authority-minted Graph Promote capability which joins the exact still-Prepared
stage lease, disjoint per-input Host source frames, Device target frames, page
identities, and full frame byte counts. Promote terminal is recorded only after
the H2D call
returns; its release alone retires the Host lease and activates the Device
stage lease. Thus two banks may complete backing callbacks out of order, while
same-bank reuse remains blocked until the exact capability release. Output D2H
similarly uses one Authority-minted move-only Graph Drain capability that owns
the exact Device source frames, Host destination frames, page identities, and
byte counts.
Terminal authenticates the complete D2H receipt; only callback-return release
commits the migration or discards both sides on failure. Logical fold and
Transient discard terminate before that Device bank is exposed as
`ReleaseComplete`. Backing persistence then runs in the separate two-slot Host
ring. The opposite Host bank may remain callback-live concurrently, while
same-bank `e+2` projection joins the exact prior Persist release. A speculative
`e+2` Host receipt cannot be omitted merely from a Device probe because no
execution token yet prevents intervening eviction; if the hit survives, stage
admission reports `fetch=false` and H2D is elided.

Graph prefetch stage execution is divided by semantic authority:
`prefetch/stage/project.cpp` owns immutable stage/lane projection,
`prefetch/stage/cpu.cpp` owns CPU backing supply, `prefetch/stage.cpp` owns
one-stage Authority acquisition and activation, and `prefetch/stage/refill.cpp`
owns bounded dependency-driven lane refill and rollback. No leaf reconstructs
another leaf's receipt or cache-key policy.

The static eligibility proof is owned by `graph/reduce/parallel.cpp`; the
reduction coordinator consumes its result without duplicating the sealed
predecessor proof. The window and completion-wait laws have one owner in
[Forecast](./execution/forecast.md#ready-horizon).

Window footprints include the exact clamped halo. Canonical page identity,
overlap pinning, tile alignment, and fusion legality are owned by
[Footprint](./footprint.md). Hierarchical Scan and fixed
Reduce trees are describable, but the executable general graph surface is not
complete. General typed collectives, Window/Scan stage domains, multiple
collectives, multiple external outputs, multiple logical backing
intermediates, Clip pre-Map, Reduce post-Map beyond the documented U64 route,
Sort/Gather/Scatter, and recurrence remain unimplemented. Source-private
cross-type/role arena tests do not widen this public scope.

The private Metal/Vulkan `GraphResident` topology is the bounded native Graph
route and requires at least two input bindings. Its necessary family
discriminator admits either a nonterminal stage with at least two distinct
Internal/Intermediate/Transient read resources and a single
Internal/Intermediate/Transient write resource, or first-stage external-input
fan-in followed by a strict single-producer/single-consumer linear chain ending
at the sole external output. For the branch/fan-in shape, its seven physical classes are three
internal banked owners plus four external endpoint classes (three inputs and
one output); endpoint classes are not owner bindings. It performs the
deterministic ready scan and internal alias reuse on the device, then emits one
aggregate terminal/publication for the cold run and matching warm rearm. The
`A→X, B→Y, X,Y→Z, C→W, Z,W→O` shape uses `DispatchComplete` before the internal
X owner is reused by W. Only a planner-marked `ExternalOutput` predecessor
uses `ReleaseComplete`. Its resident and all-staged endpoints use the same
sealed dynamic geometry proof. One 256-lane workgroup traverses the pages;
same-stage pages use disjoint `(bank,slot)` pairs, and same-bank reuse occurs
only after the batch barrier. Stage selection/completion, batch, final, and
Host-visibility barriers remain, while page fences are fused from O(S·Q) to
O(S); this contract is not generalized to multiple workgroups. The existing
Q=5, C=2 evidence remains
covered, including the five-page tail and exact output/publication invariants;
the all-staged product also proves S=5, Q=6, C=2, B=3 with the same 15
deterministic `(batch,stage)` GPU steps, one compute submit, one physical GPU
controller dispatch, one logical payload dispatch, zero Host epoch
service/callbacks, and one aggregate Final/publication/version. Its resident
form performs no runtime Forecast, Promote, Drain, Persist, Host epoch
service, or backing I/O.
The admitted execution families are the branch/fan-in shape above and a
first-stage external-input fan-in followed by a strict single-producer,
single-consumer linear continuation ending at the sole external output. The
planner-sealed proof supplies the topology and liveness witnesses for both;
the bounded depth-seven, two-input diagonal is a GraphResident route with a
Metal/Vulkan cold-and-warm evidence contract. Wide external-only fan-in
outside that structural family retains the generic GraphPointwise owner.
The depth-seven test's evidence ownership is physically split:
`graph_pointwise_depth_seven/evidence.cpp` captures generic and resident run
observations, while `evidence/validation.cpp` owns immutable route, proof,
native, statistics, continuity, and failure-diagnostic validation. Both consume
the one `ResidentObservation` model in `internal.hpp`; neither owns a second
runtime evidence or hash state.
GraphPointwise remains a distinct sliced/Host or service-free route; sparse
Forecast/Promote/Drain/Persist Graph execution remains outside this mode. The
resident sealed-geometry predicate admits `2<=S<=8`, `2<=R<=9`,
`1<=P<=16`, `1<=O<=9`, `Q>0`, `C>0`, `C<=Q`, `B=ceil(Q/C)`, and `S*B<64`,
with checked U32/U64 byte products and only parameter-free, all-Tile
stages/resources whose single sealed root is one matching unsigned U32 or U64
type. The existing wavefront/resident proof remains the final compute-shape
authority for both endpoint modes. Its product evidence must exercise the
unchanged branch order with a non-page-multiple active tail for both typed
roots, and must prove one native submit/dispatch, one Final/publication,
zero Host epoch service/callback/backing I/O, exact output/hash, and warm
proof/type identity. Reduction remains U64-only; this boundary does not claim
dynamic paging or a nonresident GPU page service. Multi-input staged admission
additionally requires every input to implement the public side
`VirtualBackingReadCohort`, return the same non-null shared
`VirtualReadCohort` provider, and authenticate one nonzero `VirtualCohortId`
with a lane limit of 1..2. The runner retains that provider through exactly
one canonical-order synchronous `read_cohort` join, then reauthenticates every
member's provider, ID, lane, backing ID/version, and page geometry. Providers
do not retain descriptor spans or reenter member callbacks. Its result reports
`joined=false` before a complete join; a successful result has `joined=true`,
`completed_bytes` equal to the full request, and `failed_member`/`failed_page`
equal to `UINT64_MAX`. A joined known failure reports the first canonical
member/page and the completed prefix. Single-input staged endpoints retain the
existing synchronous scalar `read_pages` contract and do not require a cohort.
For multi-input staged endpoints, missing or mismatched capability declines to
the ordinary Host route before preparation; after preparation begins, identity
uncertainty is terminal and cannot fall back. Resident endpoints bypass the
cohort and use direct authenticated bindings. One-input GraphPointwise remains
on the existing generic DeviceVSM/Host Persist owner path. All-staged endpoints
still require exact authenticated fallback `AccelBuffer` handles, bytes, and
versions; mixed resident/staged endpoints are rejected before mutation.
Whole-run transfer is separate from the compute submit. Generic callback-backed
cohorts that do not provide this side contract remain blocked because they lack
cross-input-wait-free reads, a cross-backing concurrency budget, a callback
no-reentry/cycle rule, and a dedicated bounded cohort owner/join. TilePartial,
AddSat, schedules at or above the tile bound, dynamic
Forecast/Promote/Drain/Persist, and dynamic nonresident GPU-native I/O remain
unimplemented/blockers.
Any H2D/D2H submission used by staging is a separate whole-run transfer and is
not counted as the GraphResident compute submit.

The same authored workload has a separate CPU Host evidence fixture. It is
intentionally classified as `GraphPointwise`/`CpuRolling`: the CPU path has no
GraphResident owner, DeviceVSM/Persistent/Window callback, or command submit.
Its receipt book is preallocated before Authority use; Authority's fixed
`CycleAuthorityState::graph_persists` LeaseSlot is the sole retry/journal
authority. The Book may
retain only a fixed, detached Unknown credential snapshot after an Unknown
result; it is not a move-only `GraphPersist` or retry mirror. Exact prepared
identity is validated before retry consumption, while an Unknown credential is retained by the sticky
Authority-owned CPU quarantine.
The prepared identity carries the lossless topology `(hi, lo)` pair alongside
all backing/version and geometry fields; it is never reduced to an XOR hash.

The DeviceVSM geometry contract is an include-only umbrella at
`accel/kernel/residency/device_vsm/geometry.hpp`. `geometry/model.hpp` owns the
plain page, ring, result, and Window-footprint values; `geometry/page.hpp` owns
page validity, projection, and overlap arithmetic; `geometry/ring.hpp` owns the
checked ring schedule, storage, dispatch, and result formulas; and
`geometry/window.hpp` owns the centered-Window footprint checksum. These
facets consume the same geometry value and do not cache or mirror a runtime
plan.

CPU epoch ownership uses a flat residency key, not a graph-enum or pointer
identity: `{Authority-owner, book-domain, role-slot, nonwrapping nonce}`.
Authority-owner is immutable and nonwrapping, so zero, exhausted, foreign,
and duplicate owners fail closed before mutation; the shared Book is never
moved or copied, preventing address/domain ABA. A slot is `Free -> Reserved(key) ->
Prepared(key, token, generation) -> Armed(key, token, generation) -> Free`.
Authority holds exactly one pending CPU key during the serialized begin
attempt; a successful begin moves that key into its exact LeaseSlot before
pending is cleared, while other already-Armed role/bank slots remain
independent.
`begin_graph_epoch` receives the sealed key and publishes the exact returned
token/generation into `Prepared` before the central bridge asks Authority to
confirm the tuple. A retry uses `begin_cpu_graph_epoch_retry`: after pure
graph/shape/wavefront/projection validation, the Authority gate preflights the
full prepared identity and all RetryReady rows, then one no-fail commit clears
their Dirty frames and publishes the first epoch. There is no standalone
domain-only retry consume. Only an authenticated CPU close/abort may clear
both Authorities; generic `complete` rejects CPU-reserved slots. Failed begin
cancels the reservation, while a failed confirm/finalize or stale close keeps
the key and credential for recovery/quarantine. This is the ownership
contract, not a claim of a new product PASS. Unknown termination retains this
same Book with raw Pipeline/Pool identity and a temporary self-reference,
calls Authority retain before dropping stack handles, and drops self only
after successful retain; a valid lifecycle makes retain rejection
unreachable. A `Free` slot may retain its one
quiet bound receipt handle for ticket reconstruction, but `idle` rejects any
live permit, active key snapshot, or credential; authenticated clear/reset_cred
preserves that handle, while explicit detach, final reset, `drop_handles`, or
receipt destruction invalidates exactly that handle before rebinding. State teardown takes the
Pool execution gate before Authority discard; the discard authenticates every
quarantined CPU epoch and GraphPersist row, checks frame claims/undo/duplicate
conflicts and native quiescence, then performs the no-fail rollback/invalidate
commit. A failed preflight terminates rather than dropping a live credential.
