# Route Execution Topologies

This page owns the structural boundary between the implemented Direct
whole-run scheduler and non-Direct Virtual routes. It is an architecture
contract, not a claim that every non-Direct topology below executes.
Product availability remains owned by [Product](../product.md); native
whole-run implementation status remains owned by [Scheduler](./scheduler.md).

## One invocation, typed topology

`execution::Plan` is the implemented Direct formula. It must not accumulate
optional carry, accumulator, graph-stage, and scratch fields until it becomes
an unchecked union. A route-neutral invocation is instead the closed product
of one shared envelope and one typed topology:

```text
Invocation = {
  credential:  (identity, Authority token, generation),
  resources:   immutable logical materializations and physical regions,
  publication: one recovery/private backing generation,
  topology:    Direct | LocalWindow | Reduction | Scan | Graph
}
```

The topology owns its recurrence formula and dependency projection. The
envelope owns no page-selection or scheduling policy. A backend lowering may
consume a topology only when it implements that exact type; an unsupported
topology selects the existing truthful route before Authority begins.

No topology materializes an array of all logical nodes. It exposes bounded
templates and an algebraic projection keyed by global coordinates such as
`(epoch, stage, phase)`. Physical journals retain global coordinate tags so a
cyclic slot can never authenticate stale work.

## Prepared Virtual topology ownership

`VirtualRunProjection` freezes one `VirtualRunTopology` at its ordinary,
multiple-input, or graph projection boundary. Direct, LocalWindow, Reduction,
Scan, MultiPointwise, MultiScan, GraphPointwise and GraphReduction are exclusive
states. Graph/reduction/scan/poolless queries are derived from that value;
Scan inclusivity comes from the existing operation. There are no independent
`reduction`, `graph_execution`, `graph_reduction`, `scan`, `inclusive_scan`,
`multi_pointwise`, or `multi_scan` fields. Dispatch, epoch projection, admission,
cache publication and native preparation all consume the same topology.
Pipeline-owner selection accepts only this topology, instead of constructing
a mostly empty full invocation to carry two selection flags.

The physical DeviceVsm decision is separately owned by
`compute/virtual/run/device_vsm/route/proof.hpp`. Its proof contains a stamp
and exactly one value alternative: Direct, StagedLoop, WindowRing, or
GraphResident. Direct has no explicit endpoint; StagedLoop always means
Staged. Only WindowRing owns a `VirtualWindowPreflight`, from which its page
count, frame capacity and endpoint are derived. Other alternatives own their
own bounded shape; GraphResident also owns its endpoint. The proof has no
parallel route tag, endpoint, page count or default Window snapshot. Its
variant storage is inline, has no heap allocation, and is bounded to 120
bytes. Window byte arithmetic, nonwrapping capacity and host/device ownership
laws remain checked before native admission.

Probe constructs the alternative before side effects; the stamp authenticates
its exact value. Admission, cold/warm preparation, resident matching and
staging consume that same proof. The prepared handoff gets its page count
from the proof, with no second count field. Native credentials and Authority
leases retain their existing owners; a topology value does not authorize a
lease or publication. The stamp serialization remains unchanged, including
zero Window fields for non-Window routes, so this internal layout change does
not alter identity or numeric behavior.

The CPU route-selection contract exercises every proof alternative, invalid
shape, unsigned stamp, byte-overflow and storage mismatch. Native route
observers additionally require a changed stamp or valid-but-altered Window
snapshot to fail `proof_matches`; existing product contracts execute the
cold/warm routes and their failure/publication paths. This refactor changes
representation and consumers of existing routes; native availability remains
as specified in the topology sections below.

## Shared terminal algebra

All topologies share four terminal laws:

1. Host-service receipts authenticate only backing Input and Output work.
   They never prove a native Dispatch.
2. Native evidence authenticates only accepted native work. A backend cannot
   invent Host-service completion or backing publication.
3. A Known failure preserves the exact completed prefix, invalidates exact
   may-write resources, suppresses already accepted suffix work, and permits
   same-owner retry after all terminals are collected. The completed prefix
   and `may_write` are independent facts: a whole-run validation prepass may
   complete input epochs and fail before any output mutation.
4. `UnknownMayWrite` quarantines every resource reachable from the accepted
   native schedule. No public backing version advances.

One invocation publishes its backing generation exactly once after Authority
close. Per-epoch or per-stage completion is never public publication.

## Direct

The implemented Direct topology currently combines backing fetch and Device
promotion in `Input`. The continuous target splits them:

```text
ForecastFetch(e) -> HostReady(e)
HostReady(e) + Output(e-2) -> Promote(e) -> Dispatch(e) -> Output(e)
```

It has two physical banks and four logical `(bank, publication parity)`
roles. This is the only topology represented by the current
`execution::Plan`; the native Schedule and fixed-chunk Stream lower this same
formula without reconstructing page or cache policy. The split Forecast law is
owned by [Forecast](./forecast.md) and is not yet an implemented Plan phase.

For the public Virtual runner, an eligible ordinary all-staged unary
nonresident Pointwise invocation with `Q>=2` first probes `StagedLoop`. The
probe requires exact mapped Host-visible/coherent views for every input and the
output and carries one route enum into preparation; shape, tier, and
max-read predicates are not recomputed. An admitted StagedLoop owns one native
recurrence submit, `Q` GPU epochs, zero transfer submits/bytes, zero Host epoch
submit/service/callbacks, and one Final/Authority/output publication/version.
If mapped structural validation returns `BackendUnsupported` before any owner,
rearm, stage, lease, or native acceptance, the route cleanly declines to the
legacy service. An owner mutation, native acceptance, or non-capability
failure is terminal and cannot fall back.

Persistent Pointwise with the R2 capability remains `BackendChunked`: it uses
two physical payload slots and two-coordinate chunks for `ceil(Q/2)` native
submissions; Vulkan may retain four fixed authentication metadata cells in
addition, not four payload slots. `OneSubmit`/legacy fixed-W remains a
lower-level fallback seam, not the generic Window product. Persistent spatial
Window declines before lease when unsupported; dedicated bounded Window
(`Q<=4`) or Stream (`Q>4`) owns one logical handoff, one Final, and one output
version (the current Metal fallback may make `Q` physical queue calls).
Explicit/resident DeviceVsm Window remains a separate route. Neither the
StagedLoop nor the legacy service claims same-submit GPU ownership of generic
nonresident backing data.

### Finite WindowRingFused

The typed `WindowRingFused` proof is the sole route decision for a finite
centered-I32/U32 Window with a checked footprint. It accepts bare semantics
and exact authenticated, parameter-free `CanonicalTotalU32` Map chains of
symmetric depth 1–3 before and after the Window. Resident input/output or
fully all-staged input/output is authenticated before the Authority lease;
downstream preparation only consumes that proof. One physical dispatch and
one native submit execute `Q` seed epochs, `Q` compute epochs, and `2Q`
internal phases. Host epoch submit/service/callback counts and transfer
submits/bytes are zero, and the common authority emits one Final and one
publication/version. This bounded proof does not extend to I32 Map fusion or
the generic dynamic PagedLoop.
Resident backings, explicitly required DeviceVsm, Q1, scan, reduction,
same-stage-fan-in, resident or mixed-input Graph shapes, and unsupported
parallel-read shapes retain their existing routes. A hard bounded-plan admission first
validates a nonresident, non-required `GraphPointwise` with Q>=2 and one
serialized external read per stage. Host deferral consumes the dependency-driven input window or output-ring
capability defined by [Forecast](./forecast.md#ready-horizon). Cross-stage
reuse of an external input remains valid; same-stage external fan-in retains
its existing route. Neither proof changes the planner's dependency edges.
Both Host branches are still Host stage scheduling with one native
submit/wait per selected cell, not GPU-owned Graph recurrence.

The DeviceVsm Direct subset also accepts a sole-public-input/sole-output pure
total U32/U64 Map DAG after common slicing has composed it into one canonical
unbounded Map Program. Q remains the number of distinct backing pages, not the
number of authored Map nodes. Actual Metal/Vulkan Q=5/9/257 contracts prove
one native submit and payload dispatch, zero Host epoch control, exact branch
fan-in output, and one aggregate publication. This static composition does not
admit multiple backing inputs/outputs or implement the general Graph
ready-wavefront on the GPU.

### Bounded nonresident Window/Stream fallback

Unsupported Persistent spatial Window declines before lease. The dedicated
bounded Window fallback owns `Q<=4`, while Stream owns `Q>4`; together they
retain one logical handoff, one Final, and one output version. The current
Metal fallback may make `Q` physical queue calls, and its bounded input/output
service remains separate from the explicit/resident DeviceVsm Window route.
The centered-Clamp `Map(+3) -> Window -> Map(*2)` proof, exact frame/tail
authentication, and cold/warm ownership rules remain where that fallback
admits them. Other Window operations, aliases, additional Map nodes/chains,
multipass shapes, and dynamic nonresident continuations retain their existing
fallback or rejection contracts. The old `OneSubmit` checks remain lower-level
backend evidence only.

## Local Window

A finite Window inside one page batch is not a cross-epoch carry. Its outer
topology is Direct, but each native role also owns an immutable local-window
template and a resettable control state:

```text
Input(e) -> ResetLocal(e) -> WindowDispatch(e) -> Output(e)
```

Admission requires the backend to prove that replay resets every local window
state, that a suppressed batch executes no payload dispatch, and that the
Pipeline terminal publishes the exact verified prefix. Merely removing the
current `windows.empty()` guard would reuse mutable window state across roles
and is invalid.

The implemented centered Clamp/Clip U32 DeviceVsm subset also accepts either
the exact wrap-add/wrap-multiply immediate pair or one parameter-free
canonical-total single-read, value-write U32 Map expression DAG on each side
of Sum/Min/Max. The generic path retains both typed ParsedIR owners, injects
the prefix as a scalar function at every authenticated halo read, and injects
the suffix once at the logical output. The combined source still has one
payload dispatch and no Map intermediate. This does not admit parameterized
Maps, another Window or collective, stateful Window control, or the general
Graph ready-wavefront.

## Reduction

Reduction needs a run-owned Device accumulator, not Host folding hidden in a
native callback:

```text
Input(e) -> PageReduce(e) -> Accumulate(e) -> ReleaseInput(e)
Accumulate(e) -> Accumulate(e+1)
Accumulate(Q-1) -> Finalize -> OutputService
```

The accumulator is an Authority resource with `(operation, type, valid,
overflow, value)` identity. Min/Max carry a valid bit. Count and unsigned Sum
use a two-limb unsigned accumulator wide enough to reproduce the current
global overflow contract before final narrowing. Queue order fixes merge order
and therefore determinism. The final scalar alone enters backing recovery;
page partials are transient and can never be published independently.

The implemented pure unsigned U32/U64 Sum, CountNonzero, Min, and Max
`DeviceVsm` subset realizes this topology with one 256-lane workgroup over all
logical pages. Sum uses a two-word accumulator to prove the destination-width
total before the scalar write; U32 narrows from the exact U64 total and U64
rejects a nonzero high word. Overflow names the exact first failed page, is
Known/no-write, publishes nothing, and permits same-state retry. CountNonzero
adds one bit per element and the admitted `logical_elements <= UINT32_MAX`
bound proves both U32 and U64 destinations cannot overflow. Min and Max use
the unsigned maximum and zero identities over a planner-proved nonempty
extent. Signed, fixed, segmented, and other Reduce shapes retain their
established route.

Wrapping the current `VirtualReduction` Host merge in a completion callback is
not this topology and cannot count as GPU-owned reduction.

The private Metal/Vulkan `GraphResident` route is a distinct bounded native
topology for executions with at least two input bindings. It additionally
requires a nonterminal stage with at least two distinct
Internal/Intermediate/Transient read resources and one
Internal/Intermediate/Transient write resource. Its bounded table has three internal banked owners and four external
endpoint classes for the branch/fan-in shape: seven classes total, with
endpoints kept separate from owner bindings. The deterministic ready scan and
internal alias reuse wait for `DispatchComplete`; a planner-marked
`ExternalOutput` predecessor alone uses `ReleaseComplete`. One queue submit
records one physical GPU controller dispatch; the controller advances the sealed
`(batch,stage)` wavefront through `S*B` deterministic GPU steps. The logical
payload dispatch remains one, and one aggregate terminal is authenticated from
the planner proof. The resident form performs no Host epoch service or runtime
backing I/O and is not GraphPointwise. The resident shared sealed geometry is
bounded to `2<=S<=8`, `2<=R<=9`, `1<=P<=16`, `1<=O<=9`, `Q>0`, `C>0`,
`C<=Q`, `B=ceil(Q/C)`, and `S*B<64`, with checked U64/byte geometry and
all-Tile stages. Existing all-staged U64 Tile evidence remains the sealed
`S=5,Q=5,C=2,B=3` case with 15 steps, one submit/controller dispatch, one
logical dispatch, zero Host epoch service/callbacks, and one aggregate
Final/publication/version on cold and warm runs. The dynamic all-staged proof
also covers `S=5,Q=6,C=2,B=3` and its 15 steps. A multi-input staged endpoint
is admitted only when every input implements the public side `VirtualBackingReadCohort`
and returns the same non-null shared `VirtualReadCohort` provider with one
nonzero `VirtualCohortId` and a lane limit of 1..2; one provider joins the
complete input set in canonical order before the native recurrence, then the
runner reauthenticates each member. Providers do not retain descriptors or
reenter callbacks. A successful result is `joined=true`, has exact
`completed_bytes`, and uses `UINT64_MAX` for both failure fields; a joined
known failure reports the first member/page and completed prefix. Single-input
staged endpoints use the existing synchronous scalar `read_pages` contract and
do not require a cohort. For multi-input staged endpoints, missing or mismatched
capability declines to the ordinary route before preparation; identity
uncertainty after preparation is terminal and cannot fall back.
Resident endpoints bypass the cohort. One-input GraphPointwise remains on the
existing generic DeviceVSM/Host Persist owner path. All-staged endpoints require exact
authenticated fallback `AccelBuffer` handles, bytes, and versions; mixed
endpoints are rejected before mutation. Whole-run transfer is separate from
the compute submit. Generic callback-backed cohorts without this side contract
remain blocked by the lack of cross-input-wait-free reads, a cross-backing
concurrency budget, a callback no-reentry/cycle rule, and a dedicated bounded
cohort owner/join. TilePartial, AddSat, S=7 outside planner-sealed static
GraphResident/GraphPointwise shapes, schedules at or above the tile bound,
dynamic Forecast/Promote/Drain/Persist, and dynamic nonresident
GPU-native I/O remain unimplemented/blockers.
Any H2D/D2H submission used by staging is separate from that compute submit.

## Scan

The public atomic-backing exception is only the ordinary accelerator Scan
whose output covers its complete backing extent and dynamically implements
`VirtualBackingTransaction`. Admission binds the provider before epoch zero;
epoch writeback is shadow staging, and one Final performs provider prepare,
physical-cache/Pipeline preflight, and exactly one provider commit. Resident
outputs, outputs without the provider, Graph/Reduce/DeviceVSM routes, and
partial-extent or externally read backings remain on their existing
nontransactional or blocked paths. This contract does not establish an
atomicity boundary for an external reader that bypasses the backing/provider
authority.

Scan needs a run-owned Device carry and overflow state. The implemented pure
Unsigned U32/U64 Inclusive/Exclusive `DeviceVsm` subsets use one 256-lane
workgroup and avoid an `O(page_count)` local-scan scratch array:

```text
Input(e) -> LocalScan(e) -> ApplyCarry(e) -> UpdateCarry(e) -> Output(e)
UpdateCarry(e) -> LocalScan(e+1)
```

One wide precheck proves the whole sum fits before any output write. On
success, 256-element chunks execute in logical order and update a device-owned
carry. On overflow, the terminal names the first failed page, reports
Known/no-write, and discards the private recovery generation. The product
publishes only after the aggregate Authority and two Pipeline banks close.

The implemented composed subset may place one exact canonical total
element-local U64 Map expression DAG before the U64 Scan. Common admission
accepts one through seven public U64 reads and one terminal U64 value write,
which exactly fills the fixed eight-row resident table. It retains the complete typed
Map artifact and parameter bytes, and emits its nodes into the same payload
dispatch before the carry operation. It does not admit more than seven backing
inputs, data-dependent reads, another collective, or a Host-created Map
intermediate.

Common lowering composes an authored pure fanout/fan-in Map DAG into that one
typed Map owner before backend preparation. Actual Metal and Vulkan Q=5/9/257
contracts cover the one-input `(x+1) + (2*x)` form and the two-input `a+b`
form followed by both Inclusive and Exclusive Scan. They require one queue
submission, zero Host epoch submit/service/callback control, Q completed
pages, one aggregate Final, exact output, and exact input-major resident and
byte accounting. This is static semantic DAG composition; it is not runtime
GPU selection of ready Graph nodes.

The focused common source contract additionally builds the exact seven-read
sum expression for both operations and both backend APIs, while an eighth
public input fails cold at the Virtual pipeline capacity boundary. Actual
maximum-width device execution remains a separate acceptance surface from the
existing one/two-input Q=5/9/257 cohort.

Signed, fixed, segmented, and other unsupported Scan shapes retain the
existing Host `VirtualScan`, which reads output endpoints, updates carry, and uploads
injected values between epochs. Calling that code from a backend listener
would retain Host orchestration and is not a native Scan lowering.

## Graph

Graph topology owns a cold stage table and resource liveness formula:

```text
StageTemplate = (domain, ports, materializations, barriers)
Node          = (epoch, stage)
```

Within an epoch, stage `s+1` depends on stage `s`. Cross-epoch edges are
derived from exact physical resource reuse, not from a blanket epoch barrier.
Input service precedes the first backing-read stage; Output service follows
the last backing-write stage. Transient resources never cross the backing
boundary.

Authority admission must reserve the invocation's recurrent resource classes
and project each stage lease from the same `TiledGraphPlan`. Native lowering
records all stages for an accepted epoch with explicit compute barriers. The
current middle-stage `submit_execution` followed by `wait_execution` is a Host
stage scheduler and therefore remains outside this topology.

Retained common state is `O(resources + stages + roles)`. A backend may charge
additional native storage, but must report it before Authority begins. Known
failure evidence names the first `(epoch, stage)` plus one suppressed accepted
suffix; it does not retain one failure row per future stage.

Graph diagnostics follow the same bounded rule: the private run `FailLog` is
first-wins for `(epoch, batch, stage, phase, check, token, generation)` and may
carry one flat Authority `CloseInfo` snapshot. A later close/recovery snapshot
attaches only when that first context and credential match; it never replaces
the primary failure. This is provenance only; the existing Authority and Graph
terminal owners still decide rollback, quarantine, and publication.

## Transfer-queue boundary

Coherent Host-visible Input and Output can be serviced without placing a copy
behind a native queue wait. Noncoherent memory cannot use that law on the same
compute queue: a queued compute batch waiting for Host readiness followed by a
Host-issued upload on that queue is a circular wait.

A noncoherent topology therefore requires one of two explicit lowerings:

- integrate the copy command and its staging ownership into the already
  accepted native batch; or
- use a distinct transfer queue and timeline edges
  `HostFill -> CopyIn -> Dispatch -> CopyOut -> HostDrain`.

Host polling, a submit from a completion callback, or relabelling a blocking
copy as service evidence is not a native whole-run schedule.

## Implementation order

The dependency order is structural:

1. Local Window extends the Direct role terminal without adding cross-epoch
   state.
2. Reduction adds one serialized Device state and one final scalar publish.
3. Scan adds a serialized carry/overflow state and ordered page processing.
4. Graph generalizes native nodes to `(epoch, stage)` and transient resource
   liveness.
5. Noncoherent transfer adds an independent queue/copy topology.

Each step needs a public product success contract, Known retry,
`UnknownMayWrite` quarantine, exact backing-version oracle, warm retained
memory proof, and backend-produced handoff/batch/queue-call evidence before its
status can move from partial to implemented.
