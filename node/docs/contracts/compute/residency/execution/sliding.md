# Continuous Sliding Execution

This page owns the backend-neutral controller that consumes
[Forecast](./forecast.md). It replaces neither page policy nor typed Graph
topology. It is the hard-cut target for removing fixed-window barriers,
Q-proportional native storage, and per-epoch Pipeline lifecycle work.

Status: the fixed-state transaction model and a policy-free fixed-W raw
transport are implemented and covered by focused model contracts. A distinct
backend-neutral persistent-product request, Control, typed backing-service
rows, capability, aggregate evidence, and validation seam is also checked in.
Its fake-backend `OneSubmit` contract proves at
Q=5, Q=9, and Q=257 that the request has no `Project`, `Release`, or
`Returned` member, one logical handoff owns all Q recurrence steps, backend epoch
submit/callback counts are zero, exact ready/wait/ack service counts are Q,
and exactly one aggregate `Final` follows the Qth acknowledgement. Metal and
Vulkan now lower this request and actual-adapter Q=5/9/257 contracts run the
accelerator coordinator through test-local wrappers around the exact
`DeviceOps` product slots. Those public-run contracts prove exact output and
the success evidence owned by [Persistent](./persistent.md). Production route
selection is active in the production `DeviceOps` table; the fixture asserts
both original slots are non-null, delegates through them, and restores the
original table unchanged. Ordinary all-staged unary nonresident Pointwise
Q>=2 first probes `StagedLoop` when exact mapped Host-visible/coherent input
and output views exist. Its route enum is carried into preparation without
recomputing shape, tier, or read-lane policy; an admitted run owns one native
recurrence submit, Q GPU epochs, zero transfer submits/bytes, zero Host epoch
submit/service/callbacks, and one Final/Authority/output publication/version.
A pre-lease structural `BackendUnsupported` with no owner, rearm, stage, lease,
or native acceptance cleanly declines to the legacy route; every other failure
after mutation is terminal. Pointwise Persistent R2 fallback uses the
`BackendChunked` Q=2/3/5 contract; unsupported Persistent spatial Window
declines pre-lease to bounded Window (`Q<=4`) or Stream (`Q>4`), while
explicit/resident DeviceVsm Window remains separate. The direct Q=5/9/257
checks below are legacy `OneSubmit` backend evidence only, and neither mode
claims same-submit GPU ownership of generic nonresident backing data. The
transport consumes immutable Authority-selected locals and masks and owns no
page, victim, or Graph policy. Direct Host Fetch now has an Authority-minted
bank-local physical frame capability carrying the exact canonical CacheKey,
backing offset/bytes, frame target offset, and frame extent for complete-frame
reads; a bound controller cannot call the model-local hit/victim entry.
Pointwise Direct additionally seals `ZeroInactiveTail`: the last backing read
retains its exact logical byte count and the Host service deterministically
zeros only the inactive suffix before publishing the complete HostReady row.
Centered Clamp stride-one Direct Window now seals the scalar-aligned boundary
recipe independently from the backing slice. `Clamp` repeats the exact first
or last resident scalar, while `Clip` fills the operation identity; neither
fill byte is counted as a backing read. The Host service performs that fill
before publishing the complete HostReady row.
Direct Promote now binds either exact one-to-one Host sources or one centered
Window `K+2 -> K` canonical Assemble transaction, plus Device input and
pre-reserved Device/Host output rows, in one Authority-minted move-only
capability. The Window form carries every source frame, expanded target
recipe, and exact slice; cross-bank canonical reuse stays pinned until the
callback-return release. Its terminal records transfer evidence while release
alone publishes DeviceReady. Direct full-frame Native, Drain, and
Persist now use the same Authority-minted move-only terminal/release law and
are covered through the one-use two-phase Final for K=1 and K=2. The
accelerator-only Direct coordinator now joins these capabilities to the
persistent-product transport and owns one-run Pipeline/Authority/backing final
ordering. Product dispatch is installed for the verified nontransactional
Direct W=2 pointwise and centered Window shapes. Unsupported Graph,
asymmetric Window, and transactional W=4 shapes fail cold into the established
route before Authority admission. The
legacy per-coordinate Sliding capability bits remain false because that
transport is fallback-only and is not persistent-product completion evidence.

Vulkan has a source-private HostCoherent GPU gate that authenticates the full
descriptor/control row before indirect dispatch. Its focused Q5/Q9/Q257 and
stale-row probes pass, but the GPU-owned fixed-native capability stays closed until the common
Authority row survives concurrent W slots through Release/Final/publication,
general generation stride is supported, and noncoherent/discrete visibility
is implemented. Metal now has the corresponding shared/unified-memory GPU
guard gate and actual Q5/Q9/Q257 plus stale-row evidence, but keeps the same
product capability closure until common Authority W2..4 Release/Final and
publication are exercised end to end.
The production Direct pointwise and centered Window paths consume the
persistent-product controller. Graph and the legacy raw fixed-W path do not.
Before a Window can create the generic Sliding owner, production rechecks the
complete pre-admission law: centered Clamp (not Clip), unary input, Q at least
2, serialized reads, both callback backings nonresident, non-required and
non-poolless execution, and exact backing id/version ownership. An ineligible
Window creates no Sliding owner or lease and continues through the established
fallback. Pointwise and other eligible Sliding routes retain their existing
attempt order. After mutation-free preparation succeeds, execution failure is
terminal for that attempt and is not converted into fallback.
The persistent data
service owns typed ready/wait/ack rows and may perform backing service, but it
cannot encode, seed, submit, or receive a native callback per coordinate.
Graph consumes planner-sealed resource and physical-reuse coordinates.
Coordinate issue remains ordinal: adjacent independent stages overlap, but a
blocked coordinate is not skipped by an arbitrary ready-node queue.

Window seal treats the raw canonical tail and the halo-expanded/frame tail as
independent extents. For N=48, F=16, R=2, P=12, the canonical extent is zero
because the raw input fills four canonical pages, while the final expanded
frame still has a clipped/fill tail extent. Exact canonical and expanded
materialization identities are retained separately; the existing N=53
partial-tail contract remains unchanged. The current backend boundary is an
O(Q) preencoded/HostCoherent service transaction with fixed-W callbacks and
backend-managed physical chunks of at most two coordinates. It is not GPU-owned
fixed-native recurrence and does not implement a Graph halo.
Product evidence for this boundary is compiled by case: `virtual/product/window.cpp`
owns the N=53 canonical route, while `window/exact.cpp`, `clip.cpp`, and
`tier.cpp` own exact-tail, Clip, and persistent-tier cases. `window/model.cpp`
is the sole expected-value/route predicate owner and `local.hpp` is its narrow
declaration and constant seam.
Callback-backed Virtual outputs may separately opt into the
`VirtualBackingTransaction` publication boundary. Bounded Sliding stages
feed provider-owned shadow pages while observers see old bytes and version.
Only after the final Authority drain does a mutation-free coverage check
permit one commit/version increment. A known pre-freeze rejection sets
`CloseRequirement::Required` and synchronously hands off to `close_generation`;
a restored post-Final failure closes as `Closed`, while rollback or control
uncertainty is `Quarantined`. This does not change the O(Q) service model or
claim GPU-owned recurrence, and legacy backings retain direct writeback.

The retained product binds Authority and the Sliding controller as one paired
lifecycle. The Authority gate is acquired before the Sliding State gate for
admission, paired abandon, and sticky quarantine. A pre-submit Known failure
may clear both owners only after the exact plan/token/generation/owner and
owner nonce match, with no issued rows, native work, or callbacks; otherwise
the pair remains retained and is quarantined. Unknown or writable evidence
retains the credentials and physical rows, and cannot rearm. A warm attempt
uses one locked idle snapshot of the run, backend ticket/control, prepared
lease, and Sliding quiescence, then rechecks that state before staging. The
common controller is retained across a Known close; Metal one-shot native
objects remain a platform allocation blocker to a strict allocation-free
native claim.
The actual Metal Persistent spatial-Window owner now closes this boundary for
the centered Clamp stride-one SharedHalo Q=2, N=48 case: one native submit and
queue call, logical backing read bytes exactly `48*sizeof(scalar)`, expanded
H2D bytes exactly `4*16*sizeof(scalar)`, no duplicate canonical-overlap read,
and one exact Final/publication. A backend-private proof authenticates the
bound `Map -> Window -> Map` schedule and each complete local ICB range;
temporal `MetalWindow`/`ResidentState` recurrence remains a separate
ServiceFree Direct/history route and is rejected by Persistent Sliding. This
evidence is Metal-only; unsupported forms remain on rolling service.

The controller declaration model follows the same ownership split under
`execution/sliding/model/`: `invocation.hpp` owns immutable direct/graph
projection, `ticket.hpp` owns the authenticated transition coordinate,
`transfers.hpp` owns move-only Fetch/Promote/Native/Drain/Persist credentials,
`evidence.hpp` owns final evidence and freeze, and `controller.hpp` owns the
state-machine surface. `execution/sliding.hpp` is only the stable include
surface and owns no duplicate state or transition formula.

Implementation authority:

- `node/src/compute/device/residency/registry/sliding_owner.{hpp,cpp}`
- `node/src/compute/device/residency/execution/sliding.hpp`
- `node/src/compute/device/residency/execution/sliding/`
- `node/src/accel/kernel/residency/sliding.hpp`
- `node/src/accel/kernel/residency/sliding.cpp`
- `node/src/accel/kernel/residency/sliding/`
- `node/src/accel/kernel/residency/persistent_sliding.hpp`
- `node/src/accel/kernel/residency/persistent_sliding/`
- `node/src/accel/{metal,vulkan}/kernel/pipeline/residency/sliding.*`
- `node/src/compute/virtual/run/sliding.hpp`
- `node/src/compute/virtual/run/sliding/`
- `node/tests/contract/compute/pipeline/residency/sliding/local.hpp`
- `node/tests/contract/compute/pipeline/residency/sliding/support.cpp`
- `node/tests/contract/compute/pipeline/residency/sliding/lifecycle.cpp`
- `node/tests/contract/compute/pipeline/residency/sliding/graph.cpp`
- `node/tests/contract/compute/pipeline/residency/sliding/terminal.cpp`
- `node/tests/contract/compute/pipeline/residency/sliding/dispatcher.cpp`
- `node/tests/contract/compute/pipeline/residency/sliding/authority/local.hpp`
- `node/tests/contract/compute/pipeline/residency/sliding/authority/physical.cpp`
- `node/tests/contract/compute/pipeline/residency/sliding/authority/window.cpp`
- `node/tests/contract/compute/pipeline/residency/sliding/authority/cache.cpp`
- `node/tests/contract/compute/pipeline/residency/sliding/authority/dispatcher.cpp`
- `node/tests/contract/compute/pipeline/residency/sliding/generation.cpp`
- `node/tests/contract/compute/pipeline/residency/native_sliding/dispatcher.cpp`
- `node/tests/contract/compute/pipeline/residency/native_sliding/backend.cpp`
- `node/tests/contract/compute/pipeline/residency/native_sliding/support.cpp`
- `node/tests/contract/compute/pipeline/residency/native_sliding/admission.cpp`
- `node/tests/contract/compute/pipeline/residency/native_sliding/terminal.cpp`
- `node/tests/contract/compute/pipeline/residency/native_sliding/concurrency.cpp`
- `node/tests/contract/compute/pipeline/residency/native_sliding/reentrant.cpp`
- `node/tests/contract/compute/pipeline/residency/persistent_sliding/`

The native sliding fixture has one fake-backend execution owner in
`native_sliding/backend.cpp` (capability, seed, submit, and worker terminal
delivery). `native_sliding/support.cpp` separately owns projection/release/final
callbacks and fixture/request construction. Both operate on the one state model
declared in `local.hpp`; neither retains a shadow descriptor or completion
authority.

The Direct product coordinator is split by authority, lifecycle phase, and
external-operation boundary. No translation unit may acquire a second policy
role merely to share mutable state:

| File | Sole responsibility |
| --- | --- |
| `sliding/internal.hpp` | private include routing only |
| `sliding/model.hpp` | fixed coordinator state only |
| `sliding/operations.hpp` | private cross-phase function contracts only |
| `sliding/state.cpp` | state validation, failure ordering, and work-cell reset |
| `sliding/pipeline/` | attempt begin, finish, rebase, and tested admission injection |
| `sliding/fetch.cpp` | Fetch transaction ordering only |
| `sliding/fetch/` | one file each for backing read, authenticated overlap reuse, sealed frame fill, and telemetry application |
| `sliding/promote.cpp` | Promote transaction ordering only |
| `sliding/promote/` | Host-to-Device copy only |
| `sliding/native.cpp` | Authority Native ticket to raw selection projection |
| `sliding/drain.cpp` | Drain transaction ordering only |
| `sliding/drain/` | Device-to-Host copy only |
| `sliding/persist.cpp` | Persist transaction ordering only |
| `sliding/persist/` | one file each for recovery, backing write, and telemetry application |
| `sliding/output.cpp` | output-phase ordering only |
| `sliding/final/` | one file each for close, completion, quarantine, status, and statistics |
| `sliding/prepare/` | one file each for admission, owner allocation, role projection, and backend preparation |
| `sliding/prepare.cpp` | cold preparation ordering only |
| `sliding/execute/` | one file each for owner validation, Authority binding, rejected-start close, and run-state lifetime |
| `sliding/execute.cpp` | top-level lease/bind/submit/wait ordering only |

Phase helpers perform one state transition or one external operation. The
orchestrators compose those helpers but do not reproduce their policy or
terminal algebra.

The raw fixed-W transport follows the same boundary under
`node/src/accel/kernel/residency/sliding/`: `internal.hpp` owns its private
state contract, `admission.cpp` performs the allocation-free role/capability
preflight, `service.cpp` owns the cold service cell, `state.cpp` validation
and failure ordering, `projection.cpp` Project handoff, `submission.cpp`
native seed/submit, `completion.cpp` backend terminal delivery, `final.cpp`
Final delivery, `failure.cpp` projection failure disposition, `returned.cpp`
callback-return reuse, and `pump.cpp` bounded W orchestration. The adjacent
`residency/sliding.cpp` contains only the public Prepare/Submit/Wake control
surface. In particular, `pump()` no longer performs Project validation or
native submission itself; it composes those single-transition helpers.

Metal and Vulkan use the same hard boundary below their backend residency
directories. Each `sliding/` folder has independent translation units for the
GPU shader source, descriptor authentication, cold preparation, capability
reporting, warm submission, terminal/final classification, and diagnostics.
Vulkan additionally separates retained command recording from live queue
submission. The former combined backend `sliding.mm` and `sliding.cpp`
translation units are absent from the build graph.

The Authority/model layer is likewise rooted at
`node/src/compute/device/residency/execution/sliding/`. Its `internal.hpp`
contains only the fixed state-machine layout and declarations; `state.cpp`
owns shared credential, ring, frontier, quiescence, and evidence operations.
`invocation/` separates
identity/factory, capacity, projection, and predecessor queries;
`lifecycle.cpp`, `invalidation.cpp`, and `final.cpp` own their respective
controller transitions. `failure.cpp` decomposes failure
handling into first-failure recording and independent input, Native, output,
and terminal suffix dispositions; its top-level `fail()` only orders those
transitions. Each `logical_*.cpp` owns exactly one model phase. Under
`physical/`, each phase has independent `issue.cpp`, `terminal.cpp`, and
`release.cpp` translation units, so capability minting, completion recording,
and callback-return publication never share a file. The longer Fetch and
Promote issue paths further isolate credentials, projection validation,
physical selection, atomic state commit, and ticket minting; their public
issue functions only hold the lock order and compose those steps.
`bind.cpp`, `owner.cpp`, and `reject.cpp` contain only their named boundary
operations. The former 3,400-line combined translation unit is absent from
the build graph.

## One invocation, one public attempt

The controller owns one invocation credential and one public Pipeline attempt:

```text
claim -> start once -> sliding native/data recurrence -> finish once -> publish once
```

Subepochs use a controller-local `(run generation, global coordinate, role
turn)` guard. They do not mint public Pipeline generations. Current per-epoch
`begin/finish/publish/reseed` calls are not reused by this path; preserving
them would retain a Host control round trip even after all data were ready.

The Direct topology is projected by `execution::Plan`. Graph coordinates and
PageUse are projected by `TiledGraphInvocation`. The controller holds only
cursors, tagged cells, and authenticated tickets. It never reconstructs
next-use, pinning, page identity, or stage dependencies. The focused model
uses those sealed facts in a bounded logical-cell victim scan; production may
not treat that simulation as physical authority. For Direct Fetch, Authority
now reprojects the exact Plan/PageUse under
`Authority gate -> Sliding State gate`, selects the real bank-local Host
frame, and mints a move-only capability. A miss has separate terminal and
callback-return release operations, so a late wrapper cannot recycle a row
merely because it reported its terminal. Promote, Native, Drain, and Persist
now follow the same physical ownership law; product connection still requires
one immutable Authority/Pipeline/transport descriptor and an outer-submit
trampoline that converts inline-callback plus submit rejection to Unknown.
Raw Sliding Final additionally carries a fixed per-frame observation journal:
plan-less close retains a Host/Input row only when this generation completed
the exact handoff and its observed key still equals the resident key. Failed,
invalidating, or never-handled rows are not retained; the journal resets with
the generation and is bounded by the fixed frame capacity.
The Direct seal also computes the maximum finite-retention live set per bank;
if that set exceeds H, construction rejects Capacity instead of exposing an
impossible success run as indefinite backpressure.

Centered Window Plan additionally owns a distinct canonical backing identity
and an O(K) `WindowFootprintProjection`: at most `K+2` unique canonical pages
and `3K` exact source-to-expanded-target slices. This removes geometry
reconstruction from the future physical Assemble phase and seals exact
next-use/pin facts. The host-driven rolling expanded-frame Fetch/Promote
capability remains the fallback; Authority minting and product consumption
of the canonical projection remain open and are not counted as completion.

For `P` pages, epoch width `K`, `E=ceil(P/K)`, tail width `t` (`t=K`
when the last epoch is full), and bank `b in {0,1}`, the finite-retention
requirement is:

```text
turns_b = E <= b ? 0 : 1 + floor((E-1-b)/2)
live_b  = turns_b*K - ((E-1) mod 2 == b ? K-t : 0)
H_required = max(live_0, live_1)
```

`NeverUse` retention requires only the current `K` rows per bank. Every term
is checked before fixed-state construction.

## Fixed state

Let `H` be Host input slots, `O` Host output/persist slots, `L` backing lanes,
and `W` native slots. All are cold-admitted constants. The mutable owner is:

```text
Run              = Plan identity + Authority token + generation + owner nonce
HostInput[B][H]  = bank-local slot + coordinate + turn + Fetch/Promote terminal
HostOutput[B][O] = bank-local slot + coordinate + Drain/Persist terminal
Native[W]        = global coordinate + role turn + Promote/Dispatch/Drain terminal
FetchLane[L]     = production target: exact ticket + bounded callback lifetime
EventQueue       = production target: fixed nonrecursive pumping/pending cells
```

The fixed Plan may describe two physical Host-input regions, but Final owns
only the banks in the active generation: `min(Q, BankCapacity)`. A Q=1 close
validates and commits bank0 only; the reserved bank1 region is an unrelated
cache row and is neither inspected nor retired. Q>=2 owns both banks. This
active-bank bound does not weaken frame reservation or credential checks: every
frame actually reserved by the generation is still validated before commit.

For Direct, `B=2`; Graph currently binds one typed Host arena. The compiled
model owns `O(B*(H + O) + W + graph projection scratch)` state, independent
of logical node count. A slot is local to the coordinate's exact physical
bank, so a clean row in bank 0 cannot masquerade as a bank-1 hit. Every ticket
authenticates Plan identity,
Authority token, run generation, an internally minted owner nonce, global
coordinate, phase, exact logical byte extent, and slot turn. The production
owner must add the admitted `L` lanes and fixed event cells without changing
that identity law.

The product factory creates an inert `create_bound` owner: projection is
read-only, but every Fetch/Promote/Native/Drain/Persist issue, terminal, and
invalidation rejects until Authority binds the exact State. Once bound, the
model-local phase APIs remain rejected; only Authority-minted physical
capabilities can advance the product owner. The separate
`create` factory is model-only and cannot consume an Authority Final receipt.
Authority binding is legal only while the owner is pristine: no issued cell,
failure, frontier movement, or service counter may precede the exact bind.
Finalization is a two-party freeze. Sliding freezes terminal ingress and
emits one owner/nonce-bound Final; Authority validates and retires the whole
reserved physical generation, then returns one consumed receipt that closes
that exact Sliding owner. A copied Final cannot close another Authority or
another controller. Unknown runs are not eligible for this Known/success
receipt and retain their quarantine owner.

The model exposes a copyable strong handle. Every asynchronous terminal must
capture a copy of that handle plus its exact ticket; terminal ingress from
independent lanes is serialized by the retained State mutex. A Known close is
legal only after issued tickets quiesce. An Unknown close returns final
evidence but remains non-quiescent and requires the product quarantine owner
to retain the handle. Dropping that final handle is not a cancellation or an
Unknown terminal.

Host input slots pass through `Fetching -> Ready -> Promoting`, then return to
clean `Ready` only after the Known native terminal (or to `Free` when failure
discards the row). Native admission is impossible before the exact Promote
terminal, so a forecast for a later coordinate cannot overwrite bytes still
being copied or aliased by a prior native node. A Known may-write failure
enters `Invalidating`; only an exact Authority invalidation acknowledgement can
make that physical owner reusable.

## Frontier protocol

Forecast completion may be out of order. A GPU-visible/native-sync
`input_frontier` advances only over the contiguous HostReady prefix. Backends
may coalesce a multi-epoch advance into one event/timeline signal.

Publication order is part of the backend contract, not an implication of the
counter value. Host service first finishes the exact slot bytes and flushes
the mapped range when memory is noncoherent, then release-publishes the
`(global coordinate, slot generation, descriptor)` row, and only then signals
the frontier. Native execution waits/acquires that signal and establishes the
required Host-write visibility before reading either descriptor or data. On a
Known failure it flushes and release-publishes the failure coordinate before
opening the accepted no-write suffix. A backend that cannot prove these
operations is not admitted to the noncoherent lowering.

```text
native node e waits input_frontier > e
native node e reads its Authority-authenticated Host slot descriptor
native node e executes Promote -> Dispatch -> Drain
native node e signals done frontier and output slot generation
```

Queue order and exact resource edges protect Device reuse. There is no
artificial `done(e) -> Host callback -> ready(e+2)` edge when the next input
and an output slot are already available. Slow persistence blocks only when
the fixed output ring is full.

The current model serializes Persist issue in global coordinate/use order.
This preserves the existing `VirtualBacking` law that callbacks over one
shared backing do not overlap; two physical bank-local output rings do not
silently widen that capability. Parallel Persist requires Authority/backing-
minted independent serialization domains plus an O(1) ordered publication
frontier and is not implemented.

On the first Known failure `f`, the owner waits for all earlier issued Fetch
terminals, publishes `failure_coordinate=f`, and opens the accepted native
suffix. Each suffix node checks the failure coordinate before mutation and
drains through the no-write path. Future unaccepted nodes remain an algebraic
unsent suffix and never receive fabricated receipts.

After Authority coordinate rejection and matching Known suffix suppression/ack,
an authenticated projected, unissued, no-write work cell may be cleared; any
live, terminaled, may-write, or mismatched cell remains retained.

The fixed model preserves Ready/issued work below `f` and cancels speculative
Ready metadata at or above `f`. It never erases a sibling Drain/Persist that
already has a ticket: every accepted terminal must return before Known close.
Contradictory `success + UnknownMayWrite`, wrong-phase tickets, wrong byte
extents, and cross-owner ticket replay fail closed.

A successful Promote makes the native cell ready but keeps its Host row pinned
until the exact native terminal; this remains safe when a coherent lowering
aliases Host storage instead of copying it. The Known native terminal returns
the row to clean HostReady. The earliest unconsumed consumer in each physical
bank may rebind the exact PageKey and increments `fetch_hits` without issuing
backing work. A later same-bank hit remains immutable cache metadata until it
becomes that bank frontier; it does not retag a row and steal the reservation.
When H is full, the bank frontier may replace an unpinned Ready row; a later
same-bank Forecast miss may do so
only when its immediate consumer coordinate is nearer than the resident row's
sealed next use. The incoming row's own next-use is installed only after
Fetch. No Forecast may replace a Ready row owned by the bank frontier, even
when its closed semantic pin ends there; the native terminal is the physical
lifetime release. The bounded scan chooses
the eligible farthest next-use, with physical slot order as the deterministic
tie. It also reserves enough slots for the current coordinate, so an
out-of-order future Forecast cannot strand the prefix.

## Backend lowerings

The persistent product seam is deliberately not an extension of the portable
transport callback protocol. Its request seals fixed `W`, `Q`, Plan/token/run
credentials, immutable strong prepared owners, local masks, exact tail-local
count, control/descriptor-generation strides, and one aggregate `Final`. It structurally has no per-coordinate
`Project`, `Release`, or `Returned` callback. Admission requires all three
capabilities: `whole_run_preencoded`, the valid `mode` submit-shape authority
(`OneSubmit` = 1 native submit; `BackendChunked` = `ceil(Q/2)` native submits),
and `host_epoch_callbacks_zero`. The stronger triple
`device_generated_recurrence && fixed_native_storage && fixed_common_storage`
remains false for the current Metal/Vulkan O(Q) intermediate lowerings. Its
aggregate evidence records accepted and GPU
completed coordinates, completed prefix, suppression/failure coordinates,
native and epoch-native submit counts, backend epoch callback count, backing
ready/wait/ack counts, Final count, queue calls, completion time, credentials,
Q, and W. Typed synchronous `signal_ready`, `wait_done`, `ack_done`, and
`fail_service` operations carry exact credential/turn/slot/generation rows but
cannot encode or submit native work. Final remains closed until the Qth ack,
after Drain/Persist rather than at the last GPU done. Backends may not advertise this seam
until one native submission lets the device advance the recurrence through Q
and reports those counters from the completed run. The existing
`PreparedResidencySliding` request is explicitly the Host-driven fallback and
does not satisfy this contract, even if it uses bounded storage or one public
caller handoff.

The portable transport owns at most `W` reusable native slots. Submission may
be Host-release-driven, but it runs on the cold accelerator service lane and
reports every actual queue call. It is not called a persistent GPU scheduler.
Native terminal delivery invokes `Release` while the exact slot remains
non-reusable. Only after that callback returns does the fixed service cell
invoke `returned`. A false acknowledgement retains the exact
`(coordinate, turn, slot)` in `Returning`; an explicit Wake or successful
downstream progress re-arms retained Returning cells without recursive
callback entry. Final requires no Returning cell and no external callback.

- Metal: fixed MTL4 allocators/commands or a proven ICB-compatible typed
  topology. Command reuse occurs only after the exact slot terminal. A generic
  external-backing Graph cannot be claimed as one persistent kernel.
- Vulkan: fixed submit/timeline cells, optionally a distinct transfer queue.
  `VK_EXT_device_generated_commands` is an admitted capability only after an
  actual device contract; it is not the portable baseline.
- Noncoherent memory: either one accepted `CopyIn -> Dispatch -> CopyOut`
  bundle per slot or distinct transfer/compute timelines. A copy submitted
  behind an unopened wait on the same queue is forbidden.

The existing whole Schedule uses Q Metal command owners or Q Vulkan submit
records. It can remain a small-Q candidate, but arbitrary huge Q must select
the fixed-state sliding lowering before Authority begins.

The common transport authenticates `(Plan, token, run generation, coordinate,
stride, slot, turn, descriptor generation)`, exact active-local masks,
asynchronous callback lifetime, and fixed retained/transient storage. It
advances `coordinate += W` only from the exact slot terminal. Projection and
submit failures wait every accepted sibling without reusing a slot. A
same-thread inline callback, or a cross-thread early callback paired with
submit rejection, is retained as an accepted contradiction and becomes
`UnknownMayWrite`. Reentrant ready
notification is coalesced without recursively chasing Q coordinates. Each
cold transport State owns one bounded service event and worker lifetime; Wake
does no allocation, a repeated Wake while service is active sets the same
event, and there is no process singleton or cross-device service queue.
Release and native-submit callback handoffs count as external work, so Final
cannot release claims while user or backend code still holds a callback
pointer. Early terminals are retained by the exact slot's `Submitting` phase;
the submission-thread marker distinguishes an inline backend return from an
accepted asynchronous terminal. After a Known failure at coordinate `f`, a
successful earlier Returned coordinate `e < f` must still finish its
Drain/Persist service; only the failed coordinate and suffix may skip output.

This transport does not itself establish GPU visibility. A backend must
publish a mapped descriptor payload, flush it when noncoherent, and establish
a queue/event acquire before native work consumes locals or data. Vulkan's
checked-in HostCoherent gate is real GPU evidence, not a CPU atomic mirror,
but its product capability remains false for the integration, stride, and
visibility limits above. Metal also keeps the capability false. The raw
transport and one Vulkan source-private lowering are implemented; neither is
yet a selected product recurrence.

The Vulkan persistent path has a source-private `GeneratedIndirectMap`
admission/ownership implementation: fixed non-history Map records, one
generated-check resource, authenticated run/full-tail lifetime, and Unknown
quarantine. Ordinary four-binding/64-byte and checked six-binding/128-byte
layouts remain distinct, with O(Q) HostCoherent intermediates and both
`device_generated_recurrence` and `fixed_native_storage` false.

No natural callback-backed public product evidence is valid. Gather is
whole-buffer, while the fixed unary page-frame owner cannot assume page-local
indices. The removed fixture was not a Persistent, mixed-route, or Known
failure E2E proof. A future route requires explicit PageLocalIndex proof,
bounded authenticated multi-page Forecast/Promote/page-table state, or a
whole-resident source; until then unsupported public shapes retain fallback.

The current raw fixed-W transport still performs one Host projection, native
submission, terminal callback, and Release per coordinate. Even after its
descriptor contract is proven it is a portable bounded-storage fallback, not
a GPU-owned persistent recurrence and not evidence that per-coordinate CPU
orchestration has been eliminated.

## Evidence

Source-private final evidence includes planned/accepted/retired coordinates,
first unsent/failure coordinate, Fetch/Promote/Drain/Persist counts and bytes,
ready-frontier signals, Host service turns, compute and transfer queue calls,
inflight peaks, ready-edge wait time, retained bytes, transient bytes, and
certainty. Public stats stay ABI-stable; one public handoff is not interpreted
as zero Host work.

Success requires every accepted Drain and Persist terminal and one Authority
prepare/accept receipt. Both successful and Known-failed generations
conservatively retire their fixed reserved rows; no stale pre-run key survives
over rewritten bytes. `UnknownMayWrite` self-retains the exact reachable
native/Host cells and quarantines the run generation. Callback quiescence
happens before controller, backing, or Pipeline owners can be destroyed.

The 2-phase controller/Authority contract does not by itself make ordinary
backing writes transactional. Product integration must hold the Pool execution
gate and backing serialization across Authority accept and the sole version
publication. Generation-capable backing supplies commit/abort; ordinary
backing retains the recovery law in [Publication](./publication.md).

The fused Persistent Final enters a private `Inflight` subphase after its
Authority/Sliding rows are committed. Those two gates are released only for
the private `noexcept` continuation; Pipeline, publication, and backing gates
remain held. The live credential makes begin, abort, close, and duplicate
finish attempts Busy/Invalid without row mutation. The continuation reacquires
Authority then Sliding, authenticates the exact plan/token/generation/owner
and Final nonce, and closes exactly once. A post-callback contradiction
quarantines the retained owner and preserves the already-published terminal;
it never retries or rolls back publication. Pipeline API reentry is outside
this private callback contract.

## Acceptance

Persistent product completion has a stricter success boundary than the raw
fixed-W fallback: after one public handoff, `BackendChunked` accepts at most
two coordinates per native submission and therefore performs `ceil(Q/2)`
submissions for Q>=2; the `OneSubmit` path retains its one-submit law. The
Host performs zero per-epoch submit/callback handoffs;
`gpu_completed_coordinates == Q`; and exactly one
aggregate `Final` precedes exactly one product publication. Fixed W, Q,
credentials, completion prefix, failure/suppression coordinates, and actual
submit/callback/Final/queue counters must be present in terminal evidence.
The backend-neutral Q=5/9/257 fake `OneSubmit` contract proves only the API shape and
aggregate accounting. Product acceptance additionally requires the same facts
from actual Metal and Vulkan execution plus the Final/Authority/backing
publication ordering. Until then, the Host-driven fallback remains available
but cannot close persistent-product acceptance.

The minimum adversarial suite is:

1. Delay node 3 and prove independent safe node 4 can enter before node 3
   Final; no rigid W4 barrier.
2. Q=100,000 with identical retained bytes, bounded callback depth, and no
   caller-thread chunk/epoch loop.
3. Out-of-order Fetch, slot ABA, stale run generation, partial-read failure,
   hung callback lifetime, and exact same-owner terminal classification.
4. H-full next-use eviction, no-victim backpressure, late-join without a
   duplicate read, and exact ready-edge stall telemetry.
5. Slow Persist with early Device bank reuse until O fills; no output-slot
   overwrite and Final waits all writes.
6. A multi-stage Graph where PageUse next-use/pin lets a safe independent
   successor advance while a different resource is delayed.
7. Metal and Vulkan actual Q=2,3,5 plus warm bounded-storage evidence, exact
   output, Known/Unknown failure, and one internal version publication.
8. CPU public sizes, objects, symbols, product semantics, and performance path
   unchanged.

The persistent fake contract proves the backend-neutral request shape and
capability fail-closure. Actual Metal and Vulkan product contracts additionally
prove `ceil(Q/2)` physical native submissions for Q=2,3,5, zero epoch native
submits and scheduler callbacks, Q GPU completions, exact output, and one
aggregate Final through the installed production operations. Known and Unknown
failure, observer-atomic publication, and same-state terminal classification
are covered by the same public-run fixture. The legacy `OneSubmit` lowering
may retain transient O(Q) native descriptors; the bounded chunked lowering
retains two physical payload slots; Vulkan may additionally retain four fixed
authentication metadata cells. Neither mode is an O(1) persistent kernel or
device-generated-command lowering.

The production default first selects `StagedLoop` for eligible ordinary
all-staged unary nonresident Pointwise Q>=2 when exact mapped Host-visible/
coherent input and output views exist. It owns one native recurrence submit,
Q GPU epochs, zero transfer submits/bytes, zero Host epoch submit/service/
callbacks, and one Final/Authority/output publication/version. A structural
`BackendUnsupported` before owner, rearm, stage, lease, or native acceptance
cleanly declines to the legacy route; mutation or any non-capability failure
is terminal. Persistent Pointwise R2 is the `BackendChunked` fallback with
two-coordinate chunks and `ceil(Q/2)` submissions. Unsupported Persistent
spatial Window declines pre-lease to bounded Window (`Q<=4`) or Stream (`Q>4`),
with one logical handoff, one Final, and one output version; current Metal
fallback queue calls may be Q. Resident, required, Q1, graph/scan/reduce,
multi-input, and parallel-read shapes retain their existing DeviceVsm or
rolling routes. Explicit/resident DeviceVsm Window remains separate.

The accelerator rolling fallback also authenticates one exact cross-bank halo
reuse when a compacted miss list still begins at the failed page. The
Authority credential carries both complete CacheKeys and FrameRegions, source
and target frame/range geometry, backing-frame bytes, owner token, generation,
and nonce through worker completion and callback-return release. A source
claim blocks eviction until its producer is terminal; a cancelled or Unknown
cohort waits for worker quiescence, invalidates the target lease, and
conservatively discards the source before either bank is reusable. The N=53,
F=16, R=2, P=12 fallback therefore reads 212 logical backing bytes while
promoting the unchanged 320-byte physical Host ring; this remains rolling
service behavior, not GPU-owned fixed-native recurrence.

The same actual-backend fixture routes centered U32 Clamp Window through the
dedicated bounded Window/Stream fallback. For `Clamp(Sum)`, `P=9` pages form
`Q=5` coordinates with a short final page; the current Metal fallback may make
Q physical queue calls while retaining one logical handoff, one Final, and one
output version. It proves exact boundary bits, backing halo reuse, and full
frame promotion bytes where admitted. Clip Window remains the explicit
DeviceVsm contract or the established rolling fallback; it is not generic
Persistent Sliding evidence. The overlap is copied from an
Authority-authenticated and callback-pinned prior Host frame. This does not
claim the general Graph `FootprintEpoch` canonical binding.

The current model contract proves the Direct node-3/node-4 nonbarrier case,
Q=100,000 with the same fixed owner, Promote-held Host-slot lifetime,
Plan/token/generation/owner ticket isolation, generation-tagged stale callback
rejection, concurrent Fetch terminals, exact logical byte receipts,
clean Graph HostReady reuse without a duplicate backing read, deterministic
next-use/pin replacement and current-prefix reservation, output-ring
pre-admission backpressure, accepted multi-output Drain lifetime,
Known may-write invalidation, future-failure prefix progress, malformed Unknown
quarantine, Graph strong-plan lifetime, exact multi-resource producer fan-in,
adjacent independent-stage overlap, zero transient Host fetch/persist, pristine
Authority owner binding, cross-Authority/controller replay rejection, a
one-use two-phase Final receipt, and conservative physical-generation retire
on success or Known failure.
Q=100,000 currently proves storage only; it is still a caller loop. The raw
Accel transport separately exercises Q=5, Q=9, and Q=257 through fixed slots
and asynchronous terminal-driven progress with no caller Q loop, including a
delayed-slot nonbarrier oracle. The same contract covers role strides 1/2/3,
multi-Wake service coalescing, immediate and sibling projection failure,
Project/Seed/submit-return races, Release handoff quiescence, exact control
generation, active-State self-retention, local-mask bounds, cross-slot and
nested inline callbacks, reverse-order failure pairing, Unknown quarantine,
false-return pressure, peer-progress re-arm without a new native edge, and
Final-callback Q=1 to Q=5 resubmission. This is transport model evidence, not
actual Metal/Vulkan execution evidence. Backend GPU-visible descriptor
ordering, the device recurrence, one whole-run Pipeline attempt, and the
product coordinator join are covered by the actual persistent product
contracts above. Out-of-order ready-node issue, canonical multi-page
`FootprintEpoch` binding, phase-typed Dispatch-versus-Drain dependency gating,
O(1) backend
native storage, and product throughput remain open.

GPU idle can approach zero only when the latency inequalities in
[Forecast](./forecast.md) hold. A 100x claim requires measured same-workload
evidence and is not implied by this state machine.
# Graph remap boundary

The fixed sliding/graph boundary accepts only the nonresident full-page
resource template described by [Residency Graph](../graph.md). `Begin` and
`End` mappings are resolved per active batch and are authenticated together
with backing id/version, plan identity, and frame capacity. Sliding's own
ordinal window and halo contracts are unchanged; resident DeviceVSM remap and
partial-byte Assemble remain partial rather than being inferred from this
binding view.
