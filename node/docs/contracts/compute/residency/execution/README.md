# Residency Execution

This directory owns the backend-neutral compound execution transaction. Each
page has one authority; backend command construction and performance evidence
remain outside this directory.

## Read Order

1. [Model](./model.md) for the immutable recurrent formula and identity.
2. [Authority](./authority.md) for reservation, service tickets, and close.
3. [Window](./window.md) for the fixed native release unit.
4. [Stream](./stream.md) for arbitrary-Q bounded-window chaining.
5. [Scheduler](./scheduler.md) for the native whole-run implementation and API
   lower bounds.
6. [Forecast](./forecast.md) for the separate backing-fetch, Host ready-horizon,
   and Device-promotion law.
7. [Sliding](./sliding.md) for the fixed-state continuous controller and
   one-attempt frontier protocol.
8. [Persistent](./persistent.md) for the ordinary all-staged nonresident
   Pointwise `StagedLoop` probe and one-submit contract, the legacy
   `BackendChunked`/fixed-W service, dedicated Window/Stream fallback, the
   explicit DeviceVsm acceptance contract, and the remaining service-aware/
   device-generated boundary.
9. [Service-Free Direct](./service-free.md) for the stronger resident fused
   recurrence handoff with zero Host service turns.
10. [Routes](./routes.md) for the typed Direct, Window, Reduction, Scan, Graph,
   and noncoherent-transfer topology boundaries.
11. [Publication](./publication.md) for backing visibility and recovery.
12. [Evidence](./evidence.md) for implemented boundaries and verification.

The older `cycle::Plan` is a bounded structural model. `execution::Plan` is
the sole recurrent product formula consumed by Authority.

## Source ownership

The public two-bank publication cursor is owned by
`compute/virtual/run/publication.{hpp,cpp}`. The backing transaction and its
bounded provider journal are owned by `compute/virtual/run/transaction.{hpp,cpp}`.
The Virtual execution surface is physically split by responsibility:
`compute/virtual/run/execution.cpp` owns asynchronous DeviceVsm start/resume,
`execution/seal.cpp` owns immutable recurrent-plan sealing, and
`execution/sliding.cpp` owns the prepared Sliding backend handoff. These
compiled owners share only the declarations and value types in
`execution.hpp`; none reselects another owner's route or stores parallel
authority.
Its bounded physical output-row lease protocol is directly owned by the
stateless `compute/device/residency/registry/transaction_owner.{hpp,cpp}` facet
over the same Authority gate/frame table; it creates no second journal, cache,
or publication owner. The transaction does not forward publication
operations. Its implementation is split below `registry/transaction_owner/`:
`validation.cpp` owns canonical key, region, and journal capture predicates;
`tag.cpp` owns output tagging; `prepare.cpp` owns the locked lease acquisition;
`cleanup.cpp` owns stale-row removal; and `final.cpp` owns commit, abort, and
move-only lease consumption. The root source only constructs the stateless
facet, and the local seam contains declarations rather than a second frame
table or token. Direct non-Scan
execution is split by phase under `compute/virtual/run/overlap/`: `model` owns
value-only epoch state, `submit` owns native acceptance and completion, `flush`
owns output/reduction/backing staging, and `scheduler` owns only the
deterministic two-bank phase order. `prepare.cpp` is only the ordered cold
coordinator. Under `prepare/`, `admission` is the sole projection, receipt, and
Authority-lease owner; `supply` owns backing supply, residency accounting,
upload, and activation; `lookahead` owns future prefetch issue; `cleanup` owns
post-lease abort; and `prefetch` owns receipt cancellation, publication, and
alias release. Their `local.hpp` carries one transient context and declarations
only, so no phase duplicates the Authority state machine or route policy.
The adjacent Virtual route dispatcher follows the same ownership rule:
`dispatch.cpp` selects only poolless/pooled and graph/accelerator/epoch paths;
`dispatch/poolless.cpp` and `pooled.cpp` own their distinct admission sequences;
`device_vsm.cpp` owns DeviceVSM disposal and execution; `graph.cpp`,
`accelerator.cpp`, and `epochs.cpp` own their respective execution families;
and `work.cpp` plus `result.cpp` own one-time work preparation and result
projection. `dispatch/local.hpp` is declaration-only and carries no shadow
route, transaction, or publication state.
Interval duration, intersection, and stall/transfer accounting are shared with
the recurrent Graph path through `compute/virtual/graph/reduce/timeline`.
The Graph HostReady issue/terminal/abort/quarantine/recovery/release/retire
protocol is directly owned by the stateless
`compute/device/residency/registry/graph_forecast_owner.{hpp,cpp}` facet over
the same Authority gate, frame/epoch table, and fixed holders; Forecast and
Ready records remain the only capability storage. The matching HostReady to
DeviceReady single/aggregate issue, group validation/bind, terminal, and
release protocol is directly owned by the stateless
`compute/device/residency/registry/graph_promote_owner.{hpp,cpp}` facet over
that same gate/table; it creates no second lease or quarantine authority.
Graph epoch admission keeps its bounded transaction phases in
`compute/device/residency/registry/graph_epoch/{validation,assignment,relocation}.cpp`;
the Authority entry remains the only lock, retry, state-gate, and final
token/generation publication boundary.
The per-epoch Virtual runner is likewise partitioned by semantic phase under
`compute/virtual/run/epoch/`: `admission` owns projection, prefetch receipt,
and physical transform admission; `supply` owns backing/read-cache supply and
lookahead prefetch; `execute` owns the selected Pipeline dispatch; `output`
owns output reservation, download, scan/retention, and execution close; and
`drain` owns Device/Host drain and backing writeback. `cleanup` owns only the
shared bounded rollback path, while `epoch.cpp` orders these phases. The
`VirtualEpochContext` is transient borrowed scratch; Pool/Authority leases and
Pipeline/backing state remain the sole storage authorities.

Route priority is part of this execution contract. An eligible ordinary
all-staged unary nonresident Pointwise run with `Q>=2` first selects
`StagedLoop` when exact mapped Host-visible/coherent input and output views are
available. It owns one native recurrence submit, `Q` GPU epochs, zero transfer
submits/bytes, zero Host epoch submit/service/callbacks, and one
Final/Authority/output publication/version. A pre-lease structural
`BackendUnsupported` with no owner, rearm, stage, lease, or native acceptance
cleanly declines to the legacy route; a mutated or non-capability failure is
terminal. Persistent Pointwise R2 remains `BackendChunked` with two-coordinate
chunks and `ceil(Q/2)` submissions. Persistent spatial Window is not the
generic Sliding `OneSubmit` product: unsupported forms decline pre-lease to
dedicated bounded Window/Stream fallback, while explicit/resident DeviceVsm
Window remains separate. A finite centered I32/U32 Window can instead carry
the typed `WindowRingFused` proof through pre-stage and Authority admission.
The proof accepts bare forms and exact authenticated, parameter-free
`CanonicalTotalU32` Map chains of symmetric depth 1–3 before and after the
Window. It uses one physical dispatch/native submit for `Q` seed epochs, `Q`
compute epochs, and `2Q` internal phases, with zero Host epoch
submit/service/callbacks, zero transfer submits/bytes, and one
Final/publication/version; I32 Map fusion is outside this proof. The current
DeviceVsm Window product case matrix keeps its arithmetic, statistics, failure,
and shape oracles in `pipeline/device_vsm_product/window/model.cpp`, its ring
execution lifecycle in `run.cpp`, and its matrix/order routing in `cases.cpp`;
the existing `fixture.cpp` and `fusion.cpp` remain the canonical shared fixture
and fusion owners.
`tools/test/run --fresh compute.pipeline-metal-persistent-sliding` target is a
passing functional E2E check, not a performance result or a 60-sample timing
claim. Its public StagedLoop/DeviceVsm fixture verifies one native
submit/handoff, Q GPU epochs, zero Host epoch callbacks/transfers, and one
Final/publication/version. Shared Persistent Q2/Q3/Q5 `BackendChunked` and
direct API Q5/Q9/Q257 checks remain separate from that public StagedLoop
evidence. Reserved-sentinel/overflow requests are rejected by range preflight
before native owner allocation; this does not classify every past IOGPU abort
or establish safety for every finite-Q device/driver. Valid finite-Q direct
Metal O(Q) encoding still awaits authoritative native command-capacity/ICB
migration.

Persistent Sliding range admission is finite and generation-safe: the final
turn for each active role is checked against its uint32 control and uint64
descriptor generation, while `UINT64_MAX` remains the reserved sentinel and
is rejected before native preparation.

The backend-neutral execution contract keeps its public `CheckExecution` entry
in `pipeline/residency/execution/dispatcher.cpp`. `execution/support.cpp`
owns only canonical request, footprint, plan, dependency, frame-owner, and
cache-seed builders; `execution/plan/{seal,project,input,window}.cpp` own the
immutable Plan sealing and projections, while `execution_owner.{hpp,cpp}` and
the registry execution leaves own the Authority journal and native
terminal/reuse contracts. The
dispatcher preserves the original plan → Authority → native phase order and
return IDs.

The Q=1 backend-neutral `execution::Run` join is split by lifecycle ownership:
`execution/run/lifecycle.cpp` authenticates begin, Input/Output issue and
terminals, native acceptance/rejection, and abandon; `execution/run/finalize.cpp`
merges host/native failure evidence and builds the one final close request;
`execution/run/authority_close.cpp` is the sole ExecutionOwner-gated close
coordinator. Its `authority_close/validation.cpp` leaf authenticates progress,
native evidence, and ordered failures; `authority_close/preparation.cpp`
preflights every frame/node mutation without applying it; and
`authority_close/application.cpp` alone restores, invalidates, or publishes
the checked rows and clears the execution slot. Their declarations-only
`internal.hpp` carries transient validation/preparation results, never a mirror
of plan, token, generation, frame, or terminal state.

Registry close paths are split by mutation boundary without adding a second
Authority: `registry/execution/close.cpp` owns only the row-clear invariant,
`close/reject.cpp` owns pre-native rejection, `close/abandon.cpp` owns the
ordinary/window/stream rollback protocols, and `close/abort.cpp` owns partial
stream abort. Generic epoch/writeback completion is separately owned by
`registry/complete.cpp`; every leaf still mutates the same gate-protected frame
and execution/cycle state.
