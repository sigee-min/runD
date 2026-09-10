# Residency Evidence

This page owns the checked contracts and evidence boundaries. It owns no
product behavior or implementation routing.

- `compute.pipeline-residency` proves compact stream geometry, capacity,
  identity, Authority state, and structural cycle rejection.
- `compute.virtual-residency-product` uses the public facade for CPU, Metal,
  and Vulkan
  preparation, cold/warm execution, active prefix, output/hash parity,
  backing failure/recovery, capacity stability, and allocation-free samples.
- Its failure/recovery cohort is owned by
  `tests/contract/compute/virtual/product/failure/dispatcher.cpp`: `support`
  owns the single fixture and offset-failing Persistent backing, while
  `boundary`, `backing`, `device_vsm`, `native`, and `multilane` own returned
  status, backing retry, partial-output poison/recovery, native transfer faults,
  and multi-lane cancellation/token reuse in that order.
- Authority and Pool leaves prove Device-global hits, bank pinning, dirty drain,
  physical owner reuse, view activation, single accounting, and exact failure.
- The Reduce cross-layout lending cohort creates and retains its own
  service-aware Graph warm-up before the Direct borrower. It validates the
  warm-up result as well as exact borrower output, nonzero cache hits, and
  reduced backing reads; preceding DeviceVsm/sequence cohorts are not an
  implicit cache-lifetime precondition.
- `pipeline/residency/pool_admission.cpp` exercises Graph allocation after an
  ordinary extent-zero arena, then same-page cross-Type and reblocked Graph
  loans. It checks nonzero Graph identities, shared canonical allocations for
  admitted loans, and unchanged live ordinary buffers, regions, and identities.
- `pipeline/residency/physical_view.cpp` exercises U64→U32→U64 views on each
  selected CPU, Metal, and Vulkan backend: exact typed capability shape,
  shared native allocation identity, unchanged allocation accounting, and
  byte-exact writes/reads through opposite views. Partial, enlarged, and
  overflowing view extents must fail without a new charge. The GraphResident
  product additionally executes kernels through the pooled U32 view; a
  shared native allocation must not be confused with a shared typed token.
- Graph product leaves prove non-prefix locals, multi-stage leases,
  deterministic fold, Transient discard, failure cleanup, and retry.
- Scan leaves prove inclusive/exclusive non-power-of-two tails and exact failed
  page recovery.
  The sequential epoch coordinator records one successful epoch after its
  initial Pipeline execution, before scan adjustment, output transfer, or
  publication. Internal scan executions do not increment it again; CPU is
  not exempt from this physical execution count. Native controller handoff,
  batch, and queue counters remain separate evidence.
- Concurrency leaves prove one-owner `PipelineBusy` and backing-set callback
  serialization without sleep-based acceptance.
- Backend fault leaves prove fail-closed terminal and D2H fallback behavior;
  injected failures are contract evidence, not native driver-failure evidence.
- `compute.pipeline` injects one lost Metal residency terminal and proves a
  bounded `DeviceLost` wake, one submitted command, one device-loss
  publication, no lost-run output callback, sticky same/peer/new-owner
  `DeviceLost`, unchanged peer callbacks, zero new-owner callbacks, stable
  committed memory, and self-retained may-write owners. Input callbacks before
  the lost native terminal are permitted. It is not a real queue-loss
  observation.
- `compute.pipeline-residency` proves the bounded structural Plan.
  `compute.virtual-residency-product` executes the production compute-flight
  journal through the stateless `CycleOwner` on Metal and its failure/retry
  routes exercise the shared cycle terminal cleanup. The implementation
  authenticates exact ranges, masks, parity, and prior-bank terminal order
  against Authority's sole storage. Multi-epoch Direct Metal additionally
  requires `command_inflight_peak >= 2` from accepted-submit/terminal
  producers; CPU requires zero. This is not a native three-phase DAG.
- CPU overlap is a separately emitted symbol that instantiates
  `UseAccelerator=false, UseCycle=false`. Object-code inspection must find zero
  coherent input/output view, Device migration, native download,
  `cycle_flight`, `bind_cycle`, `advance_cycle`, `complete_cycle`, or
  `close_cycle` references inside that symbol. The source dispatches once
  before the epoch loop, and the whole-execution gate precedes construction of
  bounded-window and Q=1 owners. This is structural hot-path evidence, not a
  throughput claim.
- The Execution leaf in `compute.pipeline-residency` separately proves the
  invocation-sealed recurrent Plan/model, Authority journal, and run-level
  Receipt: arbitrary Q uses O(1) representative validation, exact projected
  keys/ranges/dirty tails/mutation regions, recurrence predecessors, bounded
  final Evidence, one required Receipt submission transition,
  stale-generation rejection, known versus unknown may-write quarantine, and
  same-tier overlap rejection. It also proves the unsupported common
  accelerator preparation clears its Owner and that a CPU Device exposes no
  DeviceOps. Transactional backing is not implemented. Arbitrary-Q model
  evidence is combined with the Q5/Q9 product contracts below; the model test
  alone does not prove a GPU recurrent schedule.
- The one-word Window leaf in `compute.pipeline-residency` proves the bounded
  W4 journal independently of an adapter: exact two-bank Input/Output service edges,
  four fixed Pipeline Releases, contiguous ready prefix, one public handoff
  versus native batches and queue calls, actual native in-flight peak, known
  submitted abort followed by an already-completed entry and no-dispatch
  suffix terminals, and `UnknownMayWrite` retention until the final callback.
  Its warm invocation also proves global two-bank Host replacement: one-shot
  page 0/1 misses use coherent direct backing fill without evicting retained
  page 2/3, then page 2/3 use exact Host-to-Device supply. The masks are
  respectively backing/transfer/coherent `1/0/1` and `0/1/0`.
- The same leaf separately proves the native whole-schedule Authority model.
  Q=9 and Q=100,000 success use one aggregate terminal and a four-slot
  circular journal. A Q=101 Known Input failure drains every already accepted
  suffix as dispatched/completed/no-write, retains one exact failure row,
  avoids a false Unknown overflow, closes without stale cache reuse, and
  immediately re-admits the same Authority. Metal and Vulkan production
  contracts consume this same aggregate terminal; the large-Q leaf remains
  the algebraic/O(1)-journal proof rather than native performance evidence.
- The Q=1 Execution join contract proves emergency abandon invalidates every
  reserved Host and Device cache row, rejects duplicate abandon/close, and
  permits a fresh exact-plan admission. The public Virtual product injects the
  harder post-backing/pre-close contradiction: physical output writes occur,
  the run returns `PipelineInvalid`, a shorter retry observes
  `BufferPoisoned`, and only a complete retry clears recovery. This is failure
  atomicity evidence, not transactional external-storage publication.
- The public active-prefix product executes a one-epoch Direct pointwise run on
  Metal and Vulkan through the connected compound coordinator and requires one
  native submission, one Authority epoch, exact prefix/tail bytes, backing
  publication, and stable capacity. CPU uses its existing rolling owner and
  never calls the accelerator execution preparation seam.
- The Metal leaf in `compute.pipeline` proves the first Q=1 Compute Pipeline
  execution owner: Plan-free cold owner aliasing, exact submit-time Plan
  binding, canonical Pipeline seed/start/finish/publish, rejection of tampered
  GPU control without generation advance, synchronous submit failure without
  callback or partial publication, one actual backend-produced native
  submission and terminal, one post-Pipeline `NativeEvidence`, 60
  allocation-free warm submissions, and injected `UnknownMayWrite` with zero
  fabricated completion. The same leaf proves the source-private W4 raw
  adapter separately: one common handoff, four actual Metal queue calls, four
  native batches, two-bank reuse only after the prior Release, exact e0/e2
  terminal generations, stale signal rejection before control mutation, 60
  warm zero-C++-allocation windows, and an asymmetric suppressed e2 that is
  Known dispatched/completed with `may_write=false` while all four Releases
  and one Final drain. The emergency case rejects two stale readiness signals,
  invokes the exact-generation abort seam, opens every remaining no-write gate,
  and combines an injected lost last Metal done terminal into conservative
  `UnknownMayWrite` Releases and one Unknown Final after all four queue calls.
  Final removes caller-owned callback pointers while retaining common/backend
  quarantine. Metal listeners transfer callback bundles to the cold serial
  residency-service queue, and the same warm loop proves zero tracked C++
  allocation across that fixed delivery path.
  The same Metal leaf executes actual public Direct pointwise Q4, Q5, and Q9
  runs. Q4 proves one bounded window; Q5 and Q9 consume the whole-Schedule
  lowering and cross bank-generation and short-tail boundaries. Every run
  requires one public handoff, Q native batches and queue calls, exact backing
  callbacks, and transformed output. Schedule completion is gathered by one
  owner serial delivery lane; ten consecutive full contracts prove independent
  bank listeners do not reorder the global receipt prefix. A Known Q5 input
  failure has zero successful compute submits, drains all five accepted native
  batches through no-write gates, and retries on the same owner. A
  `retained_bytes-1` reservation miss rejects Schedule before Authority,
  executes the fixed-chunk fallback, and admits Schedule after refund. A lost
  Q5 terminal yields sticky `DeviceLost`.
  The mutation-free readiness query is true before claim and false while the
  fixed window owns either prepared bank. The backend has no second
  whole-execution DeviceOps authority. The successful public run traverses W4
  Host Input/Output service and Authority close; this leaf does not prove an
  external `Publication` transaction or throughput.
- The Vulkan leaves in `compute.pipeline` prove cold retained
  prefix/per-local/suffix construction, exact arbitrary-local order and tail,
  duplicate/out-of-range no-submit, and the common Q=1 native consumer. The
  bounded raw leaf executes Q=1,2,3,4 with one `vkQueueSubmit(Q)`, exact
  per-cell ready and ordered done generations, e1-before-e0 readiness, actual
  two-bank peak, asynchronous handoff before Release/Final, Q4 GPU-visible
  suppression/no-write, and one exact Final. Sixty warm Q4 windows keep the
  runD-owned Vulkan Pipeline, descriptor, and Buffer allocation counters fixed;
  the separately observed MoltenVK `operator new` count is opaque-driver
  diagnostic evidence and is not mislabeled as runD allocation.
- The same leaf executes the public Direct Virtual product at
  Q=2,3,4,5,9,17 on an actual coherent Device-local adapter. Q<=4 uses the
  bounded window and Q>4 consumes the whole-Schedule lowering; every run
  reports `handoffs=1,batches=Q,queue_calls=1`, zero H2D/D2H submissions,
  exact output, and complete backing service. Known Q5 Input failure drains
  through no-write gates and retries; a lost Q5 terminal reports the submitted
  logical batch, returns sticky `DeviceLost`, and performs no second queue
  submit. An ordinary public Vulkan Buffer is resolved
  from the native registry as DeviceLocal `Resident` with no mapping; only VSM
  Input/Output arenas carry the backend-neutral `HostVisiblePreferred` intent.
  Missing coherent Input or Output views remain pre-native rolling fallback,
  not a fabricated bounded-window success.
- The coherent Direct warm product retains the five-page working set across
  the global Host and bank-local Device input rows. Its cold cohort performs
  exactly five Persistent reads; all sixty warm Metal and Vulkan runs report
  zero backing reads, zero H2D/D2H submissions and exact output. The separate
  parallel-read rolling cohort proves one late Persistent read plus one
  Authority Host-cache supply for two logical Device page-ins; direct Host
  promotion reports no H2D submission while an exact private promotion reports
  one, and neither manufactures overlap. Pending Metal/Vulkan HostRead or
  HostWrite view-fault injections are observed by mutation-free readiness and
  executed by rolling fallback, proving the window cannot consume or hide a
  physical fault. A backing with more than one active read is likewise rolling
  evidence until the bounded controller owns two Host-service lanes.
- Persistent route regression also injects a terminal `BackendUnsupported`
  execution result after owner preparation and requires zero rolling reads,
  queue submissions, Final callbacks, and publication. This distinguishes the
  default terminal disposition from the sole clean pre-native lowering decline
  that is allowed to continue to rolling. Cold native preparation reasons are
  preserved through the common seam; a pipeline/stream-capacity result remains
  terminal and is not converted into rolling fallback.
- Vulkan abort accepts the already submitted Q4 window, opens unsignalled
  no-write gates in bank-safe order, drains every done point, emits four
  conservative `UnknownMayWrite` Releases, one Unknown Final, and sticky
  quarantine. The accel timeline leaf alternates sixteen Q1 generations with
  Q4 to prove independent ready-cell monotonics avoid finite
  `maxTimelineSemaphoreValueDifference` gaps. It also drops a transient idle
  adapter owner and requires immediate weak-owner expiry, proving completion
  workers do not pin an empty Device or self-join during teardown. These leaves
  prove fixed-chunk Direct control and its Q5/Q9 Stream chaining, not Graph,
  Scan/Reduce, persistent device scheduling, or a distinct transfer-queue DAG.
- The Vulkan timeline leaf additionally submits Q=17 actual reusable command
  batches with one `vkQueueSubmit`, four cyclic ready cells, independent-cell
  out-of-order readiness, exact same-cell contiguity, and one ordered terminal.
  The public Q=17 product consumes that same native lowering and aggregate
  Authority terminal; it is not a chunk-count surrogate.
- Execution admission must validate the first, each bank-parity interior, and
  tail representatives with a constant number of Plan projections even for a
  very large Q. O(1) allocation with `Q*3` projection work is not accepted as
  bounded Host orchestration. The success and failure oracles seed a prior
  `CacheKey`, model an opaque may-write mutation, close the transaction, and
  require the old key to miss. Restoring the prior metadata over changed bytes
  is a false hit. Host-service tests additionally authenticate the exact node
  and mutation receipt; aggregate progress alone is insufficient.

`begin_samples()`/`end_samples()` delimit the entire warm cohort. `profile()`
is busy inside it; the terminal Profile is the only accepted evidence owner.
The independent oracle validates backing order, poison tail, identity, and
allocation freedom without calling production planner/execution helpers.

## Local Vulkan GraphResident semantic evidence — 2026-08-23

The local Vulkan semantic route passed with base HEAD exactly
`ac79a09561fd7a04538b1b79b88110751bab4a19` and `dirty=true`:

```text
tools/test/run --fresh compute.virtual-graph-residency-product --backend vulkan
exit=0
focused binary: .cache/focus/compute/node/node-compute-accel
SHA-256: a3729ade008e7700e5943b2dfac27d409b46c700b9aa8beb08b4032e34125e1a
product output FNV/hash: 8788441728858461497
```

The scope is Vulkan-only; Metal was skipped. It is the fully resident,
parameter-free U64 bounded path for Q=5, C=2: a five-stage/eight-resource
branch/fan-in graph with three internal banked owners and four external
endpoints. X/W internal alias reuse waits for `DispatchComplete`. The run
set is one cold execution plus three warm executions. Each independent
execution requires one submit, one dispatch, one aggregate Final, one backing
publication, zero Host epoch submits/service/callbacks, zero backing transfer
bytes, and exact output/hash. The Final/publication requirement is per
execution, not one shared result across the four runs.

This is local semantic evidence only. The fresh route did not publish an
immutable source-manifest/source-identity/output-log packet, so it is not
sealed release or measurement evidence. No stale `.cache/source-manifest.tsv`
is cited. It does not cover runtime nonresident I/O, the general graph
surface, timing, or `100x` performance evidence.

Measurement tools and performance claims are owned by
[Virtual Performance](../../../../../../docs/reference/performance/virtual/README.md).

## Metal GraphResident local semantic evidence — 2026-08-23

The local Metal semantic route passed from base HEAD exactly
`ac79a09561fd7a04538b1b79b88110751bab4a19` with `dirty=true`:

```text
tools/test/run --fresh compute.virtual-graph-residency-product --backend metal
exit=0
elapsed_us=1292157 (harness duration, not performance evidence)
focused binary: .cache/focus/compute/node/node-compute-accel
SHA-256: 13dbe394faded048ca74b6f11bb24560ff1f4d43eb43e7a9a220b7723a123976
size=28549584
mtime=2026-08-23T11:16:31+0900
```

The PASS oracle proves the real Metal DeviceVsm GraphResident Q=5, C=2
bounded path: a five-stage/eight-resource branch/fan-in graph, three internal
banked owners and four external endpoints, one cold execution plus three warm
executions, one submit/dispatch/native Final/publication per execution, zero
epoch submits, Host service turns/callbacks, and backing transfer bytes. It
also asserts the exact output/hash, version progression, X/W
`DispatchComplete` alias, fresh proof/generation/nonce continuity, and no
quarantine. The hash `8788441728858461497` is oracle-asserted; the quiet PASS
does not print it.

This is local semantic evidence, not a performance or sealed-release claim.
It does not prove speedup, `100x`, the general/nonresident graph surface,
runtime Forecast/Promote/Drain/Persist, strict native allocation-free Metal,
or a sealed source-manifest/source-identity/output-log packet.

## Root integration semantic evidence — 2026-08-23

After the Metal narrow semantic PASS above, the root integration contract
passed:

```text
tools/test/run --fresh compute.pipeline-residency
exit=0
build steps=116
elapsed_us=2588403 (harness duration, not performance evidence)
```

This is semantic integration evidence only. It does not change the bounded
GPU-owned/nonresident blocker recorded by the execution and state contracts.
