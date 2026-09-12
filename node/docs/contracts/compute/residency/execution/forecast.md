# Forecast Residency

This page owns the continuous input-residency law shared by Direct and typed
Graph execution. It is an architecture contract until the status table below
names an implemented consumer. It does not make a throughput or GPU-idle
claim.

## The hard boundary

Backing fetch and Device promotion are different dependencies:

```text
ForecastFetch(e) -> HostReady(e)
Release(e-2)     -> DeviceBankFree(e)
HostReady(e) + DeviceBankFree(e) -> Promote(e) -> Dispatch(e)
Dispatch(e) -> Drain(e) -> HostOutputReady(e) -> Persist(e)
```

`ForecastFetch` may run before the same Device bank is reusable. `Promote`
may not. Combining both operations in `Input(e)` puts backing latency on the
critical path even when Graph next-use proves the page is needed far in
advance. The continuous execution model therefore removes backing fetch from
the Device-bank phase; it does not weaken the bank-release dependency.

Drain and backing persistence are separate for the same reason. Once an exact
D2H/coherent Drain terminal has moved output into an independently owned Host
staging slot, the Device bank may be reused. Slow persistence backpressures
only when the bounded output ring is full. Final waits every accepted Persist
terminal.

Graph reduction partials make this distinction concrete: their physical
`GraphDrain` capability authenticates the planner-sealed `Transient`
materialization while moving Device output into Host staging. Only the later
deterministic fold and final output publication own the external `Backing`
identity. Reclassifying a partial as Backing would conflate Drain with Persist
and is rejected by Authority.

For a planner-sealed single-writer `ExternalOutput` whose persistence is
actually `Backing`, Authority now owns two fixed move-only `GraphPersist`
slots. Each capability reprojects the exact 128-bit plan/invocation output,
freezes one Dirty Host-output row as Writeback, and carries distinct absolute
backing and frame-relative byte offsets. A callback terminal is not a release:
the slot and frame remain unavailable until callback-return release. Known
no-write failure restores the exact Dirty row for retry; UnknownMayWrite keeps
the slot and row in sticky quarantine. Known success release retires the exact
Host staging row after backing regains authority. This is the physical bounded
transaction seam. The public `GraphPointwise` Host fallback consumes it
for final Backing pages and publishes only after every batch releases. CPU
writes synchronously. Accelerator fallback owns two cold-configured metadata
workers: default backings remain serialized, while a backing that reports
`VirtualWriteLanes::write_lanes() >= 2` admits the two disjoint Host banks concurrently.
Device-bank reuse opens at GraphDrain release; same Host-bank reuse opens only
at GraphPersist release. The current Graph reduction route still produces
Transient partials and does not enter this backing-output ring.

The Known no-write and UnknownMayWrite lifecycle laws above are lower-level
Authority/Host-ring coverage. The natural public GraphPointwise ring fixture
proves the successful two-bank path; it does not provide a natural public
failure/retry evidence case.

The immutable execution Plan remains the only page formula. Forecast is a
bounded transaction of the same Authority lease, not a second cache, victim
selector, or scheduler. Direct tickets contain the exact
`(plan, token, generation, global epoch, semantic use, physical Host slot,
Host slot generation, key, byte range)` projected by that Plan. Semantic use
and physical slot are distinct: `H>K` may place the same local use for two
future epochs in different Host ring slots. Graph Forecast carries the full
128-bit ResidencyPlan identity, retains that cold owner, reprojects the exact
`TiledGraphInvocation` use, and mints its Host frame through Authority. The
product coordinator supplies only its already-frozen Graph materialization;
it cannot author a `PageUse`, physical frame, next-use, or replacement
decision.

The physical Graph Forecast lifecycle is directly owned by the stateless
`registry/graph_forecast_owner.{hpp,cpp}` `GraphForecastOwner` facet. It is
returned only through `Authority::graph_forecasts()` and borrows the one
Authority gate, frame/epoch table, and fixed quarantine holders; it owns no
second lock, cache, lease table, or recovery journal. The immutable
`GraphForecast` and `GraphReady` credentials remain execution records, while
Authority retains the quarantine predicates and destructor-time holder check.

The physical Graph Promote lifecycle is directly owned by the stateless
`registry/graph_promote_owner.{hpp,cpp}` `GraphPromoteOwner` facet, returned
through `Authority::graph_promotes()`. Single and aggregate Forecast/Ready
validation, destination binding, terminal evidence, and release all borrow
the same Authority gate, frame/lease table, and fixed credentials; no second
promotion, frame, or quarantine state is introduced.

The physical Graph Drain lifecycle is directly owned by the stateless
`registry/graph_drain_owner.{hpp,cpp}` `GraphDrainOwner` facet, returned
through `Authority::graph_drains()`. Device-output to Host-output issue,
paired terminal validation, and callback-return migration/rollback borrow the
same Authority gate and physical rows; no second migration or quarantine
state is introduced.

The physical Graph Persist and CPU retry lifecycle is directly owned by the
stateless `registry/graph_persist_owner.{hpp,cpp}` `GraphPersistOwner` facet,
returned through `Authority::graph_persists()`. Host-output issue, paired
terminal validation, callback-return release, Known rollback, Unknown
recovery, and same-domain retry admission borrow the same Authority gate and
fixed rows; no second persistence, retry, or quarantine state is introduced.

For centered Window input, the Plan seals the raw canonical tail extent
separately from the expanded/frame tail extent. With N=48, F=16, R=2, and
P=12, the canonical four-page materialization has extent zero while the final
halo-expanded frame retains its clipped/fill extent. Canonical overlap
therefore reuses full raw pages without changing the expanded frame identity;
the N=53 partial-tail case continues to carry its nonzero raw extent. The
current Persistent consumer remains an O(Q) preencoded, HostCoherent boundary
with Host-owned service callbacks, not a GPU-owned fixed-native recurrence or
a Graph halo implementation.

The private resident Graph proof does not service missing pages and is not a
Forecast consumer. Native resident scheduling is not implemented here; sparse
Forecast/Promote/Drain/Persist remains on the established Host routes.

## Bounded journal

Let `H` be the registered Host input-frame count, `L` the admitted backing
read lanes, `O` the Host output-staging count, and `B` the Device-bank count.
Retained mutable state is:

```text
H tagged Host-slot records + L in-flight fetch tickets
+ B promote cells + O tagged output/persist cells
```

It is `O(H + L + B + O)` and independent of total epoch count `Q`. Every cyclic
slot stores the global epoch and a monotonically changing slot generation.
A late completion must match both before it writes or publishes metadata;
otherwise a fetch for `e` could overwrite a slot already reassigned to
`e + H`.

The fixed Host slots use the exact next-use already frozen by
`TiledGraphPlan`/`PageUse`. Eligible empty slots are preferred. Otherwise the
Authority evicts an unpinned resident page whose next use is furthest in the
future; ties are resolved by physical frame index. A page whose closed live
interval covers the consumer horizon is not eligible. This is one
deterministic bounded scan over at most `H` frames, not a second heap or
future-stage rescan.

## Ready horizon

For a consumer node `n`, the cold topology projects a `ready_epoch` and a
profitable `prefetch_epoch`. The runtime maintains a bounded ready horizon:

```text
prefetch_epoch(n) <= issue_epoch(n) <= ready_epoch(n)
```

The horizon is limited by available Host slots, live pins, the backing
owner's declared parallel-read capacity, and recovery generation. Direct
execution projects sequential input pages. Graph execution consumes its
existing per-port `PageUse.prefetch_epoch`, `ready_epoch`, `next_use`, and pin
interval; it does not reconstruct consumer liveness in the runtime. After the
hard bounded-plan admission for a nonresident, non-required
`GraphPointwise` with Q>=2 and one serialized external read per stage, Host
deferral uses the dependency-driven Forecast window or the ordinary two-bank
output capability (`VirtualWriteLanes::write_lanes() >= 2`).
`graph_wavefront_parallel_eligible` derives independent middle-stage inputs
from transitive same-batch predecessors in the sealed first/recurrent batch
forms. It does not encode an exact input count, stage count, or stage pair.
Live admission still requires the actual predecessor terminals. Each admitted
stage has at most one external input, and simultaneous callbacks must name
distinct backing identities. Same-stage external fan-in retains its existing
route.

The two fixed physical Forecast lanes are a work window, independent of the
logical `(batch, stage, resource)` coordinate. Before selecting a runnable
cell, the coordinator fills free lanes in planner order. Its fixed-stage scan
skips an input with a live callback or a Ready pin awaiting promotion, allowing
later independent inputs to use the free worker. A repeated use of one input
therefore cannot block another backing or overwrite the first consumer's Ready
frames; after that consumer runs, exact Device residency can satisfy the repeat. After a callback
returns, its authenticated receipt is retired into a pinned GraphReady owner;
the freed worker can fetch another dependency-ready input while another lane
remains in flight. The selected stage alone consumes its complete Ready set
through exact Promote admission. A successful callback never impersonates
Device readiness. A later failure joins or quarantines every issued owner
before reuse, through the same abort controller. CPU receipt-book recovery is
required only for CPU tickets; an accelerator never creates that book. A
known input failure that closes every native/Forecast owner preserves its exact
reason and permits a clean retry, without an output version increment. A
missing CPU book or unresolved/unknown native ownership still poisons.
Future-batch speculation
waits until this current-batch window retires, retaining the two-worker bound.

A Pool-owned `PrefetchCompletion` wake sequence replaces repeated polling and
yielding. The coordinator snapshots it before scanning lane readiness and
parks only while the snapshot is unchanged. Workers publish Ready under their
own gate before advancing the sequence; completion before the scan, between
the scan and wait, and after parking cannot be lost. Wakeup is only a hint:
lane state and Authority receipts remain the sole result owners. The sequence
outlives worker shutdown and allocates no per-run storage. Final publication,
Host-stage native submit/wait, and the existing backing transaction remain
unchanged. This is bounded Host input scheduling, not a native GPU consumer of
the ready horizon.

A forecast miss is not automatically a late GPU stall. It becomes late only
when `Promote(e)` reaches the ready edge without an authenticated HostReady
receipt. Telemetry must distinguish forecast issue, forecast hit, backing
bytes, promotion bytes, and ready-edge wait time.

## Failure and publication

- A Known fetch failure belongs to its exact future epoch. Already-ready
  prefix work may drain, but the failed epoch and every dependent accepted
  native gate take the known no-write path.
- An `UnknownMayWrite` fetch callback, a stale slot generation, or a callback
  after run close quarantines the forecast owner and every Host slot reachable
  by that generation. It cannot be converted to a cache miss.
- Every Forecast terminal and abort authenticates one shared owner/plan/token,
  ordered page, Read binding, and pinned-frame credential under the Authority
  gate. A malformed terminal uses authenticated invalidation; an alias
  conflict retains the exact capability in the bounded Authority holder until
  recovery can close it. This guard and its diagnostics are lifecycle safety,
  not a new Forecast implementation or product feature credit.
- Authority embeds four bounded empty Forecast holders, one per physical
  Forecast epoch. Once any holder is quarantined, ordinary graph admission,
  publication, and commit are sticky-blocked; the abort coordinator may
  recover only after its worker, ticket, and alias cleanup has reached the
  exact quiescent boundary.
- DeviceVSM owner release crosses a private accelerator-bound callback. The
  CPU route may invoke that callback, but it does not link or name the
  registration implementation directly; release is accepted only on the
  authenticated `Done` result.
- HostReady is cache metadata only. It proves neither Device promotion nor a
  native Dispatch.
- Persist writes target the backing's recovery/private generation when that
  capability exists. The internal version advances once after the sole
  Authority close.

For an output advertising `VirtualBackingTransaction`, Persist stages each
authenticated dirty range into provider-owned shadow storage instead of the
ordinary backing write path. Its move-only token binds backing id, base
version, logical bytes, page bytes, page count, and run generation. Complete
coverage is validated after all leases and callbacks close; physical backing
page geometry/count and logical write tiling/count are authenticated
separately. Then one commit
publishes the shadow and advances the version. Known partial failure aborts
with old bytes/version visible; unknown may-write quarantines the transaction.
Backings without this opt-in remain on the direct, non-atomic fallback.

An ordinary `VirtualBacking` exposes only direct reads and writes; it has no
transactional byte-visibility primitive. For that owner, an external observer
may see partial bytes after a failed run even though the internal version does
not advance. Only an explicitly sealed transactional publication capability
may claim externally atomic byte visibility.

## Noncoherent transfer

On coherent unified memory, `Promote` may be an authenticated alias or CPU
copy into a Host-visible Device allocation. On noncoherent memory, forecast
still hides backing latency, but promotion requires an independent transfer
timeline:

```text
HostReady -> CopyIn -> DeviceReady -> Dispatch -> CopyOut -> HostDrain
HostDrain -> Persist
```

Putting `CopyIn` behind a compute batch already waiting for `DeviceReady`
creates a queue cycle. A truthful lowering therefore integrates the copy into
the accepted native DAG or uses a distinct transfer queue. Host polling and a
completion-callback compute submit are not equivalent.

Mapped Host writes obey `fill -> flush -> release descriptor -> signal`;
native reads obey `wait/acquire -> Host-write visibility -> descriptor/data
read`. CopyOut similarly reaches a Host Drain terminal only after the required
Device-write visibility and mapped-range invalidation. Timeline/event values
alone are not treated as cache maintenance.

## Throughput bound

For steady-state epoch compute time `C`, promotion time `P`, backing fetch
time `F`, `L` independent fetch lanes, and forecast distance `D`, hiding
backing I/O requires at least:

```text
D * C >= F / L + jitter
```

and hiding promotion requires sufficient bank/transfer overlap for `P <= C`
on the consumed recurrence. When either inequality is false, zero GPU idle is
physically impossible for that workload. Admission may enlarge the epoch
footprint within the frozen memory budget, increase a bounded lane/bank count,
or select a fused kernel, but telemetry must report the remaining ready-edge
stall. No fixed speedup follows from this model.

## Status

CPU Host Graph execution does not enter Forecast or Promote. Its multi-input
Host reads are owned by the Graph receipt book, and GraphPersist publication
uses the fixed authenticated PersistPlan. Known rollback is retryable; an
Unknown credential is retained in the terminal self-rooted quarantine rather
than being released as if it were a clean Forecast receipt.

| Boundary | Status |
| --- | --- |
| Plan-projected next-use and Graph prefetch/ready epochs | Implemented |
| Two cold Host prefetch workers used by rolling execution | Implemented |
| Fixed model separation of ForecastFetch and Promote lifetimes | Implemented |
| Independent sealed Host input/output ring capacities H/O | Implemented |
| Fixed-model clean HostReady reuse and bounded next-use/pin victim scan | Implemented |
| Direct Plan-sealed exact backing-read descriptor | Implemented |
| Direct Authority-minted full-frame physical Fetch ticket | Implemented |
| Pointwise short-tail zero-fill Authority physical Fetch ticket/service | Implemented |
| Centered stride-one Window halo Clamp/Clip Authority ticket and fill service | Implemented |
| Direct adjacent-halo Authority source-frame pin and exact overlap reuse | Implemented |
| Accelerator rolling cross-bank halo overlap | Implemented for the natural N=53/F=16/R=2/P=12 fallback: an Authority-minted source/target key, bank, frame, range, owner, generation, and nonce credential removes duplicate backing reads while preserving 320-byte Host-to-Device promotion; request compaction must leave the exact failed page at slot zero |
| Callback-return-gated physical Fetch slot reuse | Implemented |
| Direct Authority-minted full-frame physical Promote ticket | Implemented |
| Direct Authority-minted physical Native/Drain/Persist tickets | Implemented |
| Direct persistent Authority-to-backend coordinator and aggregate Final | Implemented |
| Direct bounded Forecast read batching | Implemented: one allocation-free `read_batch` callback carries every dependency-independent Authority-minted miss in a pointwise coordinate; Window boundary tickets remain ordered so the prior Resident frame can supply exact overlap reuse; partial batch failure conservatively invalidates every possibly written Host frame before retry |
| Legacy raw per-coordinate coordinator with submit-return proof | Not implemented |
| Graph 128-bit typed physical Forecast lease and public Host-supply consumer | Implemented |
| Graph fixed two-lane exact external-input Forecast lifecycle owner | Implemented: `GraphForecastOwner` directly owns issue, terminal, abort, quarantine/recovery, release, and `GraphReady` retirement over Authority's single gate/table |
| Graph Authority-minted physical Host-to-Device Promote lease and public consumer | Implemented |
| Graph callback-return Forecast retirement into a pinned `GraphReady` row | Implemented: the fixed Forecast slot is freed while the exact Resident Host row and planner retention remain Authority-owned |
| Graph Authority-minted aggregate Host-to-Device Promote lease | Implemented for up to seven sources: failure consumes none; success reauthenticates every ready row and activates the destination stage once |
| Graph Authority-minted physical Device-to-Host Drain lease and public consumer | Implemented |
| Graph Authority-minted fixed two-slot Host-output-to-backing Persist lease | Implemented: exact absolute backing/frame-relative ranges, callback-return reuse, Known retry, and Unknown quarantine |
| Public GraphPointwise bounded Backing-output Persist consumer | Implemented for the Host fallback success path: exact K=2 pages/tail and one internal publication; natural public failure/retry evidence is not claimed |
| Graph independently pipelined backing Persist output ring | Implemented for the Host fallback: two fixed workers, exact Drain/Persist lifetime split, backing-admitted two-write concurrency, same-bank callback-return reuse, and Final join; Known/Unknown lifecycle is lower-level coverage |
| Graph bounded common ready-wavefront after exact Forecast/H2D facts | Implemented: dependency-driven two-lane Forecast refill and completion-driven wait; exact Ready/Promote ownership remains per selected cell |
| Public Graph later-stage reuse of an exact already-resident external input | Implemented |
| Public Graph more-than-two-backing Forecast/H2D issue | Implemented for the seven-input multi-stage Host bound: actual Metal/Vulkan prove five pages, three batches, thirty-five exact reads, exact tail/output, and one publication; the lower-level Authority aggregate is also seven-source |
| Service-free U64 Map-to-Reduce Graph product (Sum/CountNonzero/Min/Max), one submit and aggregate Final | Implemented |
| Service-free U64 Map-to-{Sum, CountNonzero, Min, Max} Graph product, whole-run staging, one submit and aggregate Final | Implemented: one/two-input actual Metal/Vulkan Q=5/9/257 and maximum-width six-input Q=5 cover every reduction; the latter proves thirty reads and zero Host epoch control |
| Exact canonical total element-local one-through-seven-read/one-value-write U64 Map-DAG-to-Scan DeviceVsm product, one submit and aggregate Final | Implemented at the fixed resident maximum: common admission rejects eight inputs, while actual Metal/Vulkan Q=5 maximum-width Inclusive/Exclusive cases bind seven ordered inputs, read 35 pages, use one native submit, zero Host epoch control, and one publication; one/two-input Q=5/9/257 depth coverage remains independent. |
| Continuous Host-ring forecast consumed by native Direct Schedule | Not implemented |
| Graph native/GPU-owned consumer of the same ready horizon | Not implemented; every Host wavefront stage still owns its native submit/wait |
| Distinct noncoherent transfer-queue lowering | Not implemented |
| Measured GPU-idle convergence or 100x speedup | Not demonstrated |
# Remapped external inputs

Forecast and Promote consume the same Authority token and sealed plan used by
Graph page remap. They do not rebuild a target-to-source table from a public
span. A remapped external resource resolves every active page to its source
backing key before the Host/Device readiness fact is issued; all consuming
stages therefore share the source page's next-use, pin, version, and cache
identity. A malformed or foreign remap is rejected before a Forecast row,
frame assignment, or backing read is changed. Partial-byte and multi-source
Assemble remain outside the Forecast contract.
