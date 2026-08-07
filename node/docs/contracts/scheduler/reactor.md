# Readiness Reactor

This page owns the scheduler-owned fd readiness reactor, timed reactor waits,
backend normalization, readiness ordering, fd generation ownership, persistent
ready sets, and network readiness integration. Network byte, connection, and
protocol semantics remain owned by [Network](../net.md).

## Source Map

- `runtime/reactor/readiness/state.hpp`: OS-neutral, width-safe handle,
  interest, and event state. POSIX readiness masks and platform SDK records
  cannot cross this boundary.
- `runtime/reactor/readiness/mask.hpp`: the only logical readiness bit
  composition, encoding, decoding, and matching owner. Implementations include
  it directly; the broad state owner does not forward it.
- `runtime/reactor/readiness/handle.hpp`: the only checked conversion owner
  between public integer descriptors and the width-safe reactor handle.
  Implementations include it directly.
- `runtime/reactor/platform.hpp`: the only scheduler-to-platform backend
  contract; it owns normalized backend state, registration changes, probe
  values, capacity preparation, lifecycle, registration, polling, handle
  identity, and native-error results.
- `runtime/reactor/diagnostics.hpp`: backend-lifecycle and scheduler-policy
  observation counters. Diagnostics never define backend state, readiness, or
  wake order, so changing an observation field cannot invalidate every user of
  the platform contract.
- `reactor/model.hpp`: scheduler-owned wait and expanded-ready values. Each
  expanded-ready value has one exhaustive disposition rather than a boolean
  tuple.
- `reactor/poll.{hpp,cpp}`: the immediate-probe result type and single
  orchestration owner. Its exhaustive disposition carries no mirrored request
  identity, and probing reuses the Runtime-owned backend and pre-reserved
  scratch rather than opening an ephemeral reactor or allocating on a warm
  probe.
- `reactor/diagnostic/platform.cpp`: the single backend-lifecycle and
  scheduler-policy diagnostic counter owner; platform backends contain no
  mirrored counter state.
- `runtime/platform/mac/reactor/`: the independently selected kqueue owner.
- `runtime/platform/linux/reactor/`: the independently selected epoll owner.
- `runtime/platform/portable/reactor/`: the independently selected POSIX poll
  fallback owner.
- `runtime/platform/unavailable/reactor.cpp`: the complete non-Unix interface
  owner whose preparation succeeds and whose actual backend operations report
  `ReactorBackendUnavailable`.
- `runtime/platform/posix/probe.cpp`: shared POSIX immediate-probe mechanics
  over a backend-owned reusable buffer; the scheduler sees only normalized
  requests and results.
- `reactor/registry.cpp`: the sole scheduler-side wait and descriptor-state
  authority. A fixed-capacity slot arena stores each full wait once, a compact
  slot-id order preserves wait-id order, and one fd state owns linked wait
  membership, aggregate read/write counts, generation identity, and native
  registration lifetime. There is no separate wait index or registration
  vector.

`ReactorRuntime::registry` is the only full `ReactorWait` authority. Wait ids
are nonzero, and the compact order is sorted by wait id, so cancellation lookup
does not require a second full record or wait-id index. Per-fd membership uses
32-bit slot links into the stable arena; expansion touches only the matching
fd chain. For `W` live waits, `F` distinct descriptors, `P` platform-ready
descriptors, and `R` matching waits, expansion costs
`O(P log F + R)` instead of `O(PW)`. Combined interest is maintained by exact
read/write reference counts, so one descriptor lookup costs `O(log F)` and
interest projection is `O(1)` rather than scanning its `K` waits.

The selected platform's private `ReactorPlatformState` is the only reusable
owner of raw platform-ready descriptors. `ReactorRuntime::ready` owns only the
expanded scheduler-ready values. Configure reserves no second platform-ready
vector. A duplicate at capacity `C` would add one allocation and
`C * sizeof(ReactorPlatformReady)` bytes without changing lookup, ordering, or
lifetime.

On the supported 64-bit ABI, a canonical wait is 88 bytes. The fixed arena uses
a 96-byte wait slot plus one 4-byte ordered slot id and one 4-byte free slot id
per configured wait. The fd state is 80 bytes. Warm add, remove, re-arm, and
batch drain stay inside configured storage; descriptor-capacity pressure
deterministically flushes already-deferred removes before rejecting a new
descriptor.

Batch removal validates the ordered success prefix, unlinks each selected slot
in `O(1)`, and compacts the 32-bit canonical order once. The first selected
wait for each fd emits its exact pre-removal aggregate interest in canonical
wait order; an fd-local touched bit suppresses later duplicates. Therefore
`drain/batch.cpp` owns no descriptor copy, sort, unique, full-fd scan, or
interest recomputation. For `R` ready waits, `W` live waits, `F` descriptors,
and `A <= R` affected descriptors, the bound is
`O(R(log W + log F) + W + A log F)`. Result publication copies exactly the
caller-required `R` full records; no second full-wait authority exists.

- `reactor/registration.cpp`: transitions the registry-owned fd state through
  logical re-arm coalescing, deferred-remove flushing, and generation reset;
  it owns no parallel registration container and does not include native
  headers or call native identity, duplication, close, or readiness syscalls
  directly.
- `reactor/generation.cpp`: stale fd-generation discovery and invalidation
  evidence for waits whose native fd number was reused after a new admitted
  socket generation; wait cleanup routes through the reactor cleanup owner.
- `reactor/lease.hpp`: the single ready-drain lifetime scope. Every network
  wait must acquire its nonempty current-generation socket lease; a typed host
  fd wait is explicitly classified and instead retains the reactor-owned fd
  identity guard. Missing waits, empty network views, and stale nonempty views
  fail the same scope rather than bypassing lifetime validation.
- `reactor/scratch.cpp`: reusable reactor scratch vectors for ordered ready
  events, drain batches, and host-event batches.
- `reactor/apply/policy.cpp`: scoped scheduler-side registration apply
  deferral for root prepared joins; it owns policy counters and state
  transitions, not native backend syscalls.
- `reactor/backlog.cpp`: canonical ready backlog for budget-deferred ready
  waits; it owns ordered suffix storage and stale-entry removal, not native
  polling.
- `reactor/budget.cpp`: deterministic ready-drain prefix selection after
  scheduler canonical ordering.
- `reactor/cleanup/route.cpp`: cleanup entrypoint router for single wait,
  many-group, and removed-wait cleanup.
- `reactor/cleanup/request.hpp`: cleanup request values and public cleanup
  entrypoints shared by scheduler owners.
- `reactor/cleanup/operations.hpp`: private declarations shared only by the
  focused reactor cleanup owners.
- `reactor/cleanup/group.cpp`: ReadyMany group cleanup owner; it stores
  selected events, cancels group timeout waits, removes registered sibling
  waits, erases event slots/groups, and wakes the owner once.
- `reactor/cleanup/wait.cpp`: single reactor wait cleanup owner; it
  removes backlog/registry state, applies registration cleanup, reroutes
  grouped waits, cancels paired timeout timers, and wakes the owner once.
- `reactor/cleanup/removed.cpp`: already-removed wait cleanup owner; it
  handles ready-backlog removal, group rerouting, timeout cleanup, cancellation
  stats, and owner wake dispatch after the registry wait is already gone.
- `reactor/cleanup/timeout.cpp`: paired timeout timer cancellation check
  shared by single-wait and removed-wait cleanup.
- `reactor/cleanup/wake.cpp`: cleanup wake owner for record result state,
  lossless admitted-task ready insertion, and `IoWake` evidence. Ready drain
  and every cleanup route call this owner; none reconstructs the record
  transition or evidence locally.
- `reactor/many/probe/raw.cpp`: raw batch ReadyMany immediate probe owner; it
  converts request records to raw fd probe records, consumes the platform
  probe disposition directly without a boolean result mirror, records
  observations and host events, and appends through the event-slot owner.
- `reactor/many/store.cpp`: canonical group-range lookup, `O(N log N)` sorted
  duplicate validation, one-time public-to-internal descriptor projection,
  consecutive wait-id lookup, and group-range erasure.
- `reactor/many/probe/dispatch.cpp`: Scheduler-facing borrowed-request immediate probe
  dispatch, output limit, direct event projection, and ready-event telemetry.
- `reactor/many/drain.cpp`: ReadyMany wait and timeout wake routing through
  the single reactor cleanup owner.
- `reactor/many/fairness.cpp`: ReadyMany primitive router only; it dispatches
  entry, immediate, park, and resume phases.
- `reactor/many/entry.cpp`: ReadyMany task context validation, stop-token
  checks, output limit calculation, sorted duplicate validation, and
  stale-generation cleanup over a borrowed descriptor span before immediate
  probing.
- `reactor/many/immediate.cpp`: immediate ReadyMany probe, deterministic
  copied-event ordering, no-timeout result, and zero-timeout result.
- `reactor/many/park.cpp`: ReadyMany park orchestration, record park state,
  observation records, one park-failure result projection, and published-group
  rollback through the reactor cleanup owner.
- `reactor/many/park/register.cpp`: exactly-once contiguous group request
  storage, consecutive wait-id assignment, event-slot alignment, atomic group
  storage publication, wait registration, and wait-registration observation
  records.
- `reactor/many/park/timer.cpp`: ReadyMany timeout timer setup and
  timed-wait evidence counters.
- `reactor/many/resume.cpp`: ReadyMany resume orchestration, group lookup,
  copied event routing, ready-code selection, and primitive commit.
- `reactor/many/resume/cleanup.cpp`: ReadyMany resume cleanup request
  construction and dispatch through the single reactor cleanup owner.
- `reactor/many/resume/record.cpp`: ReadyMany resume record wait-state reset and
  task-cancellation result construction.
- `reactor/many/resume/result.cpp`: ReadyMany resume final public result-code
  normalization and result payload construction.
- `reactor/many/events.cpp`: ReadyMany direct event-slot projection owner; each
  event stores its public socket identity, interest, insertion index, and
  status once, then bounded resume copies require no request-span lookup.
- `reactor/ready/set/result.cpp`: shared ready-set result materialization and
  current-generation validation helpers.
- `reactor/ready/set/identity.{hpp,cpp}`: the one ready-set capability
  validity, equality, live-state transition, generation retirement, and
  process-wide non-wrapping slot-id issuer. The public carrier shape is owned
  by `rund/net/ready/set/identity.hpp`; the compiled issuer is independent of
  Scheduler reset and deterministic replay state.
- `reactor/ready/set/model.hpp`: scheduler-owned set and member storage values.
- `reactor/ready/set/store.cpp`: live-set lookup, bounded member counts,
  duplicate checks, and member clearing.
- `reactor/ready/set/operations.hpp`: private declarations shared only by focused
  ready-set owners.
- `reactor/ready/set/lifecycle.cpp`: scheduler-owned per-set dense member
  storage, create/reuse, destroy, clear, and clear/destroy cancellation routing
  through the reactor cleanup owner.
- `reactor/ready/set/membership.cpp`: local duplicate checks, stable dense
  erase, insertion-index preservation, and stale-member removal validation.
- `reactor/ready/set/wait.cpp`: one-time ready-set membership snapshot into
  internal request scratch, stale-generation invalidation, and ready-set
  telemetry. Parked groups retain the snapshot independently of later set
  mutation and need no resume-time index remapping.
- `reactor/backend.cpp`: scheduler-facing reactor backend lifecycle and
  delegation of registration-change application through the queue owner and
  platform reactor boundary.
- `reactor/change/queue.cpp`: cursor-based queued registration apply owner; it
  snapshots the failed change identity before erasing an applied prefix,
  consumes the platform batch disposition without a boolean result mirror,
  drains successful prefixes without repeated front erases, and preserves
  best-effort invalid-remove semantics.
- The scheduler-facing apply result has exactly one disposition: `Success`,
  `Invalid`, `Failed`, or `BackendUnavailable`. Only `Invalid` carries the
  failed logical handle; every other disposition fixes that payload to the
  invalid-handle sentinel. A missing invalid handle is normalized to `Failed`
  instead of publishing an `Invalid` disposition without an identity.
  Platform errors and batch indices remain owned by the platform result that
  the apply boundary consumes.
- `reactor/close.cpp`: scheduler-side fd close invalidation discovery and
  invalidation evidence before native close; wait removal and wakeup are
  routed through the reactor cleanup owner.
- `reactor/stats.cpp`: live scheduler telemetry slot mutation helpers; it is
  not a second readiness or public snapshot authority.
- `reactor/invariants.cpp`: test-only read-only cleanup and cost snapshots over
  live scheduler state; it counts outstanding waits, timeout timers,
  ready-backlog entries, many groups, ready-set waits, dense member storage,
  validation comparisons, descriptor copies, and storage growth, and validates
  identity ownership between backlog wait ids, single-wait timeout timers, and
  many-wait timeout timer owners. It is scheduler evidence and is not a
  cleanup authority.
- `reactor/timeout.cpp`: timer-bounded reactor wait admission router.
- `reactor/timeout/immediate.cpp`: immediate timed reactor poll failure,
  invalid-fd, ready-result, zero-timeout, and host-event evidence.
- `reactor/timeout/park.cpp`: timed reactor wait/timer capacity checks, wait
  registration, timeout timer insertion, park records, and blocked-record
  state.
- `reactor/timeout/resume.cpp`: timed coroutine suspension and resumed IO
  result materialization.
- `reactor/timeout/storage.cpp`: reactor timeout timer reserve/cancel helpers
  and cancel-counter evidence shared by single and many waits.
- `reactor/timeout/validate.cpp`: timed reactor task, duration, fd,
  stop-token, host-handle, and fd-generation preflight.
- `reactor/timeout/wake.cpp`: timer-fired reactor timeout cleanup routing and
  cleanup failure evidence.
- `reactor/drain/batch.cpp`: deterministic ready-batch orchestration and
  aggregate registration-change collection over the registry-emitted
  canonical first-occurrence descriptor rows before backend apply.
- `reactor/expand.cpp`: platform-ready and failure expansion into scheduler wait
  events; it owns ready-vector capacity reservation, registry-native fd wait
  lookup, invalid-fd expansion, poll-failure expansion, and invalid-all
  expansion, but not cleanup, wake, or host-event recording.
- `reactor/drain/ready.cpp`: canonical ready batch drain and wake dispatcher;
  it records ready stats, cleans waits before observation and host-event
  recording, records host-event batches, and dispatches every record
  transition through `reactor/cleanup/wake.cpp`.
- `reactor/drain.cpp`: ready-reactor orchestration only: deferred-remove
  flushing, empty-wait handling, budget and backlog prefix selection,
  apply-policy decisions, backend apply/poll, ready ordering, budget suffix
  storage, and dispatch to the expansion and canonical drain owners.
- `timer.cpp`: timer park/resume primitives.
- `timer/store.cpp`: timer heap and wait-id index owner for deterministic
  `(deadline_ns, sequence, task_id)` ordering, due-pop, cancel-by-wait-id, and
  next-poll-timeout lookup.
- `io.cpp`: scheduler-side IO park/resume bridge through the fd reactor.

Raw fd configuration, raw batch immediate probes, and native backend syscall
owners are listed in [Platform](./platform.md).

## Timed Reactor Wait Ownership

Timer-bounded reactor waits are owned by `reactor/timeout.*`. A timed wait is
represented as one reactor wait and one reactor-timeout timer using the same
wait id. `timer.cpp` owns wall-clock deadline calculation and due-timer
ordering, then delegates reactor-timeout cleanup to the reactor timeout owner.
Exactly one side may complete the task. The losing reactor wait or timer entry
is removed before the task resumes.

Timed and many-wait ingress receives one scheduler-qualified `StopIdentity`.
Preflight validates that complete value against the active scheduler source
record and its requested state. Only then does the reactor retain its complete
`StopSourceIdentity` projection in the wait, paired timer, and many-wait group.
Cancellation joins those records by exact value equality; there are no
independent source-id, generation, or epoch fields that can drift apart.

When IO readiness wins, the ready drain cancels the paired timeout timer before
waking the task. When timeout, stop-token cancellation, fd close, stale
generation, ready-set invalidation, or many-wait sibling cleanup wins, the race
is completed by `reactor/cleanup/`. That owner removes the reactor
wait, invalidates the ready backlog entry for the wait id, cancels the paired
timeout timer when the request names one, removes sibling waits for a many-wait
group, applies registration cleanup, and enqueues exactly one owner-task wake.

Cleanup failure is reported as `IoPollFailed`; the public success/failure
reason remains the initiating reason (`TaskCancelled`, `IoTimedOut`,
`IoFdInvalid`, or `IoPollFailed`) unless backend cleanup fails. Coroutine tasks
resume through the scheduler-owned ready queue only after IO readiness,
timeout, invalidation, or cancellation wins the race.

The record's `io_result` reason is the only completion authority. In
particular, timeout is exactly `io_result == IoTimedOut`; no timeout boolean,
mirror state, or second reset path exists. Timed single-wait completion,
ReadyMany completion, cleanup failure precedence, and public `timed_out()`
therefore derive from the same value.

`tasks.reactor().timeout_timer_cancels()` counts every scheduler cancellation of a
paired reactor timeout timer, including IO readiness wins, close/generation
invalidation cleanup, many-wait sibling cleanup, and stop-token cancellation.
It is not limited to stop-token-triggered timer removals.

## IO API And Reactor Precedence

`readable` and `writable` accept an admitted owner's borrowed `io::FdView` and
request their interests through the scheduler-owned fd reactor. Public task code and
scheduler progress code do not own POSIX interest constants. A source-private
bridge unwraps the checked native descriptor and host id; the public task
headers expose no raw-integer readiness overload.

IO precedence:

1. malformed `io::FdView`: `io_fd_invalid`
2. no scheduler: `node_runtime_missing`
3. no task: `task_context_missing`
4. wait capacity full: `reactor_wait_capacity_exceeded`
5. parked: record `IoPark`

Persistent reactor registrations are keyed by native fd and aggregate the
interests of all live waits for that fd. When a ready wait is removed, a
zero-interest transition is first marked as a deferred native remove. If a
later wait on the same still-live fd re-arms the same aggregate interest before
the scheduler is otherwise idle, the deferred remove is cancelled and no
native remove/add churn is emitted. Deferred removes flush when scheduler ready
depth is zero; cleanup removes are best-effort, so an already-closed or
already-missing fd records cleanup evidence and does not fail unrelated work.
Kqueue applies paired read/write deletion with per-filter receipts: a missing
read filter cannot mask removal of a live write filter, or vice versa. The
platform registration projection is updated only after every non-missing
filter result succeeds.

Admitted network sockets carry an in-memory fd generation in addition to their
fd-derived host id. `WaitReactor` receives that generation from network
readiness calls. Before immediate probing or parking a new generation wait, the
generation owner invalidates older waits on the same fd whose generation no
longer matches. Those stale waits wake with `IoInvalid` and `io_fd_invalid`,
and registration state for the fd is reset so the new generation forces a
native add rather than modifying stale scheduler state. Plain typed
`rund::host::io::FdView` waits use generation `0`; the reactor snapshots their
`fstat` device, inode,
and mode identity and retains one close-on-exec duplicate while the native
registration is live. Retention prevents the old kernel object identity from
being recycled before a same-number replacement can be compared. Mutable file
timestamps are not identity: pipe reads and writes may change them while the
pipe remains live. A close followed by reuse of the same integer fd therefore
forces native re-registration, while a same-object re-arm still cancels the
deferred remove without backend churn. The retained descriptor is bounded by
the reactor registration capacity and is released on registration removal,
generation reset, or reactor teardown. Kqueue isolates replacement modifies
so its best-effort delete completes before the new filter is added.

Ready drain keeps those two lifetime mechanisms under one lease scope. An
admitted network wait acquires a socket-generation lease before registry
removal; a typed host-fd wait is explicitly marked as such and relies on the
registration's retained identity descriptor. An empty `SocketView` alone is
never treated as proof of a host fd, so malformed network waits and missing
registry waits remain fail-closed.

Ready-drain budgeting is applied only after backend ready events have been
expanded and sorted by the scheduler-owned key. A nonzero
`reactor_ready_budget` consumes at most that many canonical ready waits in one
drain; `0` uses ready-queue capacity as the per-drain budget. Available wake
space is derived from `task_capacity`, because these waits already own admitted
task records; the spawn backlog bound does not reject them. Budget deferral cannot
import native backend order because every prefix is selected from the already
canonical ready list. Reactor scratch vectors are scheduler-owned reusable
storage; scratch reuse changes allocation behavior only, not ready order,
evidence order, or replay meaning.

Reactor result mapping:

- invalid fd readiness: `IoInvalid`, `io_fd_invalid`
- error or hangup readiness: `IoReady`, `ok`, revents preserved
- requested interest readiness: `IoReady`, `ok`, revents preserved
- non-interrupted backend failure: `IoPollFailed`, `io_poll_failed`
- selected unavailable backend: `ReactorBackendUnavailable`,
  `reactor_backend_unavailable`

An expanded scheduler-ready value has exactly one disposition: `Ready`,
`Invalid`, or `PollFailed`. An immediate probe has exactly one disposition:
`NotReady`, `Ready`, `Invalid`, `PollFailed`, or `BackendUnavailable`. Only the
`Ready` and `Invalid` immediate factories carry backend event bits; every other
factory fixes them to `None`. The probe result does not retain wait id, task
id, fd, or interest; its two callers own those inputs. Platform-native
`invalid` bits remain local event-payload classifications and are projected
once at the scheduler boundary rather than mirrored as scheduler booleans.
Single-wait and ReadyMany probes consume the same exclusive platform probe
disposition; ReadyMany does not project it through a second result carrier.

Interrupted backend waits restart without an observation.

A native backend event is not itself scheduler progress. Registration removal
may race with an event that was already queued by kqueue, epoll, or the
portable poll boundary. If expansion maps a nonempty native batch to zero live
logical waits, the reactor drains that stale batch and polls again inside the
same fixed timeout window. It neither emits a host observation nor reports
quiescence. For a positive timeout `T`, every retry uses the ceiling of
`max(0, deadline - now)` rather than a fresh `T`, so stale native events cannot
extend timer precedence. With no timer the poll remains blocking; with a zero
timeout it consumes at most one native batch and reports scheduler activity so
already-ready tasks retain their turn before another nonblocking poll.

## Timer And IO Ordering

Timer waits keep a `std::chrono::steady_clock` deadline only for live waiting
against the host. Public and recorded timer time is logical: `deadline_ns` is
saturating addition of scheduler logical time plus `duration_ns`; overflow
clamps to `int64_t::max`. Deadline assignment ignores host elapsed time and
live wait readiness; it uses only scheduler logical time at the scheduler
primitive boundary. Negative durations are rejected before addition.

Due timers are observed only at scheduler boundaries or reactor/backend timeout
return. The scheduler observes the earliest logical timer first and uses its
steady-clock deadline only to decide whether that logical timer may wake.
Equal deadlines wake by:

```text
(deadline_ns, timer_sequence, task_id)
```

`timer_sequence` is assigned at `TimerPark` commit.

The scheduler reactor wait order is sorted by:

```text
(wait_id, task_id, fd, interest)
```

Multiple ready fds are processed in that order. The live storage is the
fixed-capacity slot arena in `ReactorRuntime::registry`; public
`reactor_wait_capacity` bounds its slots and `stats.reactor_waits()` reports
the same registry live count. `revents` is stored exactly as reported by the
reactor backend.

One ReadyMany park owns one contiguous request range and an aligned event-slot
range. Request wait ids are consecutive, so a wake resolves its member by
`wait_id - first_wait_id` in `O(1)` after the group lookup. Wake, cancellation,
timeout, and resume borrow the canonical range; no phase creates a second
group-wide request vector. Descriptor copies at park are therefore exactly
`N` for `N` members, independent of the number of ready events or sibling
cleanup operations.

Requests, aligned event slots, and the group record publish as one storage
transaction. A failure before group publication restores every prior live
prefix inside `park/register.cpp` and never calls published-group cleanup. Once
the group is published, wait- or timer-registration failure has one rollback
request builder in `park.cpp` and routes through the canonical reactor cleanup
owner. Neither rollback path rewinds an already-issued group, wait, timer-wait,
or timer-sequence cursor; Scheduler reset remains the lifecycle boundary that
reinitializes those cursors.

Duplicate validation sorts a bounded index projection by
`(fd, generation, interest)` and checks adjacent entries. Its comparison bound
is `O(N log N)`, replacing the pairwise `N(N-1)/2` scan, while the caller's
request order remains unchanged for probing and public event ordering.

## Ready Sets And Network Integration

Persistent ready sets are scheduler-owned fd readiness membership records.
They are persistent fd readiness carriers, not connection registries.
The public `Set` carrier remains one intact value through deferred ReadyMany
operations; no scheduler or network bridge decomposes it into mirrored id and
generation storage.
The `reactor/ready/set/` split owners cover create, destroy, clear, add,
remove, wait snapshot, insertion-index ordering, generation
revalidation, and ready-set telemetry. A ready set stores only fd readiness
membership: admitted fd, host handle id, fd generation, native interest, and
insertion index. It does not own session lists, connection registries, packet
dispatch, send queues, retry, rate limiting, matchmaking, or gameplay.

Each set owns one dense vector reserved to `max_members`. Stable erase compacts
storage without changing surviving insertion indexes. Clear returns live size
to zero and resets the next index while retaining the bounded capacity, so
steady add/remove churn does not allocate or grow storage. A parked wait owns
its pre-suspension membership snapshot, so later mutation affects only future
waits.

`ReactorReadySetIdentityOwner` is the sole authority for capability validity,
exact-pair matching, and create/destroy transitions. Scheduler storage keeps
the public pair and its live bit in one identity state; lookup, ready-set wait
groups, cancellation, resource counts, and lifecycle code consume that owner
rather than comparing independent id/generation fields. The live bit is the
state authority: odd generation is necessary for a live incarnation, but the
maximum odd generation can also be a retired non-live tombstone. Reusable even
tombstones are selected before maximum-generation tombstones that require a
new process id, so issuer exhaustion cannot strand locally reusable storage.

All vector growth and member reservation complete before a process id is
issued. After issuance, publication consists only of non-throwing identity and
configuration assignments. The process issuer uses one compiled atomic CAS
owner, emits `[1, UINT64_MAX-1]` once, and permanently treats `UINT64_MAX` as
exhausted. It is deliberately excluded from replay fingerprints. The
fingerprint retains the former deterministic scheduler-local meaning through
the physical ready-set slot count plus one; activity in another Session cannot
make an input capture appear mutated.

Network callers see only the meaning-neutral ready-set API routed through the
network contract. Public telemetry includes `ready_set_creates`,
`ready_set_destroys`, `ready_set_members`, `ready_set_waits`,
`ready_set_ready_events`, and `ready_set_invalidations`.
`ready_set_members_added()` is a compatibility projection of the canonical
cumulative `ready_set_members()` slot, and `ready_set_events()` is a
compatibility projection of the canonical cumulative
`ready_set_ready_events()` slot. They do not own independently mutable
counters. Physical slots 186 and 190 remain reserved only to preserve the
234-slot, 1,872-byte `Stats` layout; production code neither writes those
reservations nor projects a current getter from them. A matched exact SDK
artifact supplies both headers and the static library, so no old-header/new-
library mixed tuple is supported. `ready_set_members_removed()` remains a
separate cumulative removal-event counter, while
`resources().live_ready_set_members()` remains the snapshot gauge derived from
the current ready-set store. Add, remove, clear, and destroy therefore need not
make those two distinct meanings equal.

Every additive evidence mutation in the reactor implementation consumes
`rund::detail::counter::Accumulate`; `UINT64_MAX` is absorbing and no reactor
counter wraps. `max_backlog_depth` remains a maximum gauge rather than an
addition, and wait/timer identity sequences are not telemetry counters.

UDP datagrams, selected socket options, vectored IO, and network resource
limit checks are routed by [Network](../net.md): datagrams are address-byte
observations, socket options are selected server-operation options only,
vectored IO uses caller-owned slices without scheduler-owned queues or retries,
and resource limits fail closed with the documented reason codes.

## Backend And Budget Evidence

Root prepared joins may defer opportunistic zero-time native registration
apply while scheduler-ready tasks are still parking fd waits. Add-only native
registration changes inside the prepared-join batch scope may also wait until
the batch boundary when the scheduler is otherwise idle. Blocking or
timer-bounded reactor polling forces a flush before native polling. This
changes native backend apply batching only; wait identity, canonical ordering,
host evidence, and replay meaning stay scheduler-owned.

When a ready budget consumes only a canonical prefix, the scheduler stores the
remaining canonical suffix in a ready backlog. Later drains consume backlog
entries before a new native poll. Backlog entries are already
scheduler-ordered and therefore do not import backend ready order. Generation
invalidation and reactor close remove affected backlog entries.

Cross-set wake order is a contract only when the producer establishes a causal
boundary between deliveries, such as completing the first waiter before the
next delivery. If multiple sets are ready in one native observation, the
canonical scheduler order above is authoritative; elapsed delays do not create
an ordering edge.

The reactor backend parity benchmark is compiled-backend evidence only. It
proves scheduler-visible canonical wake order for the backend selected by the
current build and does not assert identical wall-clock wake timing, identical
backend ready-list order, or identical native readiness schedules across
operating systems.
