# Scheduler Progress

This page owns root/scope execution, ready progress, yield/sleep progress,
join and scope waits, recursive progress, deadlock wake behavior, and final
drain failure propagation.

## Source Map

- `state/storage/ready/queue.hpp`: the single fixed-capacity intrusive-index
  ready queue.
- `progress/ready/queue.cpp`: O(1) front selection and restore, O(1)
  ready-depth accounting, and bounded scope-specific selection.
- `progress/ready/wait.cpp`: direct-job waits, timer sleep, scheduler-owned fd
  reactor waits, and deadlock wake.
- `progress/ready/batch.cpp`: deterministic ready-batch construction for lane
  dispatch.
- `progress/ready/root.cpp`: root single-join ready-target fast path.
- `progress/ready/step.cpp`: one progress-step orchestration from ready
  selection to dispatch, including lossless pop-to-dispatch ownership and the
  final offload completion harvest after direct-job wait races.
- `progress/drain.cpp`: final scheduler drain, replay-tail mismatch checks,
  and final failure materialization.
- `progress/join.cpp`: single-handle join validation, task-owned join park,
  root/control single-join progress, and terminal join result materialization.
- `progress/join/many.cpp`: prepared task-group joins and dynamic-width
  `join_all` progress.
- `progress/join/wake.cpp`: stable linear join-wait compaction, ready enqueue,
  and `JoinWake` evidence.
- `progress/scope.cpp`: child scope enter/park/wake completion authority.
- `state/progress.hpp`: progress-loop result contracts.

Lane dispatch internals are owned by [Lane](./lane.md), and fd reactor waits
are owned by [Reactor](./reactor.md).

## Scope And Root Execution

Only one `Session::scope` may be active for a `Session`. A second concurrent
scope fails with `runtime_scope_busy` before installing providers or a
scheduler.

Root callback code runs only on the scope thread while no scheduler batch is in
flight. Root `spawn`, `channel<T>::make`, and `channel.close()` are immediate
sequencer submissions in C++ call order. Root code cannot call `yield`,
`sleep`, `send`, `recv`, or IO wait APIs because those require task context.

Node-native Compute coordinators may finish on lanes outside a root Scope. A
root wait enters the commit sequencer only after its target is terminal and
holds that ticket through result materialization and record retirement. It
does not hold a ticket while waiting for the target or driving progress, so an
unrelated coordinator remains independently executable.

Scheduler batches run only when root calls `join`, `scope`, or when
`Session::scope` performs its final drain. Root submissions cannot interleave
with an in-flight batch.

## Runtime Progress Model

The scheduler adopts park/ready transitions for runtime-aware blocking
primitives, one canonical ready FIFO, channel-owned sender and receiver wait
queues, timer wait nodes, reactor wait nodes for low-level fd readiness,
lane-local primitive trap packets, and deterministic effect-log merge by
canonical task/ticket order, not OS completion order.

Ready progress is driven by ready queue nonempty, running batch, timer heap
nonempty, IO wait set nonempty, channel immediate completion, terminal join
target, or terminal scope target. The same FIFO owns independent spawns,
structural child admission, semantic side exits, primitive wakes, restore, and
scope-specific progress.

`ready_queue_capacity = R` is the admission backlog bound for independent
spawns. The FIFO reserves `task_capacity = T` physical slots because at most
`T` task records exist and one task has at most one ready entry. An
already-admitted task waking from yield, timer, channel, join, or reactor state
enters that queue without re-admission. Logical ready depth may exceed `R`
after wakes but never `T`; while depth is at least `R`, a new independent spawn
fails with `ReadyQueueCapacityExceeded`. An awaited `Task<T>` child is
structural continuation work for its already-admitted parent: it must still
claim one of the `T` task records, then enters the same FIFO under the same
`T` proof as a wake. It cannot bypass `task_capacity`, and an explicit
`spawn(...)` inside a task remains an independent spawn. Ordinary enqueue,
front pop, and skipped-ready restore move only the cursor and one slot; they
perform no front compaction or warm allocation.
Scope-specific progress and cancellation may scan a bounded queue to locate a
particular logical task. Removal then relinks two fixed indices; it never moves
the remaining entries. Those control paths are not the ordinary ready-pop
path.

`EnqueueProgress` is the guaranteed queue owner for an indexed
`TaskRecord` transitioned to `Ready`, whether it is an awaited child
continuation or a parked task being woken. It has no boolean failure surface.
The state machine permits one ready entry per task, so a duplicate enqueue is
an internal invariant violation rather than another queue result. The bounded
ratchet yields one root-submitted coroutine 32 times with `R=1,T=2` and proves
exactly 32 yields, 33 resumes, one live ready entry at high water, rejection of
a second spawn, and 33 global pushes/pops: one for initial root admission plus
32 for the wakes. Queue telemetry exposes physical selection traffic as
`global_ready_queue_pushes/pops` and separates the two admission causes as
`ready_spawn_pushes` and `ready_progress_pushes`. Pushes count successful
spawn/progress admissions. Pops count valid physical selections, including a
selection later restored at the front after a lane-availability race; that
restore is not a new admission and therefore is not another push. The counters
need not be equal under physical backpressure even though logical ready order
and committed effects remain deterministic.

The task-id index uses fixed-capacity open addressing with explicit empty,
occupied, and deleted slot states. Retirement marks one slot deleted in O(1)
instead of backward-shifting or scanning the remaining collision cluster. When
the final live index entry retires, one bounded O(capacity) clear removes all
deleted slots before the next admission epoch. If long-lived tasks keep an
epoch open, deleted slots are rebuilt in the same fixed storage only after
retirements amortize that bounded scan; admission never grows or replaces the
index storage.

Root progress may continue selecting ready calculation while earlier direct
jobs are in flight. Any task that reaches a scheduler primitive or runtime
state query first acquires its canonical commit ticket, so timer and evidence
mutation remains ordered without serializing pure user calculation. A control
path waits for outstanding direct jobs only before an exclusive lifecycle or
terminal boundary that cannot overlap them.

A successful ready pop transfers one task id into the progress step, not
directly into a lane. Exactly one of these state transitions must follow:

```text
ready queue -> progress step -> lane
ready queue <- progress step    (temporary lane backpressure)
```

The lane availability probe and lane submission use different lock intervals,
so availability can change between them. Submission owns the task only after
the lane accepts it. If submission is rejected and the record is still
`Ready`, the progress step restores the same id at the front while holding the
scheduler state lock. A worker may otherwise complete or retire the record in
that interval; observing that state change is itself progress and does not
create another ready entry. The restore path performs no allocation and moves
one fixed-ring cursor and one id.

At a root no-progress boundary, dependency deadlock is possible only when all
runtime progress sources are empty:

```text
ready_depth = 0
and timer_count = 0
and reactor_wait_count = 0
and direct_jobs_in_flight = 0
```

Scoped progress uses a narrower predicate. A Ready task in another scope does
not prove that the selected scope can progress. Ready selection therefore
returns both the selected id and whether an eligible id was temporarily blocked
by its lane. Only that scope-local blocked bit causes a retry; otherwise timer
activity, reactor activity, and direct jobs are evaluated before deadlock.
This prevents both a false deadlock after a wake and a busy loop caused by an
unrelated scope's ready task.

Unavailable ready ids use one control-thread vector reserved to
`task_capacity` before lane startup. Root join, root scope, and final drain are
the only progress owners; task-side synchronous join and scope reject before
entering `Step`. A progress step clears and reuses that storage, restores
skipped ids in reverse-pop order, and never constructs a per-step deque or
grows storage on the ready hot path. Since at most one ready entry exists for
each of at most `task_capacity` records, exhausting the reserved vector is an
internal invariant violation rather than an allocation fallback.
Root multi-lane batch construction and external-prefix dispatch reuse one
retained `ready_deferred` vector at mutually exclusive sequencer boundaries.
It is reserved to `task_capacity`, the maximum possible canonical prefix, and
never grows on the warm path. The active ready batch uses a different vector,
so batch assembly cannot alias its own deferred ids.

For root progress, a nonzero ready depth is a progress source. For scoped
progress, only an eligible ready id or eligible lane backpressure is a ready
progress source. The installed SDK server consumer proves the direct nested
peer-handler journey once; deterministic scheduler contracts own queue depth,
scope-local timer progress beside an unrelated ready task, and allocation-free
warm batch reuse.

## Sleep

Sleep precedence:

1. no scheduler: `node_runtime_missing`
2. no task: `task_context_missing`
3. negative duration: `timer_duration_invalid`
4. zero duration: `ok`; record `SleepZero`; no timer capacity consumed
5. timer heap full: `timer_capacity_exceeded`
6. parked: record `TimerPark`; later `TimerWake` with `ok`

`co_await rund::task::sleep(duration)` parks the coroutine task by recording
timer evidence, leaves the frame suspended on the existing timer queue, and
wakes through the scheduler timer authority. A later scheduler
quantum resumes the coroutine frame, runs code after the await, and accounts
coroutine park/resume/wake state, completions, and frame destruction.

## Yield

`task::yield` records scheduler-owned yield evidence and parks or resumes
through the same ready authority as other scheduler primitives. If a record
parks with `yield` and the ready queue front is the same record on the same home
lane, the lane consumes that ready entry, advances the commit ticket
deterministically, and resumes the record locally without returning through root
ready selection.

Yield records one deterministic logical event. A suspended coroutine resumes
through the ready queue; a leaf task cannot call `yield`.

## Join, Scope, And Deadlock

Join validation precedence:

1. invalid handle object: `task_handle_invalid`
2. failed `spawn` handle: preserves that failed submission reason
3. stale, wrong scheduler, or unknown task id: `task_handle_unknown`
4. duplicate handles: allowed and observed once by terminal checks
5. terminal successful target: contributes no wait
6. terminal failed target: join returns that target failure reason
7. target currently running while called from a task: `task_deadlock`
8. otherwise recursively steps scheduler progress until all targets are
   terminal or no progress source exists

The synchronous `join`/`join_all` surface returns `Status` and is a root
progress operation. Calling it directly from a coroutine task returns
`TaskCoroutinePrimitiveAwaitNotLive` without registering a waiter, changing
the task to parked, or entering a progress loop. Only `co_await Handle` invokes
the source-private join-await admission that registers one waiter and parks the
current coroutine.

The live scheduler does not maintain a separate public wait edge graph. It
detects task-owned running-target cycles directly: a task attempting to join
itself or another task that is already running in the current scheduler batch
or recursive scheduler call stack fails with `task_deadlock`. Cross-scope joins
are allowed when the handles belong to the same scheduler and do not hit that
running-target guard.

`join_all(std::span<const Handle>)` is the dynamic-width join surface. It uses
the same validation, duplicate-handle, terminal-target, failure, progress, and
deadlock rules as variadic `join`, but the handle set is supplied at run time by
the span's current contents. An empty span is a successful no-op join. The span
is consumed only for the duration of the call; it is not retained by the
scheduler.

`Group` is the coroutine fan-out/fan-in owner. Construction requires a
caller-provided `span<Handle>` and therefore performs no hidden allocation.
The group is neither copyable nor movable: one local coordinator owns one slot
span and an active join state can never retain a pointer to a relocated group.
`spawn` fills those fixed slots and rejects overflow. `co_await group.join()`
awaits handles in stable slot order, reports the first failing slot, retires
each terminal record after observation, clears the group for reuse, and
resumes with one `Status`. The join coroutine's own completion failure and
the first child failure are flattened at the await boundary; callers never
inspect a `Result<Status>` or a status nested inside another result.
It is not a scheduler scope and creates no second wait or cancellation
authority.
Root-thread dynamic fan-in uses `join_all(span<Handle>)`.

The scheduler reserves exactly `task_capacity` join-wait records during
configuration. A single-target envelope can park `C - 1` waiters beside its
one live target for task capacity `C`; registration remains inside the reserved
storage and performs zero heap allocation after configuration. The full `C`
bound also covers a cycle in which every live task owns one wait edge. Each
record is three `uint64_t` coordinates, so its checked payload model is `24C`
bytes (24,576 bytes at the default `C = 1,024`), paid once to remove allocation
failure from the borrowed-lifetime join path. Target
completion scans the current `J` records once. Matching waiters are woken in
admission order while nonmatching records are written back in their original
order. The work is at most `J` reads, `J - K` retained-record writes, and `K`
wakes for `K` matches: `O(J)` time and `O(1)` auxiliary storage. Repeated
`vector::erase` would shift up to
`sum(i=0..J-1, i) = J(J-1)/2` records when all waiters match; that quadratic
authority is not present. The mixed-target contract admits waiters in
`A0,B0,A1,B1` order, completes `A` first, and ratchets wake order
`A0,A1,B0,B1`; removing `A` therefore cannot reorder the retained `B` edges.

Join wake uses the same admitted-task invariant as every primitive wake. It
cannot be rejected by the smaller spawn-admission backlog bound, so an upper
joiner cannot remain parked and a frame-owned value borrowed by a child is not
destroyed during wake. The one stable `O(J)` scan is the complete join-wake
path.

Hot owners that must not create the convenience `join()` coroutine use the
same ordering and first-failure authority through `begin_join()`. Its
move-only `Group::JoinState` carries only the group pointer, current slot,
and first failure; the outer coroutine awaits `current()`, calls `advance()`,
then calls `finish()`. This direct form adds zero Task records and zero
coroutine frames. `finish()` clears the group only after every admitted handle
was observed. A second `begin_join()` while a join is active returns an inert
state whose `finish()` reports `TaskInvalid`; `spawn` is rejected and `clear`
is a no-op until the active state finishes. Moving a state transfers the sole
join authority, while the moved-from state is inert: `pending()` is false,
`current()` is invalid, and `finish()` reports `TaskInvalid`. Join state is not
copyable or move-assignable. The caller-owned handle slots and the `Group`
must remain alive and unmodified from `begin_join()` through `finish()`.
On the checked 64-bit ABI, `Group` is 32 bytes and its move-only
`JoinState` is 24 bytes; exact layout contracts ratchet both values.
`JoinState::finish()` returns the same direct `Status` as the convenience
await, so hot owners using `begin_join()` do not acquire a second outcome
shape.

`scope(fn)` is a scheduler-control operation. Root code may create a child
scope; descendant membership follows the active scope at spawn time. After
`fn`, the control thread recursively steps scheduler progress for nonterminal
tasks in that child scope until the scope is terminal or no progress source
exists. A task cannot synchronously drive a nested scheduler from its worker
lane: a leaf receives `task_leaf_primitive_forbidden`, a coroutine receives
`task_coroutine_primitive_await_not_live`, and the scope callback is not
invoked. Task-owned fan-out/fan-in uses coroutine await or `Group`, so there is
no second recursive progress owner or shared progress scratch.

Scope success occurs when the descendant set is empty. `task::scope` failure
is:

- callback throw: `task_scope_callback_failed`
- descendant failure: first descendant failure by task id
- running-target guard or deadlock: `task_deadlock`

Root owner id is `0`; user tasks cannot wait on root id `0`.

If no progress source exists and any wait state remains, the scheduler emits
`DeadlockWake` victims in `wait_sequence` order. Parked primitives resume with
false `task_deadlock`; this does not automatically fail the task. Uncaught
exceptions and fatal record errors fail the task.

A later real channel wake for the same task and channel supersedes a pending
speculative `DeadlockWake` result if the task has not resumed yet. This keeps
close or counterpart delivery from leaking a stale `task_deadlock` into a
parked channel call after the channel has produced a real wake source.

Failure propagation:

- `join` reports target failures only.
- `scope` reports descendant failures only.
- unrelated failures discovered during global progress are deferred to their
  owner or root drain.
- root final drain reports the first unreported task failure or global
  deadlock.
