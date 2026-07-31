# Coroutine Task Contract

This page owns the only suspending task representation in Node: native C++20
`Task<T>` coroutine frames scheduled by the runtime.

## Authority

The execution path is:

```text
Task<T> frame
  -> scheduler task record
  -> admitted -> ready -> running
  -> parked <-> ready
  -> committing
  -> completed | failed | cancelled
```

There is no alternate suspending task representation, future-backed execution
job, or runtime mode router.

## Frame Ownership

Coroutine frames are allocated from the active scheduler's bounded reusable
frame arena. Admission validates size and alignment. Exhaustion, oversized
frames, and over-aligned frames fail without redirecting ownership to a
general-heap frame. A frame is destroyed exactly once after its result has
been committed to the matching completion cell.

Primitive operations retain a private `Status`, payload where required,
and a private await decision only. Public result objects never carry suspend
flags, deferred flags, task ids, fd identities, or frame-control metadata.
Primitive awaiters never embed the runtime's `task::Stats`: telemetry is owned once by the runtime/scope report,
not once per suspended frame. Frame capacity therefore scales with coroutine
locals and awaited payloads rather than with the number of scheduler counters.
The checked-in scheduler and host-operation coroutine frames fit the default
4096-byte maximum. Native Task promise frames require 16-byte alignment by
default. Frames up to 160 bytes use lazy 256-slot compact pages and do not pay
the maximum-frame stride. A compact slot carries a 16-byte ownership prefix.
At the default 16-byte alignment its stride is
`align_up(16 + 160, 16) = 176` bytes. The wide
tier creates scheduler-owned slots lazily, never beyond `task_capacity`, and
retains them for allocation-free warm reuse.
The first cold admission of a new wide slot may allocate its arena storage;
there is no per-resume allocation or unowned fallback. Larger application
frames request an explicit bounded size.

Completion cells keep phase, generation, observer, and waiter metadata in a
compact fixed array. `Task<void>` owns no typed-result payload. A `Task<T>`
creates its scheduler-owned result slot on first use, bounded by
`task_result_bytes`, and the slot remains attached to that cell for
allocation-free warm reuse. Coroutine join-wait storage is reserved once to
`task_capacity` during scheduler configuration, so a valid post-configuration
join park cannot allocate or lose a live target to allocation failure.
Join-many scratch and root-retire ranges grow only when those root operations
are used and retain their bounded storage; ordinary leaf tasks do not prepay
those root-only capacities.

Once admitted, a parked coroutine owns one of at most `task_capacity` task
records and wakes through the task-capacity-sized global ready queue. The
smaller `ready_queue_capacity` bound applies only to independent-spawn backlog
admission. A wake therefore cannot fail after the coroutine has published
borrowed frame state, and it cannot strand an upper joiner or destroy a parent
frame while a child still borrows it.

A `co_await Task<T>` child is a continuation of an already-admitted parent,
not an independent backlog submission. It still claims a task record and
completion cell under `task_capacity`, but enters the task-capacity-sized
global queue even when the independent-spawn backlog has reached
`ready_queue_capacity`. Explicit `spawn(...)` calls, including calls made by a
running task, remain independent submissions and retain the smaller bound.
This distinction makes admitted coroutine structure progress-safe without
creating an unbounded queue or weakening caller-controlled fan-out pressure.

A Runtime Compute submission is also independent admission, but its external
`wait()` needs the scheduler completion observer. The scheduler therefore
uses the same coroutine materialization owner with spawn admission and returns
the handle and observer together; it does not misclassify the coordinator as
an awaited continuation or maintain a second completion mechanism.

Leaf callables likewise do not occupy every task record. The scheduler stores
them in a bounded 256-slot page pool that grows only while leaf tasks are live
and retains pages for allocation-free warm reuse. Coroutine-only workloads
therefore pay one pointer per task record instead of the leaf callable's
inline payload. Lane workers destroy their owned callable after execution, and
the deterministic commit phase recycles the already-destroyed slot. This keeps
user callable destruction parallel without a shared pool lock while leaving
free-list mutation under the scheduler sequencer. Join, channel, and reactor
parking also share one wait-source slot because `TaskState` admits only one
parked source at a time. External and direct wakes share one intrusive queue
link for the same reason. The task index encodes empty, deleted, and occupied
state in one 32-bit slot value. The checked 64-bit layout ratchet keeps
`TaskRecord` exactly 96 bytes. Task identity is the scheduler id plus the
monotonically issued task id; record-slot reuse cannot alias an older handle,
so no second per-record hash token is retained. Parent identity is consumed by
spawn evidence before admission returns instead of remaining in every live
record. The non-empty task name follows the same rule: admission folds its
stable hash into canonical spawn-order evidence and retains no name pointer or
string bytes in `TaskRecord`.
Leaf callable
storage and coroutine frame storage are mutually exclusive payloads.
Coroutine result/destroy behavior uses one static ops table, and the record
stores only the completion slot and generation because the Scheduler already
owns the completion-pool authority. Wait-source tokens and direct-wake
sequence numbers share state-discriminated storage; logical wait identity and
the retired-record free link likewise share storage because their task states
are disjoint. The wake sequence is written only after the blocked-source
identity has been validated. Failure state is
retained as `ReasonCode`, not a string pointer, and is
materialized as text only at the public result boundary. Unused per-record
resume and operation counters are not retained as dormant payload.

Frame addresses and raw frame bytes are physical storage, not logical
identity. Replay and deterministic evidence use task id, generation, wait id,
and logical operation identity.

## `Task<T>` and Completion

`Task<void>` and `Task<T>` share the same promise and scheduler path. A plain
`Task<void>` observed through `Handle` uses the task record's existing terminal
state and does not claim a completion cell. Nested or externally observed
`Task<T>` claims one completion cell. The pool creates 256-cell pages on first
claim and retains them for warm reuse; configuring `task_capacity` alone does
not materialize one cell per possible task. A claimed cell owns phase, failure
code, typed result storage, external wait notification, and coroutine/task
waiters. Expected operational failures are
reported with `ReasonCode`/result status. Unexpected C++ exceptions are caught
at the task boundary and converted to task failure.

Nested `co_await Task<T>` admits the child through the same scheduler, parks
the parent on the child's completion, and resumes the parent exactly once.
After the parent copies the bounded typed result, the terminal child task
record is retired immediately. Sequential nested awaits therefore reuse the
configured task-record and coroutine-frame pools instead of consuming
`task_capacity` cumulatively.

A status-valued `Task<Status>` resumes directly with `Status`. The
awaiter maps its own admission, execution, or completion failure and the
returned status into that one value; it never exposes
`Result<Status>`. Other value-bearing `Task<T>` instances resume with
`Result<T>`. This semantic flattening gives every operation one outcome
authority without changing the scheduler's typed completion storage.
The scheduler publishes that typed completion before making its parent join
waiter ready. In happens-before terms, the completion cell's terminal publish
precedes the ready-queue release; a resumed parent can therefore read exactly
one terminal value and never race a still-committing cell.

## Await Contract

Suspending primitives follow standard coroutine staging:

```cpp fragment
bool await_ready();
bool await_suspend(std::coroutine_handle<> continuation);
Result await_resume();
```

Construction of a yield, sleep, readiness, or join operation does not park a task.
`await_suspend` validates the current task, registers the wait source, records
the continuation ownership in the current task record, and atomically changes
the task to a parked state. A wake returns that same record to the ready queue.

Implemented await families are:

- yield;
- sleep;
- single-fd readiness;
- timed readiness;
- multi-fd readiness;
- channel send and receive;
- task handle join;
- nested `Task<T>`;
- Runtime Compute coordinator external completion.

Discarding a deferred await token consumes no timer/reactor capacity and emits
no park evidence.

## Leaf Tasks

`spawn(lambda)` is a non-suspending leaf. It runs to completion while occupying
its scheduler lane. A leaf must not call task suspension primitives or blocking
syscalls. A primitive attempt fails the task with
`task_leaf_primitive_forbidden`. Suspending work must be expressed as
`Task<T>`.

## Wake, Cancellation, and Shutdown

Every wait has one scheduler-owned record and one logical wake. Wake sources
may be timer, reactor, channel, child completion, or accelerator callback, but
they only make the parked task ready; they do not resume a frame directly.
When direct lane work can publish additional ready tasks, the reactor performs
a nonblocking readiness probe and the root coordinator waits on the scheduler
completion epoch. It never enters an indefinite fd poll that can hide a ready
coroutine behind unrelated I/O.

Cancellation is observed at defined scheduler boundaries. Runtime drain stops
new admission, lets admitted work reach a terminal state, and destroys frames
and completion cells exactly once. Callback-after-destroy and duplicate wake
are invalid transitions.

## Verification

The `node-runtime` coroutine contracts cover typed results, nested await,
yield/sleep, channel, single and multi readiness, cancellation, frame capacity,
frame reuse, completion state transitions, discarded-operation non-admission,
and leaf primitive rejection. Lifecycle semantics, nested typed-result reuse,
and those operation boundaries have one `runtime.task.coroutine` owner;
frame-arena and completion-cell internals remain separate focused owners.
Compute callback parking and wake evidence are covered by `node-compute` and
runtime Compute contracts.
