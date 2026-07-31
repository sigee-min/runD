# Session Product Contract

## Scope

This page owns the current public `rund::Session`, `Session::scope`, and
`rund::run` behavior. Session is a domain-neutral execution host. It owns
lifecycle, discovered resources, task/Compute integration, bounded trace, and
replay evidence; it does not own a domain run proposal, admission ledger,
workspace lease protocol, or physics/game state.

Public authority:

- `/node/include/rund/session.hpp`
- `/node/include/rund/session/config.hpp`
- `/node/include/rund/session/run.hpp`
- `/node/include/node/runtime/runtime.hpp` for the source-private compiled owner
- `/node/src/runtime/session/state.hpp`
- `/node/src/runtime/compute/` for the compiled Session Compute bridge
- `/node/include/node/runtime/backend.hpp`
- `/node/include/rund/session/trace.hpp`
- `/node/include/node/runtime/replay/`
- `/node/include/rund/task/` for transitive task support reached through
  `<rund/task.hpp>`, not as additional product entrypoints

Session evidence exposes the independent `task/observation.hpp` owner.
Scheduler operation tags and direct scalar hashing remain non-product task
support; no operation record is part of the Session or installed task surface.
Replay record, comparison, decoded result, and
host evidence use `replay/model.hpp`, `replay/check.hpp`,
`replay/codec/result.hpp`, and `replay/host/evidence.hpp`. No generic type
aggregate is part of the source or installed surface.

Implementation authority is `/node/src/runtime`. Verification authority is the
runtime, runtime-compute, task, replay, and installed-consumer contract owners.
Scheduler semantics remain in [`scheduler/README.md`](./scheduler/README.md),
host semantics in [`host.md`](./host.md), and Compute execution in
[`accel.md`](./accel.md).

## Product Surface

`SessionConfig` directly owns:

- non-zero session `id`
- backend policy and requested worker width
- optional verified NUMA, affinity, and worker-capacity requirements
- one telemetry `Sink`
- trace capacity
- scheduler and replay configuration
- deterministic random seed

There is no nested runtime config, manual run proposal, direct Session run
method, public workspace pool, or separate resource default. Successful
`Session::open` stores the discovered `ResourceEnvelope` once; `resources()`
is an explicit read-only observation of that owner.

`Session::open`, `drain`, and `close` return `Session::Status`. Ordinary
reusable-session code ends with `close()` alone. `drain()` remains the
deliberate admission-shutdown boundary for callers that need to inspect the
Draining state before final close; it is not a prerequisite for `close()`.
`Session::scope` and `rund::run` return the same move-only `Session::Result`.
Every outcome stores exactly one `ReasonCode`; `code()` observes it, `Ok` is
the sole success value, `ok()` and truth conversion derive from it, `error()`
projects stable text only for a non-`Ok` code, and `exit_code()` is `0` or `1`.
`Session::Status::state()` observes the lifecycle state paired with that
decision. Both fields and their paired constructor are private, and one
runtime access owner constructs them together. Callers therefore cannot
manufacture an inconsistent public outcome by selecting a success code with a
failed lifecycle state. The public default is the single fail-closed
`NotConfigured`/`Unconfigured` value used for deferred storage. There is no
stored bool, reason string, secondary result projection, or public payload
archive.
A default `Session::Result` fails closed with
`SessionResultMissing` and `session_result_missing`.

Private replay evidence is constructed directly in a bounded inline Result
slot. The compiled owner proves the archive's size, alignment, nothrow move,
and nothrow destruction at compile time; no heap indirection is introduced at
this boundary. The warmed empty-scope allocation contract remains the runtime
proof that returning a Result does not add an allocation.

`scope()` is the admitted physical scope identifier used for diagnostics and
telemetry correlation. It is not replay identity and does not contribute to a
replay hash. `tasks()`, `memory()`, `observations()`, `events()`, and `trace()`
are the one public result projections; `trace_hash()` is derived from the
retained trace rather than stored as a second authority.

`Session::Result` has no public exact `sizeof` or alignment constant. Its
move-only ownership and observer semantics are the contract; standard-library
containers and private telemetry may change its physical size. Binary
consumption is admitted by the exact installed artifact tuple, not by a second
source assertion. Performance packets may observe Result size for one artifact
but cannot promote that observation into SDK ABI.

## Configuration Transaction

`Session::open()` has a strong state boundary. Discovery, trace reservation,
Compute host construction, Scheduler storage, replay storage, worker evidence,
and warm Compute task slots are prepared in a local internal Runtime. The
internal state changes from `Unconfigured` through `Configured` to `Running`
only after each preparation and start succeeds. Allocation or Scheduler
preparation failure destroys the candidate and returns
`task_scheduler_allocation_failed`; the same Session remains unopened and may
be opened again.

Replay storage metadata is copied into a local value before Runtime
preparation. After start, that value and the scalar replay limits move into
the existing empty Session state through statically proved non-throwing
assignments; the live Runtime pointer is the last commit. There is therefore
no post-commit throwing operation that can report failure while leaving a
Running Session behind. A repeated `open()` delegates to the Runtime lifecycle
owner, which returns `AlreadyConfigured` with the synchronized actual state;
Session does not infer configuration from pointer presence.

The lifecycle enum is the sole configuration authority. A second `configured`
bit is forbidden because it would permit disagreement with the five-state
machine. The Compute host's internal prepared flag is separate: it describes
that host object's Scheduler preparation, not Session lifecycle.

## Lifecycle And Snapshot

The state machine is:

```text
Unconfigured -> Configured -> Running -> Draining -> Stopped
```

- `Session::open()` performs the internal `configure -> start` transition and
  is valid only for an unopened Session.
- `Session::drain()` is valid only from `Running`, stops new work, and leaves
  the observable state at `Draining`. This closes resident Compute and bound
  compile-service admission before returning; already admitted work retains
  its normal completion owner.
- `Session::close()` from `Running` performs the deterministic
  `Running -> Draining -> Stopped` path in one blocking call. It closes
  admission first, waits for an active scope to leave, and delegates active
  Compute cancellation and drain to the blocking Compute-host terminal owner
  before it returns success at `Stopped`.
- `Session::close()` from `Draining` waits for the same quiescence boundary and
  performs only the `Stopped` transition. It neither repeats the drain
  transition nor emits a second draining trace record.
- `Session::close()` from `Stopped` is an idempotent success with no state,
  trace, or resource mutation. `drain()` remains a strict transition and is
  rejected outside `Running`.
- `Stopped` is terminal.

The compiled Runtime is constructed only by its Session friend and owns one
private `shutdown(intent)` transition function; it does not expose internal
`drain` or `close` mirrors. Public `Session::drain` selects the admission intent
and public `Session::close` selects the terminal intent. `rund::run` does not
reproduce the state branch. The state mutex
serializes each transition but is released by the scope condition-variable
wait and before Compute cancellation, completion, telemetry, retirement, or
Scheduler reset. Concurrent closers join the same Draining transition; exactly
one caller records `Stopped`, and every caller returns the same terminal state.
`open` reserves trace storage, so both lifecycle records append or
overwrite without allocation. Ordinary callers perform one public `close()`
operation; this is an operation and allocation bound, not a wall-time speedup
claim.

`Session::Snapshot` reports only actual Session state:

- `state`
- `active_compute_jobs`
- `scope_active`
- typed diagnostic `code`, with `ok()`, truth conversion, `error()`, and
  `exit_code()` derived from that one value

Snapshot stores no raw reason pointer. A successful observation has `Ok` and
an empty error; same-Session reentry reports `RuntimeReentryForbidden`. The
lifecycle state remains an observed state and is not inferred from the
operation code.

Manual admission, lease, running-run, pool, or reserved-memory counters do not
belong to Session and must not reappear in its snapshot or replay schema.

## Scope And Run

`Session::scope(callback)` validates the lifecycle/backend, installs the
Session's kernel parallel provider and Scheduler, invokes a borrowed
no-argument callback, drains admitted task work, captures evidence, and restores
the saved thread-local providers. One Session admits one active scope. A nested
or concurrent scope fails closed with `runtime_scope_busy`; callback exceptions
become `runtime_scope_callback_failed` after task drain.

Successful scope admission retains the admitted ComputeHost as a lifetime
capability. Provider installation and task-scope construction consume that
capability directly; they do not re-read lifecycle, backend, host identity, or
Scheduler readiness after admission. Plan installation and Scheduler
`BeginScope` remain the mutable-state admission authorities.

The public template owns only callback-shape checking and a stack-borrowed
`void*`/function-pointer trampoline. The compiled `run_scope` owner contains the
state machine, exception boundary, drain, and restoration. For `C` callback
types, code generation is one `Theta(S)` lifecycle body plus
`Theta(C*T)` small trampolines.

After drain, one Scheduler evidence snapshot owns stats, prepared memory,
scope-local observations, host events, payload storage, and canonical-input
row and byte counts. It acquires the evidence mutex once and projects every
field from that same state, rather than taking five independently locked
snapshots that could observe different boundaries. Reusable public
`Session::scope` copies observation and host-event evidence so the Scheduler
retains its configure-time reserved capacity; moving that storage away would
make a later `noexcept` record path allocate and possibly terminate. The
terminal one-shot `rund::run` path has a source-private take operation instead:
because no later scope can produce into that Scheduler, its two vector buffers
transfer in `O(1)` with no element copy or second retained allocation. Host
payload archives keep their distinct Store-to-public projection because their
internal and public shapes differ. The 5-to-1 mutex reduction is a structural
operation count, not a wall-time claim.

`rund::run(config, callback)` is the one-shot facade:

```text
configure -> start -> scope -> close -> capture trace
```

The one `close` call records the internal Draining and Stopped transitions. If
scope completion failed, that first exact failure remains the
result authority; otherwise a close failure becomes the result code.

It accepts a callback taking the exact active `Session&` or no arguments. The
callback is borrowed, never copied or retained. Task statistics, prepared
memory, observations, host events, payload archive, and trace are moved into the
single `Session::Result` owner. Run does not retain lifecycle-result mirrors.

For `N` scopes, let `L` be fixed open/drain/close work and `W_i` the work of
scope `i`. A loop of one-shot runs performs `N*L + sum(W_i)`, while one reusable
Session performs `L + sum(W_i)`. Reuse therefore removes exactly `(N-1)*L`
fixed lifecycle work from the model. This is an ownership and operation-count
bound, not a numeric speedup claim; wall-time claims require the installed
scheduler measurement route.

## Trace

The current trace vocabulary is exactly:

- `RuntimeConfigured`, `RuntimeStarted`, `RuntimeDraining`, `RuntimeStopped`
- `ComputeSubmitted`, `ComputeAdmitted`, `ComputeDispatchStarted`
- `ComputeBackendSubmitted`, `ComputeCompleted`
- `TelemetryEmitted`, `TelemetrySkipped`

Trace stores typed identity, never borrowed diagnostic pointers. The root
`Trace::code` is the `ReasonCode` for the trace observation itself. Every
record owns one four-byte `TraceCode`: its domain is `Runtime` or `Compute`
and its 16-bit value is the exact `ReasonCode` or `compute::Reason` from that
domain. `runtime_code()` and `compute_reason()` expose the selected typed
value, while `error()` is only its stable text projection. Invalid raw
domain/value construction is not public. The nested state snapshot owns one
`ReasonCode code`, matching `Session::Snapshot`.

The Runtime trace text projection is generated directly from the canonical
`rund/compute/reason.def` schema. It does not link through `compute::Status`:
the Runtime-base closure can therefore report an exact Compute reason without
pulling the higher CPU-Compute component into every Runtime-only edit and
link. `reason.def` remains the one text-and-value authority; Trace and Status
are two boundary projections of that schema, not independent reason tables.

Lifecycle and successful event labels are already identified by `TraceEvent`,
so their code is `Runtime/Ok`; successful Compute events use `Compute/Ok`.
Failed Compute completion retains its exact `compute::Reason` rather than
collapsing into a generic Runtime failure. A rejected telemetry callback emits
`TelemetrySkipped` with `Runtime/TelemetrySinkFailed`. Thus the event names,
failure identity, and observed state each have one non-overlapping authority.

Trace retention is a bounded circular buffer. Insert after saturation is
`O(1)` and increments `dropped`. Public `trace()` preserves the Session and
returns an ordered copy. Terminal one-shot Run capture instead rotates a
wrapped ring into canonical sequence order and moves its vector buffer.
Wrapped rotation costs `O(N)` and terminal capture allocates no second buffer
and copies no trace element. Replay hashes are computed once from the
result-owned canonical trace.

## Compute Node Host

The only Node-native Compute entry is
`Session::compute(rund::compute::Job<R>&).submit()`. Session accepts one mutable,
compiled, resident Job; it does not accept a Flow, Program, graph, temporary
Job, buffer list, or per-submit backend override.

The compiled bridge has one physical authority per responsibility:

- `entry.cpp` owns Session-bound Device binding and the public
  Session-to-Runtime handoff;
- `operation.cpp` owns the immutable Job/Pipeline operation tables and their
  common completion publication;
- `execution.cpp` owns backend coordination, slot claim, join, release,
  host-wide retirement, and cancellation;
- `request.cpp` owns Request admission and Submission observation/control;
- `terminal.cpp` owns terminalization.

`local.hpp` declares only the source-private links between those compiled
owners. `node/cmake/node/sources/runtime/compute.cmake` is their sole CMake
source list; there is no mirrored expected-source inventory. Repository-wide
source-ownership closure remains the independent omission/duplication guard.
Operation table, state transition, publication order, and public type are
independent of the physical source partition.

The admission vocabulary is wholly owned by `rund::compute`:

```text
Request -> Submission -> Poll | Completion
   |                         ^
   +---- direct co_await ----+
```

`Session::compute(job)` returns the move-only `compute::Request`.
`Request::submit()` returns the move-only `compute::Submission`, whose
`poll()`, `wait()`, and `cancel()` observe or control that one admission.
Direct coroutine await returns the same `compute::Completion` publication.
The coroutine bridge is the nested `compute::Request::Awaiter`; it does not
create another root product type or result family.

Submission uses the Session Scheduler and one bounded Compute task slot.
`poll()` observes admission, backend submission, and terminal publication
without changing them. `wait()` and coroutine await join the same completion;
`Job::read()` remains the explicit readback boundary. CPU work uses the
configured Kernel backend. Metal and Vulkan use their prepared
asynchronous completion services. There is no backend fallback or hidden pool.
Task and Compute work that must share one deterministic drain boundary is
submitted inside the same `Session::scope`. A Compute submission started
outside a scope must reach terminal publication before a new scope is admitted;
otherwise `Session::scope` returns `RuntimeScopeBusy` instead of installing a
second Scheduler owner around in-flight work.
The poll and final result each retain only `compute::Reason`; category code and
stable diagnostic text are projections of that typed value. Pending admitted
work has no synthetic failure: it reports `Reason::Ok` until terminal
publication, while an immediate admission rejection is already terminal.

Cancellation and completion publish through one atomic job gate. Active job
count is incremented before accepted execution becomes observable and is
released exactly once on terminal publication. Session close reads that owner
through `Session::Snapshot::active_compute_jobs`.

Compute lifecycle owns one blocking terminal operation used by Session
`close()` and the Runtime destructor. It closes admission, requests
cancellation when the authoritative active count is non-zero, waits on the
host completion condition, then retires tasks, drains the compilation service,
resets the Scheduler, and detaches callbacks exactly once. Admission continues
to report `RuntimeDraining` until terminal teardown changes the host to not
running. An already terminal host is an idempotent no-op. Runtime lifecycle
code does not mirror the host's `closed` flag, cancellation loop, retirement
loop, Scheduler reset, or callback detachment.

Terminal Compute status and statistics cross the source-private synchronous
telemetry projection by constant reference. The completion owner writes the
durable `TaskState` result once; there is no intermediate by-value callback
adapter or second statistics snapshot. The callback cannot retain either
reference beyond the call. On the checked arm64/libc++ ABI, `compute::Stats`
is 280 bytes and `compute::Status` is 24 bytes. The reference boundary carries
two addresses and performs zero logical `Stats` or `Status` payload copies per
completion. Event shape, callback ordering, and Basic/Detail selection are
owned by [Telemetry](./telemetry.md).

Compute task and CPU worker-batch reuse share one `SlotSet` implementation.
The Compute-host mutex serializes object-vector growth; one atomic bitmap is
the sole availability authority, so release does not acquire that global
lock. Claim clears the lowest free bit, giving deterministic lowest-index reuse
without a second per-object `used` flag. For `A` allocated
slots, claim inspects at most `ceil(A / 64)` contiguous 64-bit words and issues
one successful word compare-exchange rather than chasing and issuing a
compare-exchange against up to `A` independently allocated objects. At the
default capacity 1,024, each set occupies 128 bytes and the worst claim probe
bound is 16 words. Growth creates the exact next index; the bitmap is not a
parallel object-count authority.

## Prepared Memory And Scheduler Evidence

Session retains one read-only prepared-memory observation in
`Session::Result::memory()`. The
public value is `rund::PreparedMemory`; its capacity proof is the nested
`PreparedMemory::Capacity`, not a second top-level runtime type.
Capacity stores one `ReasonCode code`; `ok()`, truth conversion, `error()`, and
`exit_code()` derive from that code, while `valid()` admits only `Ok` or the
prepared-memory reason category owned by the reason schema. It does not retain
a parallel boolean or diagnostic pointer. `task::Stats` does not mirror any
capacity, high-water, overflow, epoch, or status field; the complete value
is the one retained authority, so two public observations cannot disagree.
`rund::record_memory(memory)` is the sole writer. Recording is a
root/sequencer operation; task-worker attempts and codes outside the
prepared-memory category fail closed and cannot mutate the snapshot. The
public header does not accept Kernel telemetry types or own a second
Kernel-to-Node projection helper.

Scheduler capacities in `SessionConfig::scheduler` are the same
`SchedulerConfig` value consumed by the Scheduler. Session resolves only
documented zero sentinels such as worker and observation capacity. It does not
flatten or mirror those fields.

## Discovery And Parallel Provider

Session configuration is the sole resource-discovery entry. Its private
`resource::Resolve` transaction returns one typed code, backend kind, backend
lifetime owner, and one `ResourceEnvelope`; Session never parses a diagnostic
string back into a code. Verified capacity requires one non-zero milli-capacity
entry per requested worker. Unknown topology remains unknown; Session does not
manufacture evidence or expose a standalone discovery authority.

An active Session scope installs a scoped Kernel parallel provider. Kernel
`par()` outside that scope fails with `parallel_runtime_missing`; an explicit
width that differs from the configured width fails with
`parallel_runtime_width_mismatch`. A Scheduler worker cannot recursively enter
the same pool and fails with `pool_nested_dispatch`.

Provider acquisition inside the admitted user callback is normal use of that
scope capability, not Session reentry. It validates the current lifecycle,
configured width, and backend under the Session state mutex. The per-thread
active-session chain guards lifecycle, scope admission, snapshots, resources,
trace, and telemetry control entry; it does not duplicate the provider's own
admission authority.

## Concurrency And Reentry

Session state has one mutex authority. User callbacks do not run while that
mutex is held. Session callbacks are guarded by a per-thread active-session
chain and its one owned Scheduler identity. A direct scope callback matches the
chain; every leaf and coroutine body running for that Session matches
`ActiveScheduler()` against the Scheduler retained by the Session's Compute
host. Reentering lifecycle, scope admission, snapshots, resources, or trace
through either path fails immediately with `runtime_reentry_forbidden` instead
of allowing a task to wait for the scope that is waiting for that same task.
The identity comparison is Session-specific, so a task cannot falsely reject
an independent Session. Consequently, a nested `rund::run(...)` that creates a
different Session is admitted; only reentry through the same Session is
rejected. It does not guard `AcquireParallelRuntime` or the
Compute-job submission path; those retain their existing provider and
coordinator admission authorities. The reentry failure observes its paired
lifecycle state under the state mutex, so a concurrent close and a reentry have
one valid linearization and no unsynchronized lifecycle read. Close waits for
`!scope_active` on the Session-owned condition variable, which releases the
state mutex while sleeping, and invokes blocking Compute teardown only after
releasing that mutex. Compute host locking follows Session-state then
Compute-host order during admission; terminal callbacks publish outside the
host lock before taking Session state.

## Replay

Replay's public and persistence authority is the
[Replay contract](./replay.md). Session owns only the lifecycle and exclusive
scope admission used by that contract.

`replay::live`, `replay::record`, `replay::run`, and `replay::scenario`
require one open, Running `Session`. They reuse that Session's workers,
scheduler, prepared storage, Compute host, backend, and bounded evidence
owners. All modes invoke the same application callback shape through
`replay::Context`; applications select a verb and do not branch their
simulation graph by mode.

Session configuration owns bounded replay storage and diagnostic policy. The
Replay contract owns canonical Input/Writer capture, producer-call
counts, strict substitution, choice admission, immutable Record evidence,
comparison, checkpoint persistence, continuation, and retention. No replay
surface may add domain meaning, a second lifecycle, a per-verb Session
configuration, or a caller-authored mode branch.

## Stable Runtime Reasons

The Runtime owner emits the following lifecycle/scope failure codes:

- `already_configured`, `not_configured`, `not_runnable`
- `runtime_id_required`, `backend_invalid`, `backend_width_required`
- `runtime_resources_invalid`
- `host_replay_storage_invalid`, `task_scheduler_allocation_failed`
- `telemetry_level_invalid`
- `verified_topology_required`, `affinity_truth_unavailable`
- `worker_capacity_truth_unavailable`, `backend_invalid`
- `runtime_reentry_forbidden`, `runtime_scope_not_started`
- `runtime_scope_busy`, `runtime_scope_callback_failed`
- `parallel_runtime_missing`, `parallel_runtime_width_mismatch`
- `telemetry_sink_failed`

Successful lifecycle and scope results carry only `ReasonCode::Ok`; configured,
started, draining, and stopped are lifecycle/trace state labels, not parallel
success reasons.

Telemetry level selection and event behavior are owned by
[Telemetry](./telemetry.md). Scheduler, task, host, network, and Compute reasons
are owned by their nearest contract pages. Adding or changing a Runtime reason
requires source, docs, and the nearest focused contract in one change.

## Verification

Use the narrowest current owners first:

- `runtime.lifecycle`: result UX, lifecycle, actual snapshot, circular trace
- `runtime.kernel`: resource discovery, scoped Kernel provider, and concurrent
  independent one-shot callbacks
- `runtime.stress`: deterministic task evidence and lifecycle cleanup
- `runtime.memory`: prepared-memory recording and result observation
- `runtime.task.basic`: scope/task allocation and configuration-failure cleanup
- runtime Compute and replay owners for asynchronous and codec behavior
- installed SDK consumers for direct-header and package closure
