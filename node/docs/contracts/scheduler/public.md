# Scheduler API Surface

This page owns the scheduler SDK projection, source-internal header boundary,
and public result shapes. SDK consumers reach selected scheduler names through
`<rund/task.hpp>` or the `<rund/rund.hpp>` composition; private state, source
files, and verification route through the sibling owner pages.

## Scope

The scheduler owns `task::spawn`, `task::join`, `task::scope`,
`task::yield`, runtime-aware `task::sleep`, runtime-aware `task::channel<T>`,
typed runtime-aware `rund::host::io::readable` and
`rund::host::io::writable`, task-safe `rund::host::io::read_some` and
`rund::host::io::write_some`, the
scheduler-owned fd readiness reactor used by both low-level IO waits and
`rund::net` readiness, task-worker lanes, record lifetime, scheduler
evidence, and blocking park/resume state.

`rund::task` is the sole public namespace for task values and verbs.
Installed task support types that must cross template and compiled boundaries
have one non-product authority, `rund::detail::task`. The compiled Scheduler
class remains source-private under `rund::node`. Product-wide reason values are
owned directly by `rund::ReasonCode`.

NodeHost accelerator completion uses the Scheduler's internal external-ticket
primitive. The admitted record transitions `Pending -> Parking -> Parked`, with
the in-flight count reserved before `Parked` is release-published. Backend
completion transitions the ticket to `Complete` and links the same record into
its home lane's intrusive wake queue. The lane records one external park and
one external wake. The record's existing wake ticket is zero while parked and
becomes the enqueue ticket exactly once, so no duplicate-wake bit, completion
task, future, or per-wake heap node exists.

The scheduler owns only task-safe admission, dedicated host-IO execution,
replay substitution, and evidence ordering for host IO read/write. It does not own file formats,
worker protocols, gameplay commands, buffering policy, retry, or hidden queues.
Replay storage is scheduler-owned byte evidence only: record mode appends
admitted `IoRead` and `IoWrite` bytes into the configured archive store, and
replay mode resolves the matching archive record by event sequence and kind.
Replay `read_some` never repairs a missing payload by reading the native fd;
missing, corrupt, or mismatched archive storage fails closed before gameplay
logic can consume substituted bytes.
Configuration transfers expected events and expected payload ownership once
into this Scheduler. Runtime lifecycle state does not mirror either value.
Replay execution reads the Scheduler-owned evidence for the whole session and
materializes public report evidence only at the observation boundary.

The scheduler does not own user callback arithmetic or side effects, raw
pointer races or unsynchronized shared memory, kernel element traversal,
partitioning, fold order, workspace layout, network/file wrapper semantics,
`select`, work stealing, priority scheduling, or preemptive task migration.

## SDK Projection And Source Owners

The focused installed consumer entry is `<rund/task.hpp>`; the convenience
composition is `<rund/rund.hpp>`. Both link through `runD::sdk`.
The following paths own implementation and transitive support; none is an
additional SDK direct-include entry:

- `/node/include/rund/task/`
- `/node/include/node/runtime/runtime.hpp`
- `/node/include/rund/host/io.hpp`
- `/node/include/rund/session/run.hpp`

Implementation entrypoints that back the public surface:

- `/node/src/runtime/task/api/bridge.cpp`
- `/node/src/runtime/task/await/bridge.cpp`
- `/node/src/runtime/task/io/bridge.cpp`
- `/node/src/runtime/task/memory/prepared.cpp`
- `/node/src/runtime/task/scope/frame.cpp`
- `/node/src/runtime/task/scheduler/`
- `/node/src/runtime/runtime.cpp`

`/node/include/rund/task.hpp` is the sole task aggregation owner. Its
transitive support headers split the scheduler surface by semantic owner. The
top-level `task/*.hpp` support headers may compile independently for dependency
hygiene, but they are not SDK direct entries and carry no path-stability
promise. The deeper
`task/channel/*.hpp` files are class-body fragments owned only by
`task/channel.hpp`; they are not standalone include targets. Semantic
contracts verify the subsystem owners; the installed-package consumer compiles
`<rund/task.hpp>` independently and verifies the same names through the
umbrella composition.

Repository production and semantic-contract translation units consume those
top-level support owners directly and never consume `<rund/task.hpp>` or
`<rund/rund.hpp>`. This keeps an edit to channel, cancellation, group, or
telemetry support out of contracts that use only task admission or awaiting.
It does not promote support paths to SDK entries: package consumers retain the
sole aggregate-surface proof.

Internal scheduler and net helpers such as reactor wait bridges, cleanup
owners, socket-registry accessors, scheduler host-event record methods, and
host API event-record bridges must not appear in `/node/include/node` unless
the owning contract promotes them to the documented subsystem API.

## Public Handle Boundary

`rund::task::Handle` is the public task identity value. It exposes invalid
construction, boolean validity, id, reason, and reason code. Scheduler,
channel runtime, coroutine completion, join/spawn/scope admission, stop-token
identity extraction, and host/network active-scheduler lookup are not public
`Handle` methods.

Template-required bridges live under the owning `task/api/`, `task/await/`,
`task/channel/`, and `task/cancel/` domains as `access.hpp`; installed
direct-header and compile-fail contracts validate their public boundary.
Source-only scheduler lookup lives under
`node/src/runtime/task/scheduler/access.hpp` and is not included from public
headers.

Host APIs that only need active-task identity, logical time, the active random
seed, or host-event recording use the source-only
`node/src/runtime/task/scheduler/host.hpp` bridge. That header forward-declares
its value types and does not include Scheduler state. Only readiness, lifecycle,
and other source owners that invoke Scheduler methods include
`scheduler/state.hpp` directly; common network helpers never transmit that
private state dependency to unrelated socket translation units.

## Source Header Map

- `rund/reason.hpp`: product-wide reason ABI and its text projection.
- `task/operation/kind.hpp`: the non-product scheduler operation tag consumed
  by template-required channel bridges and compiled scheduler owners. No
  operation record or tag is declared in `rund::task`.
- `task/observation.hpp`: scheduler observation ABI.
- `task/callable.hpp`: `detail::task::Callable`, the bounded inline
  leaf-callable owner; oversized or
  over-aligned callables are rejected at compile time and never use the
  process heap.
- `task/cancel.hpp`: scheduler-owned stop source and stop token cancellation
  surface.
- `task/cancel/identity.hpp`: the sole storage-shape owner for the source-only
  stop identity carried by public token and network-awaitable internals. It
  defines one scheduler-qualified identity and its complete scheduler-local
  source projection; consumers do not repeat their component fields.
- `task/stats.hpp`: read-only scheduler telemetry snapshot and direct getter
  surface.
- `task/stats/storage.hpp`: `detail::task::StatStorage`, the compact 232-slot live telemetry
  block. It owns no public field spelling and performs no allocation.
- `task/stats/schema/slots.def`: the sole exact numeric slot identity and
  compact storage-layout schema. Deleting a counter removes its row and
  compacts every following internal index. `storage.hpp` derives its count
  from these rows; `slots.hpp` validates their contiguous indices and owns the
  internal counter access primitive.
- `task/stats/schema/public/*.def`: the sole category-specific one-to-one
  mapping from short public getter spelling to its internal slot. No storage
  index or replay order is repeated in these mappings.
- `task/stats/{reactor,network,resource}.hpp`: small owning category snapshots
  returned by `reactor()`, `network()`, and `resources()`; they never retain a
  pointer into a temporary full snapshot.
- `node/runtime/replay/task/stats.hpp` and
  `node/runtime/replay/task/schema/semantic.def`: source-private replay wire
  slot set and semantic-hash order. Both reference current scheduler slots by
  name; neither is an installed task header or a second numeric slot authority.
- `node/src/runtime/task/stats/access.hpp`: source-private snapshot and
  slot-value access authority. Installed headers expose only an incomplete
  friend declaration and no mutable statistics accessor.
- `<rund/session/scheduler.hpp>`: the transitive owner of the public
  `rund::SchedulerConfig` schema reached through `<rund/session.hpp>` and
  consumed directly by Runtime and Scheduler.
- `task/active.hpp`: `detail::task::ActiveState`, the active scheduler/task
  identity snapshot.
- `task/handle.hpp`: task handle identity value and private construction
  friendship only; public callers cannot name scheduler, join/spawn/scope,
  coroutine completion, stop-token identity, or active-scheduler bridges.
- `task/handle/typed.hpp`: `detail::task::ResultHandle<T>`, the move-only
  typed completion observer, and its
  generation-checked result access.
- `task/api/access.hpp`: `detail::task::ApiAccess`, the template-required bridge from public `spawn`,
  `join`, `join_all`, and `scope` APIs into scheduler admission
  and join internals.
- `task/await/access.hpp`: `detail::task::AwaitAccess`, the template-required bridge from coroutine
  awaiters into scheduler-owned suspended-result completion.
- `task/channel/access.hpp`: `detail::task::ChannelAccess`, the template-required bridge from
  `channel<T>` class-body fragments into scheduler-owned channel runtime
  authority.
- `node/src/runtime/task/scheduler/cancel/access.hpp`: the source-only
  `detail::task::StopAccess` bridge that extracts one complete stop-identity
  value
  for source-owned cancellation integrations; callable source wrappers for
  timed-ready completion and stop-token identity are declared only under the
  source tree, not from public bridge headers.
- `task/status.hpp`: the sole payload-free scheduler outcome value.
- `task/results.hpp`: yield/sleep operations, IO operation/result, and typed
  receive result. Join, scope, channel send/close, and stop requests return
  `Status` directly.
- `task/channel.hpp`: `channel<T>` class aggregation.
- `task/channel/`: channel lifecycle, send/recv, close, waiter queues, ring
  buffer, rendezvous slots, completion slots, cleanup, operation gates, and
  storage fragments.
- `task/api.hpp`: public `spawn`, `join`, `scope`, `yield`, and `sleep`
  entrypoint templates/declarations.
- `host/io.hpp`: the sole public `io::Fd` owner, `io::FdView` borrow,
  readiness, task-safe transfer, and blocking host-IO declaration authority.
  It is reached through `<rund/host.hpp>` and is not aggregated by
  `<rund/task.hpp>`.

## Native Task Bridges

`task/api/bridge.cpp`, `task/await/bridge.cpp`, `task/io/bridge.cpp`,
`task/memory/prepared.cpp`, and `task/scope/frame.cpp` are the source-private
bridges from public task objects to
the scheduler. Native coroutine frames are owned by the scheduler frame arena;
there is no second frame ABI.

`task::Task<T>` is the move-only RAII owner of a native coroutine frame only
before scheduler admission. Its public surface is the coroutine return value,
move construction/assignment, validity, completion, and failure observation.
It exposes no raw handle type, raw-handle constructor, frame observer,
`release`, or `destroy` operation. The language-required `promise_type` names
one `detail::task` implementation type; promise storage, final suspension, and
frame allocation are not product vocabulary.

For every frame `f`, ownership is linear:

```text
O(f) in { Task, Scheduler, None }
create:          None -> Task
admit exactly once: Task -> Scheduler
retire/reject:   Scheduler -> None
```

`detail::task::TakeCoroutine` is the sole `Task -> Scheduler` transfer. A
failed admission or missing runtime destroys that transferred frame through
the same type-erased operations used by normal retirement. Destruction clears
the internal start capability before releasing the frame, so repeating the
internal cleanup is a no-op rather than a second destroy. User code can
therefore neither fork frame ownership nor bypass scheduler retirement.

The configured coroutine-frame byte limit and task capacity remain owned by
the Scheduler's `rund::SchedulerConfig`. The frame arena consumes those frozen limits but does not
mirror them in mutable telemetry state. Public resource reports materialize
the configured maximums from that owner and the live/high-water/allocation
counters from the arena. Internal admission code that needs only the byte
limit uses the source-private constant-time scalar observer; it does not call
the public statistics snapshot boundary.

Frame and typed-completion storage consume one private alignment predicate:
`a != 0 && (a & (a - 1)) == 0`. Frame stride rounding additionally proves
`value <= SIZE_MAX - (a - 1)` before applying the power-of-two mask. The frame
arena and completion pool do not keep local copies of those bit-level rules;
the shared functions are inline, stateless, and add no allocation or pass.

## Public Result Shapes

`Status` is the sole payload-free scheduler outcome. It stores exactly one
`ReasonCode`; its public default is fail-closed `TaskInvalid`. `operator bool`,
`ok()`, `code()`, `error()`, and `exit_code()` are projections of that one
code. `exit_code()` is zero only for `Ok` and one for every failure, so a
command-line task boundary does not invent a second error number. Success is
created by `success()` and failure by `fail(code)`; neither a bool nor a reason
pointer is stored.

`join`, `join_all`, `scope`, channel send completion, channel `close`, and
`stop_source::request_stop` return `Status` directly.

Calling synchronous `join`, `join_all`, or `scope` from an active task crosses
one private rejection transition. That owner derives the coroutine-versus-leaf
reason, marks a leaf failed, and completes primitive accounting exactly once;
the three public operations do not mirror that mutable transition.

`YieldOp` and `SleepOp` are deferred requests, not result objects. They contain
one private `Status`; `SleepOp` additionally owns its requested duration.
Their synchronous observers preserve root/leaf failure inspection, while
`co_await` returns `Status`. Evaluating and discarding either operation
does not call the suspension bridge.

`IoOp` owns the admitted fd identity and interest needed only until
`await_suspend`; `co_await` returns `IoResult`, which privately composes one
`Status` and the readiness `revents` payload. `ReceiveResult<T>` likewise
composes one `Status` and one optional payload. Neither public result stores
coroutine suspend/deferred flags, task ids, fd identities, generations, or
durations. Those state transitions are private decisions owned by the awaiter,
reactor, or channel operation.

Host transfer and file results share one inherited `io::Status`. Its
`ReasonCode` is private and observed through `code()`, `ok()`, truth
conversion, `error()`, and `exit_code()`. A caller cannot mutate that code or
construct a `ReadResult`, `WriteResult`, `OpenResult`, or `CloseResult` from a
reason. Source-private construction normalizes invalid success inputs to
`TaskInvalid`; operation payload fields do not become a second completion
authority. These values stay fixed-size and trivially copyable and add no
allocation, ownership transfer, or status indirection to the host-IO
completion path.

Task-safe host transfer has one source-private operation authority. Configure
prepares the bounded slot array; live admission projects fd identity once,
claims and queues through intrusive `O(1)` owners, and the FIFO worker performs
one native byte operation. Equal-capacity scope reset retains that storage and
is `O(1)` in `host_io_capacity`. The first live use may pay platform thread
startup, while the focused host-IO contract proves zero process-heap
allocations for the complete warm would-block operation. Exact ordering,
generation reuse, replay substitution, and cancellation semantics are owned by
the Host contract rather than repeated here.

`Handle` stores scheduler id, scope id, task id, and one `ReasonCode` only.
Validity is `task_id != 0 && code == Ok`; `error()` derives text from the code.
It stores no bool or reason pointer and has no user destructor, so the identity
value remains trivially copyable and trivially destructible.

`stop_token::state()` returns `StopState`. Its truth value reports whether the
query itself succeeded, and `requested()` reports the independent cancellation
state. Operation validity and cancellation state therefore remain independent
observations.

Public telemetry uses short event names whose exact meaning lives here:
`effect_events_merged` counts logical effect events merged into canonical
commit order, and `terminal_yield_rejections` counts yield effects rejected by
the terminal-continuation proof. The names do not repeat the surrounding
`task::Stats` and deterministic-commit context.

Operation results never embed a `task::Stats` snapshot. Awaiters retain
their result across suspension inside the coroutine frame, so embedding the
full counter set would multiply telemetry storage by every live task and copy
it at each primitive boundary. The move-only `Session::Result` is the sole
public scheduler telemetry snapshot. Its `code()`, `ok()`, truth conversion,
`error()`, and `exit_code()` contract is owned by
[`../runtime.md`](../runtime.md); this page owns only the rule that `tasks()`
returns the single scheduler telemetry snapshot at the run boundary. This
applies equally to hot, suspending, and failure results.

`task::Stats` is a 1,856-byte, 8-byte-aligned, trivially copyable report
value containing 232 ordered 64-bit counters.
The scheduler mutates one inline source storage with a compile-time slot load,
add, and store. It materializes the public value once at the report boundary;
host-event recording returns only success/failure and never returns or copies
the full snapshot. Public counters are read through methods such as
`tasks.spawned()`, `tasks.reactor().ready_events()`, and
`tasks.resources().max_tasks()`. Prepared-memory state is deliberately absent:
the complete `rund::PreparedMemory` returned by `Session::Result::memory()`
is its sole report authority.

`Result<T>` owns exactly one active alternative: a value or a failure
status. One discriminant is sufficient to answer both value and
outcome queries. The concrete Task and Compute result families share one
generic storage and observer owner; neither family is an alias or a parallel
implementation. Their typed Status policies select the domain failure without
changing the storage law. Move exception specifications derive from the
complete value-or-failure storage operation; a nothrow value assignment alone
cannot incorrectly mark a throwing cross-alternative construction as
`noexcept`. If moving a user value throws and invalidates the active
alternative, the checked outcome surface maps that state to `TaskFailed`
instead of throwing from a `noexcept` observer. The result provides
checked-path `operator->`, dereference, `ok()`, `code()`, `error()`, and
`exit_code()` access.
`Result<void>` carries the same status contract without manufacturing a
dummy value. These types are used at terminal and resource boundaries; they do
not add `.value()` handling to lazy Compute graph construction.

The scheduler configures one `CompletionPool` with a `task_capacity` admission
bound and a fixed result slot size/alignment. Cells are created in retained
256-slot pages only when a typed observer claims them. A plain `Task<void>`
joined through `Handle` uses the task record terminal state and claims no cell.
A fixed set of pool-owned synchronization stripes protects materialized cells;
mutexes and condition variables are not repeated per task. Typed result storage
is lazy and cell-owned. Cells enforce the explicit
idle→admitted→ready→running→parked/committing→terminal state graph, retain a
generation across reuse, and share one terminal value/status with polling and
external waiting. Invalid transitions, stale generations, oversized values,
and type mismatches use stable Task reason codes. Pool retirement defers its
configure-time storage release until every live completion lease returns; warm
claim, transition, terminalize, publish, poll, wait-ready, and release allocate
no heap. A cell generation increases monotonically without wrap: recycling a
cell at the maximum generation burns that cell instead of re-admitting an
identity that an old handle could equal. Thus for any two admissions of one
recyclable cell, `g(next) > g(previous)`; finite-width overflow cannot turn a
stale handle into a live handle.
The coroutine's typed result observer is move-only and observes the same
generation-checked cell. It is private task support, not a second public handle
surface.
`poll()` is read-only, `wait()` uses its cell's synchronization stripe, and
`result()` performs type-checked copy-out. A scheduler task cannot call the
blocking wait path: it receives `task_worker_wait_forbidden`. Producer release
does not recycle a terminal cell until the last observer is destroyed, and a
stale generation reports `task_handle_stale`.
Coroutine waits use caller-owned intrusive waiter nodes. Registration inserts
at the list head in `O(1)` and cancellation removes its exact doubly-linked
node in `O(1)`; neither operation scans other waiters. Terminal publication
performs the necessary `O(W)` work in two bounded passes: it reverses the LIFO
registration list into canonical FIFO arrival order and marks every detached
node as waking while holding the cell lock, then invokes each callback exactly
once outside that lock. The predecessor link costs one machine word per
caller-owned waiter, which is the local metadata needed to remove an arbitrary
node without either an `O(W)` predecessor search or an external index. A
detached node cannot be cancelled or re-linked until its callback ownership is
released. Completion waiting therefore adds no warm-path heap allocation and
does not create an executor or worker thread.

Normal coroutine completion stages one outcome in `Committing`, then the
scheduler's canonical commit owner publishes it. Rejection and retirement use
one direct terminalization operation from any live nonterminal phase; they do
not replay the public state graph through repeated poll/transition locks.
Terminalization holds one cell stripe once, and the first successful terminal
state is immutable. One phase field and one terminal value preserve
deterministic first-terminal ownership without result-presence mirrors.

Nested `Task<T>` is admitted as a scheduler task, including non-void results.
The record owns a type-erased native coroutine frame plus result move/destroy
operations; terminal commit moves the value into its bounded completion cell
before join waiters resume. A nested task that suspends is never resumed inline
by its parent. Once `await_resume` copies the typed result, the terminal child record
is retired and its scheduler-owned frame slot is reusable; sequential nested
awaits are bounded by peak concurrency rather than lifetime submission count.
Native task promises retain only a stable terminal `ReasonCode`; they do not
own `std::exception_ptr`. An unexpected C++ exception is caught once by the
promise boundary and converted to `task_failed`, while typed completion publish
commits the stored reason without rethrowing it in the scheduler.

Coroutine `yield()` evaluation is side-effect free. It produces `YieldOp`, and
its `await_suspend` performs the scheduler enqueue and
park atomically. Discarding the token does not increment yield telemetry or
change the task state.
Coroutine `sleep(duration)` returns `SleepOp` and follows the same rule:
construction records only the requested duration; validation, deadline
creation, timer admission, and parking occur in `await_suspend`. A discarded sleep operation
does not consume timer capacity or emit timer evidence.
Coroutine join uses `co_await handle`. `HandleAwaiter::await_suspend` alone
registers the join and parks the current task; constructing the awaiter is side-effect
free. The boundary borrows either an lvalue or temporary long enough to copy
the handle once into coroutine-owned awaiter storage; it does not copy through
an intermediate by-value parameter. This is the only coroutine join suspension
expression.
Coroutine I/O readiness calls create an `IoOp` only. Its `await_suspend`
performs validation, reactor admission, and park; discarding a readiness
operation consumes no reactor slot and emits no wait or park
evidence.

`spawn(lambda)` is a non-suspending leaf. The scheduler invokes it once on a
worker without allocating a suspension frame. Suspending work must use
`Task<T>`. A leaf primitive attempt fails with
`task_leaf_primitive_forbidden`; no alternate suspension engine exists. Leaf
eligibility is decided before either timed or untimed readiness performs an
immediate native probe, so descriptor readiness cannot change that result or
publish IO observations for the rejected primitive.
Synchronous `task::scope` is a scheduler-control operation, not a task
primitive. Calling it from a leaf or coroutine fails before its callback runs;
task-owned structured concurrency uses coroutine await or `Group` instead.
The named overload requires a non-null, non-empty, NUL-terminated task name.
Admission hashes the name once into the canonical spawn-order evidence; it does
not copy the string or retain it in the 96-byte `TaskRecord`. Equal names under
an equal schedule therefore produce equal scheduler trace evidence, while a
name change changes that evidence. The unnamed overload supplies the canonical
name `task`.
After the scheduler has materialized one leaf record and callable slot, a warm
leaf `spawn` through ready admission, worker execution, terminal publication,
and root `join` performs zero process-heap allocations. The
`runtime.task.basic` contract instruments global allocation across that entire
interval on a reused Runtime; narrower frame and completion tests do not stand
in for this end-to-end contract. Its terminal single-leaf row also owns the
exact lifecycle evidence for one inline callable admission, one terminal
callable reset, and one root single-join ready fast path; there is no separate
callable contract or implementation-shape move-count authority. The same
contract primes a four-lane segment and then proves that a complete second
multi-lane spawn-to-join batch performs zero process-heap allocations. Segment
job and effect capacity is retained by the Scheduler rather than rebuilt per
worker or per batch.

`task::Stats` exposes nested snapshots for scheduler reactor, admitted
network host-event activity, and scheduler resources. The live reactor slots
are mutated only by scheduler reactor owners. The live network slots are
incremented from scheduler host-event admission for successful network
lifecycle and byte/connection events. For compatibility, `would_block()`
currently counts every admitted host event whose status is `WouldBlock`,
including non-network host I/O; restricting it to Network event kinds would be
a public telemetry meaning change and remains unresolved. The source-private
network recorder owns the one `EventKind -> call/lifecycle slot` projection.
Every additive Network slot uses `counter::Accumulate`, so call, lifecycle,
would-block,
admission-rejection, and byte evidence all saturate at `UINT64_MAX`.
`NetworkStats::bytes_received()` and `bytes_sent()` sum successful basic,
datagram, and vectored event `completed_bytes` at that same O(1) commit point.
Zero-byte, would-block, and failed events add zero bytes. Operation results and
network wrappers do not mutate the scheduler snapshot returned by
`Session::Result::tasks()` or copy local stats snapshots.

## Public API

```cpp fragment
YieldOp yield() noexcept;
SleepOp sleep(std::chrono::nanoseconds duration) noexcept;
Status join(Handle handle) noexcept;
Status join_all(std::span<const Handle> handles) noexcept;
Status scope(Callable&& callable);

namespace rund::host::io {
IoOp readable(FdView fd) noexcept;
IoOp writable(FdView fd) noexcept;
ReadOp read_some(FdView fd, std::span<std::byte> buffer) noexcept;
WriteOp write_some(FdView fd, std::span<const std::byte> buffer) noexcept;
ReadResult read_some_blocking(FdView fd,
                              std::span<std::byte> buffer) noexcept;
WriteResult write_some_blocking(
    FdView fd, std::span<const std::byte> buffer) noexcept;
}

template <class T>
class channel {
 public:
  channel() noexcept;
  channel(const channel&) = delete;
  channel& operator=(const channel&) = delete;
  channel(channel&&) noexcept;
  channel& operator=(channel&&) noexcept;
  ~channel() noexcept;
  static channel make(std::size_t capacity) noexcept;
  explicit operator bool() const noexcept;
  bool ok() const noexcept;
  ReasonCode code() const noexcept;
  std::string_view error() const noexcept;
  std::size_t capacity() const noexcept;
  SendOp send(T value) noexcept(std::is_nothrow_move_constructible_v<T>);
  RecvOp recv() noexcept(std::is_nothrow_move_constructible_v<T>);
  Status close() noexcept;
};
```

`SendOp` and `RecvOp` are move-only await tokens. They are consumed as
`co_await channel.send(value)` and `co_await channel.recv()` inside
`Task<T>`; `await_resume` returns `Status` and `ReceiveResult<T>`
respectively. There is no synchronous send/receive or alternate source-batch
channel API.

`ReadOp` and `WriteOp` are likewise move-only await tokens. Task code uses
`co_await io::read_some(fd, buffer)` and
`co_await io::write_some(fd, buffer)`. The `*_blocking` functions are only for
callers outside an active scheduler task and fail with `task_invalid` inside a
task. `io::readable` and `io::writable` accept only an admitted owner's
borrowed `io::FdView`, return
`IoOp` values in a coroutine, and must also be consumed with
`co_await`. There is no raw-integer readiness overload in the public task
headers, `Handle` friendship, or umbrella surface; the native descriptor is
unwrapped only by the source-private scheduler bridge. Read/write await tokens
do not take a stop token: once admitted, scope drain preserves the frame and waits
for the native transfer and deterministic completion commit. Use nonblocking
descriptors with readiness waits when cancellation latency must be bounded.

`T` must be move-constructible. If moving `T` into or out of scheduler storage
throws, the task follows the normal uncaught user exception path and reports
`task_failed`. Deterministic channel payload tests should use nothrow-movable
types.
