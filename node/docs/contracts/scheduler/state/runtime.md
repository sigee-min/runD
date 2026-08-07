# Scheduler State Runtime

This page owns stop-token identity, configuration normalization,
resource-budget defaults, and runtime kernel-provider interaction.

## Stop Tokens

`stop_source::create()` succeeds only inside an active scheduler runtime scope.
The scheduler stores scheduler id, reset epoch, source id, generation, and
requested state; it does not store callbacks, user payloads, or external
cancellation work. Scheduler id prevents tokens from one runtime object from
aliasing another runtime object, and reset epoch prevents tokens from an older
scope from aliasing records after `Reset()`.

A first `request_stop()` transition removes matching timed reactor waits,
many-wait groups, paired timeout timers, and ready backlog entries, then wakes
affected tasks in task-id order with `TaskCancelled`. Repeated requests for the
same source are idempotent and do not enqueue duplicate wakes.
`stop_token::state()` is a sequenced scheduler primitive: it traps through
the scheduler primitive boundary, waits for the current commit ticket, reads
the token record, and completes the primitive commit. Its `StopState` separates
query success from the explicit `requested()` value. `request_stop()` returns
`Status`; it never reuses the requested-state bit as operation success.

## Configuration

`rund::SchedulerConfig` is the one scheduler configuration schema. The public
`SessionConfig::scheduler` member and the Scheduler carry that exact type; no
facade projection, mirror record, or scheduler-private capacity exists. All 22
scheduler limits, including every network admission limit, are public through
`SessionConfig::scheduler`.

The compiled configure stage inside `Session::open` is the only normalization
boundary. Let `W >= 1` be the resolved Session worker width, `T` the
`std::size_t` trace capacity, and `U` the largest `std::uint32_t` value. For
requested values `w` and `o`, it computes

```text
effective_task_workers = (w == 0) ? W : w
effective_observation_capacity = (o == 0) ? min(T, U) : o
```

The `min(T, U)` clamp is evaluated before conversion to `std::uint32_t`, so a
64-bit trace capacity cannot wrap at the storage-width boundary. No later
layer reinterprets either zero sentinel.

The facade and runtime configuration entrypoints take configuration by value.
An lvalue is copied once when the caller transfers ownership; an rvalue is
moved. `SessionConfig::replay` then moves expected host events, payload archive,
and storage policy into the persistent Scheduler session. Runtime lifecycle
state does not keep a second replay configuration copy.

| Field | Meaning |
| --- | --- |
| `SessionConfig::workers` | Kernel backend width. `0` resolves to the host worker hint, then at least `1`. |
| `SessionConfig::random_seed` | Session-bound seed admitted through the scope for deterministic host random streams. |
| `scheduler.task_workers` | Node scheduler lane count. `0` inherits resolved `workers`. A normalized value below `1` fails with `task_workers_invalid`. |
| `scheduler.task_capacity` | Maximum simultaneously retained task records and coroutine frames. Joined terminal tasks retired by the control plane may be reused. Default `1024`. |
| `scheduler.ready_queue_capacity` | Maximum independent-spawn ready backlog. The single ready FIFO reserves `task_capacity` slots so already-admitted primitive wakes and awaited child continuations are lossless. Explicit `spawn(...)` remains independently bounded even inside a task. Logical ready depth may transiently exceed this admission bound but remains at most `task_capacity`. Default `1024`. |
| `scheduler.coroutine_frame_bytes` | Maximum admitted coroutine-frame payload bytes. Default `4096`. |
| `scheduler.coroutine_frame_alignment` | Maximum admitted coroutine-frame alignment. Default `16`. |
| `scheduler.task_result_bytes` | Maximum admitted typed task-result payload bytes. Default `128`. |
| `scheduler.task_result_alignment` | Maximum admitted typed task-result alignment. Default `64`. |
| `scheduler.timer_capacity` | Live positive-duration sleep wait nodes. Default `1024`. |
| `scheduler.channel_capacity` | Live scheduler-owned channel records. Default `1024`. |
| `scheduler.channel_buffer_capacity` | Total reserved buffered channel slots. Default `65536`. |
| `scheduler.channel_wait_capacity` | Total parked channel send/recv wait nodes. Default `4096`. |
| `scheduler.reactor_wait_capacity` | Live scheduler reactor wait nodes for IO/net fd readiness. Default `1024`. |
| `scheduler.net_ready_set_capacity` | Live persistent network ready-set records in one scheduler runtime. Default `256`. |
| `scheduler.net_ready_set_member_capacity` | Live members allowed per persistent network ready set after `ready::Config::max_members` is applied. Default `4096`. |
| `scheduler.net_iov_capacity` | Maximum caller-owned vectored IO slices admitted by network vectored APIs. Default `64`. |
| `scheduler.net_datagram_capacity_bytes` | Maximum caller-owned UDP datagram byte count admitted by network datagram APIs. Default `65507`. |
| `scheduler.net_socket_registry_capacity` | Active-runtime socket generation registry admissions. Default `65536`. |
| `scheduler.host_handle_capacity` | Simultaneously active physical host handles admitted to canonical replay identity. Default `1024`; `0` disables nonzero host-handle replay identity. |
| `scheduler.reactor_ready_budget` | Maximum canonical ready waits consumed by one reactor drain. `0` derives the budget from ready-queue capacity. |
| `scheduler.observation_capacity` | Copied observation records retained in the runtime report. `0` inherits `trace_capacity` with the width-safe clamp above. |
| `scheduler.host_io_capacity` | Maximum concurrently admitted task `ReadOp`/`WriteOp` slots, including the active syscall and queued operations. Configure prepares one retained exact-capacity array; equal-capacity reset does not allocate or scan it. On the supported 64-bit ABI the private array is at most `104 * host_io_capacity` bytes. Default `64`; `0` disables task host IO. |
| `scheduler.host_event_capacity` | Retained scheduler host-event rows. Default `1024`. |
| `scheduler.host_payload_capacity_bytes` | Logical task host-IO payload bytes retained for record/replay. Default `1048576`. |

The default resource-budget authority is the relationship between the one
`SchedulerConfig` value and runtime reports:

| Relationship | Contract |
| --- | --- |
| Single schema | `SessionConfig::scheduler` and Scheduler configuration are the same `rund::SchedulerConfig` type. |
| Complete public limits | Generic host-handle identity, persistent ready-set count, ready-set member count, vectored IO slice count, datagram bytes, and active-session socket registry count are independent public fields of `SessionConfig::scheduler`. |
| Reactor ready budget derivation | A default `reactor_ready_budget` of `0` means reactor drains derive the effective ready budget from `ready_queue_capacity`; it is not an unbounded native-backend poll policy. |
| Runtime report projection | `task::Stats::resources()` materializes the current task, coroutine-frame, IO, and network capacities only. |

## Runtime Session Ownership

A configured `Session` owns one persistent Scheduler through its internal
Runtime. `Session::open` creates bounded records, coroutine-frame arena, lanes,
replay storage, and Kernel provider once. `Session::scope` borrows that owner
and installs it as the caller's active scheduler; the Scope frame does not
allocate, configure, reset, or destroy another Scheduler.
`Session::compute(job).submit()` uses the same root task-record authority
without entering a Scope.

The Scheduler is also the sole configured owner of expected host events and
the normalized expected host-payload store. Event vector storage is transferred
without element copies. One source-private immutable `payload::Bytes` owner is shared by each
archive chunk and its normalized store blob; configuration copies retain that
owner but never copy or expose mutable encoded bytes. Archive validation may
read those bytes and may allocate different-type record/piece metadata.

Recording exports a metadata snapshot whose encoded owners are independent of
the Store's later lifetime. Consecutive `Session::scope` calls may append more
canonical records to the persistent Store, but a previously returned
`Session::Result` retains its original chunk owners and metadata. Store clear,
Session drain, and Session close cannot invalidate that prior report.

Plan install, plan clear, and Scheduler reset clear expected replay cursors,
failure state, evidence vectors, payload storage, byte accounting, and active
input capture through one private `ClearReplay` transition. Plan ownership,
retained input allocation, capture-token generation, and logical-clock reset
remain explicit at their narrower lifecycle boundaries; they are not boolean
modes hidden inside the common transition.

No Scheduler field may retain a span, `data()` pointer, `c_str()` pointer, or
other view into a consumed `SessionConfig` or archive object.
The spill-directory path remains an owned `std::string` in each store; its SSO
or heap representation is never used as identity and no pointer into it escapes
configuration.

Root Scope and external Compute submit/wait/retire operations are serialized
by the Runtime control gate. A coordinator already running on the same
Scheduler submits and retires without re-entering that gate. Lane-owned task
bodies remain parallel. A Compute submission cannot create a lane, OS thread,
`std::future`, or nested scheduler.
Session drain closes Compute admission, and close/teardown reset the persistent
runtime only after accepted Compute work is complete.

Scheduler reset clears ready-set storage but does not reset the process-wide
ready-set slot-id issuer. A `ready::Set` may persist across scopes of the same
open Session, but a handle from an earlier reset/open lifetime or another
Session is stale even when used through the currently active Scheduler. Raw
ready-set numbers remain opaque and are excluded from deterministic trace,
host-event, ordering, and replay identity; only the scheduler-local physical
slot count participates in capture-mutation state.

The Runtime control gate serializes external callers; it is not a second
Scheduler-state lock. External Compute wait may retire one completed
coordinator while another coordinator is still committing. Pending root-join
range mutation, task-record retirement, snapshot batch materialization, and
the resulting logical batch flush therefore enter the same Scheduler commit
ticket sequencer as lane commits. The ticket is held only for bounded metadata;
it is never held across a progress loop or backend wait. Backend work remains
parallel and waiting for one Compute task does not wait for an unrelated
backend submission.

Root-join retirement retains one fixed pending telemetry range. Contiguous task
ids with the same result code extend that range; a semantic boundary flushes
and resets it. Task records are recycled immediately and have no mirrored
pending-retirement bits. The range owns no task lifetime and performs no heap
allocation, and allocator growth policy cannot influence trace packet
grouping.

## Kernel Provider

Every task lane installs the runtime kernel provider before running records and
restores it on lane exit.

`Runtime::parallel_runtime` returns one thread-local workspace per
task lane.

The built-in Runtime Kernel backend is the same fixed scheduler-lane resource,
not a separate pool. External/root Runtime scope code may use synchronous
`kernel::each(kernel::par())`. A leaf or coroutine task already occupying a
lane may not synchronously re-enter that backend: it fails with
`pool_nested_dispatch` instead of blocking a lane on work queued to the same
lanes. Suspending parallel work uses `Session::compute(job)`, whose coordinator
submits partitions asynchronously and parks. Node never schedules Kernel
elements.

Resident Compute does not use the ambient thread-local provider lookup.
NodeHost passes the configured `WorkerBackend` explicitly to Kernel's prepared
tile executor; StandaloneHost uses the Program-owned backend explicitly.
