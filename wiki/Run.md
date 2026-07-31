# Run API

Entry header: `<rund/session.hpp>`

Namespace: `rund`

[Back to API Reference](https://github.com/sigee-min/runD/blob/main/wiki/API)

## When To Use

Use this header when an application wants the shortest path to configure and
execute a local session. `rund::run` opens a Session, applies the supplied
limits, runs the callback, drains scheduled work, closes the Session, and
returns a single `Session::Result`.

For task APIs used inside the callback, see [Tasks](https://github.com/sigee-min/runD/blob/main/wiki/Tasks).

The focused `<rund/session.hpp>` header exposes Session lifecycle, one-shot
`run`, configuration, Result, and telemetry through `runD::sdk`. Include the
other focused domains only when the callback uses them.

## Choose A Path

| Need | Use | Why |
| --- | --- | --- |
| One bounded callback | `rund::run(config, callback)` | Opens, drains, closes, and returns all evidence in one value. |
| Repeated ticks or experiments | `Session::open` then repeated `scope` | Reuses prepared workers, Scheduler storage, and Compute admission. |
| Resident Compute in either path | callback receives `Session&`, then `session.compute(job)` or `session.compute(pipeline)` | Keeps lifecycle and Compute admission on the same owner while reusing the common Request/Submission/Poll/Completion UX. |

Do not build a loop of one-shot `run` calls when the work belongs to one
session: that repeats setup and teardown `N` times. Open once, execute `N`
scopes, then close once. Call `drain()` first only when the application needs
to stop new admission and deliberately observe the Draining state.
The ordinary `close()` call is blocking: after publishing Draining it waits
for an active scope and cancels/drains resident Compute work before returning
success at Stopped.

## Related APIs

- [Replay](https://github.com/sigee-min/runD/blob/main/wiki/Replay) records and checks canonical session evidence.
- [Telemetry](https://github.com/sigee-min/runD/blob/main/wiki/Telemetry) observes terminal Session work through one sink.

## Key Types

| Type | Purpose |
| --- | --- |
| `rund::Session` | Reusable open session for repeated scopes and resident Compute submissions. |
| `rund::SessionConfig` | Session identity and resources, plus nested scheduler and replay policy. |
| `rund::SchedulerConfig` | The 22 scheduler admission, storage, worker, and evidence limits consumed directly by the Scheduler. |
| `rund::Session::Result` | Move-only outcome for one-shot and reusable Session scopes, with prepared-memory evidence, task statistics, observations, host events, and trace evidence. |
| `rund::Session::Snapshot` | Allocation-free observation of lifecycle state, active Compute count, active scope, and one typed observation code. |
| `rund::PreparedMemory` | Read-only prepared-capacity, high-water, overflow, and epoch evidence retained in a completed run. |
| `rund::Resources` | Host resource envelope discovered once by `Session::open`. |
| `rund::Topology`, `rund::EvidenceTruth` | NUMA, affinity, and worker-capacity evidence with explicit Unknown, Hint, or Verified truth. |
| `rund::Trace` | Bounded lifecycle and Compute trace snapshot. |

Common `SessionConfig` fields:

- `id`: run identifier.
- `workers`: worker count; `0` asks runD to choose a host-based width.
- `require_verified_numa`, `require_verified_affinity`, and
  `require_verified_worker_capacity`: reject `open()` unless the selected host
  evidence is `EvidenceTruth::Verified` for each requested property.
- `telemetry`: one lvalue-bound observation sink.
- `compile`: workers and queue capacity for the Session-owned Compute compile
  service.
- `trace_capacity`: bounded product trace retention.
- `scheduler.task_workers`: task worker count; `0` follows the resolved worker
  count.
- `scheduler.task_capacity`: maximum admitted task records and the physical
  global wake-queue bound.
- `scheduler.ready_queue_capacity`: independent `spawn(...)` ready-backlog
  admission bound. Awaited `Task<T>` children remain bounded by
  `task_capacity` and can progress after their parent is admitted.
  Already-admitted primitive wakes are lossless, so logical ready depth may
  exceed this value but remains bounded by `task_capacity`.
- Frame, result, timer, channel, reactor, and all five network fields are the
  other bounded scheduler resources. `SessionConfig::scheduler` is the exact
  `SchedulerConfig` consumed by the Scheduler, not a facade projection.
- `scheduler.observation_capacity` and `scheduler.host_event_capacity`:
  retained evidence records.
- `scheduler.host_handle_capacity`: simultaneously active physical handles in
  canonical replay identity. It is independent of network registry capacity;
  `0` disables replay identity for nonzero host handles.
- `scheduler.host_io_capacity` and
  `scheduler.host_payload_capacity_bytes`: admitted host-IO work and replay
  bytes.
- `replay`: bounded replay input, storage, and optional raw-diagnostic policy.
  Execution selection and expected evidence belong to the scope-level
  [Replay API](https://github.com/sigee-min/runD/blob/main/wiki/Replay), not Session configuration.
- `random_seed`: seed used by `rund::host::random::active_run_seed()` and
  default random streams.

On a successful `resources()` result, `workers` is the resolved width,
`topology` records each evidence truth and NUMA shape, and
`worker_capacity_milli` has one positive capacity value per worker only when
worker-capacity evidence is Verified. Callers check the `Resources` result
before using those observations.

## Key Functions

| Function | Purpose |
| --- | --- |
| `rund::run(config, callback)` | Run `callback` in a one-shot Session configured by `SessionConfig`. |
| `rund::run(callback)` | Run `callback` with the default `SessionConfig`. |
| `Session::open(config)` | Prepare and start one reusable Session. |
| `Session::scope(callback)` | Run and drain one callback scope without rebuilding the Session. |
| `Session::close()` | Stop admission, drain active scope/Compute work, and return Stopped in one blocking terminal call. |
| `Session::drain()` | Advanced control that stops admission while leaving the Session observable in Draining. |
| `Session::snapshot()` | Observe current state, active work, and observation failure without changing lifecycle. |
| `Session::trace()` | Copy the current bounded Session trace snapshot. |
| `Session::resources()` | Observe the resource envelope fixed by the successful `open()`. |
| `rund::record_memory(snapshot)` | Record one prepared-memory snapshot from the root run callback. Task-worker calls and codes outside the prepared-memory category fail without mutation. |

The callback is borrowed for the synchronous call; it is neither copied nor
retained. It may accept the active `rund::Session&` or no arguments. If both
forms are available, the Session-taking form is selected. The SDK
template is a thin callback trampoline; lifecycle execution has one compiled
owner, so each callback type does not instantiate another run state machine.

## Result Rules

- `code()` is the sole outcome authority and returns
  `rund::ReasonCode`.
- `ReasonCode::Ok` is the only success value; `ok()` and `operator bool`
  derive from it.
- `error()` returns an empty view for `Ok` and the stable code projection on
  failure.
- `exit_code()` returns `0` on success and `1` on failure.
- A default `Session::Result` uses `SessionResultMissing` and
  `session_result_missing`; an unexecuted value cannot appear successful.
- If configuration, start, scope, or close fails, `code()` is propagated
  from the failing phase without parsing text.
- Nested calls to `rund::run` from an active run are rejected with
  `"runtime_reentry_forbidden"`.
- `memory()`, `tasks()`, `observations()`, `events()`, `trace()`, and the
  derived `trace_hash()` are the sole public result evidence. The Result is
  move-only and the facade keeps no duplicate scope or compact-summary copy.
- `Trace::code` and each nested snapshot own one `ReasonCode`. A trace record
  owns one four-byte `TraceCode`, whose `Runtime` or `Compute` tag preserves
  the exact typed reason. `error()` derives text; no trace stores a borrowed
  reason pointer.
- `memory().capacity` is the single nested capacity-proof value. Its
  `ReasonCode code` is the sole proof outcome; `ok()`, truth conversion,
  `error()`, and `exit_code()` derive from it, and `valid()` admits exactly
  `Ok` or a defined prepared-memory failure. No borrowed reason pointer is
  retained. `task::Stats` does not carry a partial prepared-memory mirror.

The completion invariant is therefore
`result.ok() == static_cast<bool>(result) == (result.code() ==
rund::ReasonCode::Ok)`, with `result.error().empty()` on success and
`result.exit_code() == (result.ok() ? 0 : 1)`. Consumers normally observe
completion through `ok()` or truth conversion and inspect `code()` when control
flow needs an exact failure class.

## Reusable Session

```cpp compile run
#include <rund/session.hpp>

int main() {
  rund::Session session;
  const auto opened = session.open(rund::SessionConfig{.workers = 1u});
  if (!opened) {
    return opened.exit_code();
  }

  int operation = 0;
  for (int tick = 0; tick < 3; ++tick) {
    const auto completed = session.scope([tick] { static_cast<void>(tick); });
    if (!completed) {
      operation = completed.exit_code();
      break;
    }
  }

  const auto closed = session.close();
  return operation == 0 ? closed.exit_code() : operation;
}
```

## Example

```cpp compile run
#include <rund/session.hpp>
#include <rund/task.hpp>

int main() {
  rund::SessionConfig config{};
  config.id = 42;
  config.workers = 4;
  config.scheduler.task_workers = 4;
  config.random_seed = 1234;

  bool worker_done = false;
  bool worker_yielded = false;
  rund::task::Status worker_status = rund::task::Status::success();

  rund::Session::Result result = rund::run(config, [&] {
    auto work = [&]() -> rund::task::Task<void> {
      auto yielded = co_await rund::task::yield();
      if (!yielded) {
        worker_status = yielded;
        co_return;
      }
      worker_yielded = true;
    };
    auto worker = rund::task::spawn("worker", work());
    if (!worker) {
      worker_status = rund::task::Status::fail(worker.code());
      return;
    }

    auto joined = rund::task::join(worker);
    if (!joined) {
      worker_status = joined;
      return;
    }
    worker_done = worker_yielded;
  });

  if (!result)
    return result.exit_code();
  if (!worker_status)
    return worker_status.exit_code();
  return worker_done ? 0 : 2;
}
```
