# Host API

Entry header: `<rund/host.hpp>`

Namespaces: `rund::host::chrono`, `rund::host::env`, `rund::host`,
`rund::host::io`, `rund::host::random`, and `rund::host::timer`

[Back to API Reference](https://github.com/sigee-min/runD/blob/main/wiki/API) | [Network](https://github.com/sigee-min/runD/blob/main/wiki/Network)

## When To Use

Use this header for host-facing operations modeled through runD task results:
logical time, environment reads, file descriptor IO, random streams,
canonical task sleeps, and deadline timer projection. Ordered opaque workload
inputs belong to the root [Replay API](https://github.com/sigee-min/runD/blob/main/wiki/Replay).

For sockets and network readiness, use [Network](https://github.com/sigee-min/runD/blob/main/wiki/Network).

## Key Types

| Family | Key Types |
| --- | --- |
| Chrono | `chrono::nanoseconds`, `chrono::time_point`, `chrono::logical_clock` |
| Env | `rund::task::Result<std::string>` |
| Host events | `host::Event`, `host::EventKind`, `host::Status`, `rund::StableHash` |
| IO | move-only `io::Fd`, borrowed `io::FdView`, move-only `io::ReadOp`/`io::WriteOp`, `io::ReadResult`, `io::WriteResult`, move-only `io::OpenResult`, `io::CloseResult` |
| Random | `random::RunSeed`, `random::StreamId`, `random::DrawId`, `random::Stream` |
| Timer | `rund::task::SleepOp`, awaited as `rund::task::Status` |
| Task wait | `rund::task::YieldOp`, `rund::task::SleepOp`, awaited as `rund::task::Status` |

## Key Functions

Chrono and environment:

- `chrono::logical_clock::now()` returns a logical `time_point`.
- `env::get(name)` reads an environment value as
  `rund::task::Result<std::string>`.

Host events:

- `host::hash_event(event)`, `host::hash_events(events)`, and
  `host::event_kind_name(kind)` cover event summaries.

Random:

- `random::stream(seed, id)` and `random::stream(id)` create deterministic
  random streams.
- `random::active_run_seed()` returns the active run seed.
- `random::split(parent, child)` derives a child stream.
- `random::u64(stream, draw)`, `random::u32(stream, draw)`,
  `random::below(stream, draw, upper_exclusive)`,
  `random::range(stream, draw, lower_inclusive, upper_exclusive)`,
  `random::unit_f32(stream, draw)`, and
  `random::fill_bytes(stream, first_block, out)` draw values.

IO:

- `io::take_native_fd(fd)` transfers a native descriptor into the move-only
  `io::Fd` owner and replaces `fd` with `-1`.
- `io::replay_fd(host_id)` creates a replay-only `io::Fd` with no native close
  operation. Its id must match the next recorded host event; Record mode still
  requires `take_native_fd(fd)` and never accepts this synthetic owner.
- `fd.view()` creates a lightweight borrowed `io::FdView` for operations. A
  temporary owner cannot create a view, and views never carry close authority.
  Keep the owner alive until every operation using the view has completed.
- `co_await io::readable(view)` and `co_await io::writable(view)` wait for
  readiness inside a coroutine task. Both accept only `io::FdView`; there is
  no raw-integer or owning-handle overload.
- `co_await io::read_some(view, buffer)` and
  `co_await io::write_some(view, buffer)` perform task-safe byte transfers.
- `io::read_some_blocking(view, buffer)` and
  `io::write_some_blocking(view, buffer)` are synchronous and are accepted only
  outside an active scheduler task.
- An admitted task-safe read/write is not interrupted by a stop token. Use a
  nonblocking descriptor plus `readable`/`writable` when shutdown latency must
  be bounded.
- `io::pread_some(view, buffer, offset)` reads at an offset.
- `io::open_file(path, options)` opens a file.
- `fd.close()` releases an admitted or opened file descriptor early. Normal
  code relies on the `Fd` destructor for exactly one close.

Host I/O replay payload storage is selected through
`rund::SessionConfig::replay.storage`. `StorageMode::Memory` retains encoded
chunks in process memory; `StorageMode::Spill` uses the configured bounded
spill directory. Replay substitutes recorded payload bytes from that archive
and does not read the native descriptor again to recover missing worker bytes.
An unavailable native substrate is recorded as
`host::Status::Unsupported`, so replay preserves `IoUnsupported` instead of
collapsing it into a syscall failure.

Replay payload storage is private. `Session::Result` keeps its recorded payload
snapshot alive after later scopes append evidence or the Session closes, but
the SDK exposes no archive, chunk, codec, offset, or encoded-byte owner.

Timer helpers:

- `rund::task::sleep(duration)` performs the canonical logical duration wait.
- `timer::at(deadline)` sleeps until logical scheduler-visible time reaches
  `deadline`.

## Result Rules

- Host result structs carry one `ReasonCode code` plus operation-specific
  payload and native-error fields. Their `ok()`, truth conversion, `error()`,
  and `exit_code()` views derive from that code. Aggregate scheduler statistics
  are reported by the completed Session result or scope, not copied into every host
  result.
- IO byte results place the completed byte count in `bytes`; failed operations
  keep `bytes` negative and may set `native_error`.
- Dereference the environment `rund::task::Result<std::string>` only after it is
  truthy; failures expose the same `code()`, `error()`, and `exit_code()`
  projections as other task outcomes.
- Random draw functions are pure for the same stream id, seed, and draw id.
- Task sleep/yield helpers require an active task context.
- Timer helpers use scheduler-visible logical time. Awaiting the returned
  `rund::task::SleepOp` produces `rund::task::Status`.

## Example

```cpp compile
#include <rund/host.hpp>
#include <rund/task.hpp>

rund::task::Task<rund::task::Status>
ReadSample(rund::host::io::FdView fd) {
  auto stream =
      rund::host::random::stream(rund::host::random::StreamId{.value = 1});
  auto value = rund::host::random::range(
      stream, rund::host::random::DrawId{.value = 0}, 10, 20);

  std::byte buffer[64]{};
  auto ready = co_await rund::host::io::readable(fd);
  if (!ready) {
    co_return rund::task::Status::fail(ready.code());
  }
  auto read = co_await rund::host::io::read_some(fd, buffer);
  if (!read) {
    co_return rund::task::Status::fail(read.code());
  }
  (void)value;
  (void)read.bytes;
  co_return rund::task::Status::success();
}
```
