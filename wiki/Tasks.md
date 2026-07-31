# Tasks API

Entry header: `<rund/task.hpp>`

Namespace: `rund::task` (`rund::ReasonCode` is product-wide)

[Back to API Reference](https://github.com/sigee-min/runD/wiki/API) | [Run](https://github.com/sigee-min/runD/wiki/Run)

## When To Use

Use this header inside Session callbacks or other cooperative task code to create
cooperative work, wait for tasks, use scoped scheduling, yield, sleep, and
exchange values through channels.

## Key Types

| Type | Purpose |
| --- | --- |
| `Handle` | Opaque task handle returned by `spawn`; truthy when the task was accepted. |
| `Task<T>` | Coroutine task type for APIs that spawn coroutine work. |
| `Group` | Uses caller-provided handle slots for allocation-free coroutine fan-out/fan-in. |
| `channel<T>` | Typed coroutine channel with `make`, awaitable `send`/`recv`, and synchronous `close`. |
| `Status`, `IoResult` | Compact status for task/channel operations and the readiness result carrying `revents()`. |
| `Result<T>`, `ReceiveResult<T>` | Value-or-status results for coroutine tasks and channel receives. |
| `rund::ReasonCode`, `Stats` | Stable product reason codes and aggregate task statistics exposed by `Session::Result`; `network()` includes saturating received/sent byte totals. |
| `stop_source`, `stop_token`, `StopState` | Cooperative cancellation token family; stop requests return `Status`. |

## Key Functions

| Function | Purpose |
| --- | --- |
| `spawn(name, callable)` | Schedule a void callable with a task name. |
| `spawn(callable)` | Schedule a void callable with the default task name. |
| `spawn(name, Task<void>&&)` | Schedule a coroutine task. |
| `join(handle)` | Wait for one task handle. |
| `join(first, rest...)` | Wait for several task handles. |
| `join_all(handles)` | Wait for a span of handles. |
| `scope(callable)` | Run nested task work and wait for the scope to complete. |
| `yield()` | Cooperatively yield the current task. |
| `sleep(duration)` | Park the current task for a duration. |
| `co_await io::readable(fd)`, `co_await io::writable(fd)` | Wait for readiness through an admitted owner's borrowed `io::FdView`. |
| `co_await io::read_some(fd, buffer)`, `co_await io::write_some(fd, buffer)` | Perform task-safe record/replay byte IO. |

Channel family:

- `channel<T>::make(capacity)` creates a channel.
- `co_await channel<T>::send(value)` resumes with `Status`.
- `co_await channel<T>::recv()` resumes with `ReceiveResult<T>`.
- `channel<T>::close()` closes the channel.

Cancellation family:

- `stop_source::create()` admits a source only inside an active scheduler;
  test the returned source before using it.
- `source.token()` returns the copyable identity passed to cancellable waits.
- `source.request_stop()` returns `Status`. Its first success wakes matching
  waits in task-id order with `TaskCancelled`; repeated requests are
  idempotent and enqueue no duplicate wake.
- `token.state()` returns `StopState`. Check that query result before reading
  its independent `requested()` observation.

## Result Rules

- `Status`, `Result<T>`, `YieldOp`, `SleepOp`, `IoResult`, `IoOp`, and
  `ReceiveResult<T>` are truthy on success and expose `ok()`, `code()`,
  `error()`, and `exit_code()` from one result authority. Value-bearing
  results add dereference and arrow operators.
  Aggregate scheduler statistics belong to the completed Session result or scope
  rather than every leaf result.
- A status-valued `Task<Status>` resumes directly with `Status`.
  Await admission, execution, completion, and the returned status are flattened
  into that one value; `Group::join()` therefore never exposes a nested
  `Result<Status>`.
- Awaiting `io::readable(fd)` and `io::writable(fd)` resumes with `IoResult`,
  which exposes `ok()`, `code()`, `error()`, `exit_code()`, and readiness
  `revents()`.
  Transfer the native descriptor with `io::take_native_fd(native_fd)`, keep the
  returned owner alive, and pass `owner.view()`; raw integers and owning `Fd`
  values are not readiness overloads.
- `io::read_some` and `io::write_some` return move-only await tokens and must
  be consumed with `co_await`; their `ReadResult`/`WriteResult` carries the
  completed prefix count. The explicitly blocking variants are forbidden in a
  task. Accepted host read/write syscalls are drained to completion rather than
  interrupted by cooperative task cancellation.
- `Handle` is truthy only when a spawn was accepted.
- A supplied task name must be a non-null, non-empty, NUL-terminated string.
  Admission hashes it into deterministic spawn evidence without retaining a
  copy in the live task record. The unnamed overload uses `task`.
- `join()` with no handles succeeds.
- Channel receives return `ReceiveResult<T>`; check the result before
  dereferencing it.
- Channel operations report closed, capacity, wrong-runtime, and missing
  task-context conditions through the same `code()`, `error()`, and
  `exit_code()` projections.
- `yield()` and `sleep()` are meaningful inside an active task context.

## Example

```cpp compile run
#include <rund/session.hpp>
#include <rund/task.hpp>

int main() {
  bool channel_ok = false;
  rund::task::Status channel_status = rund::task::Status::success();
  auto result = rund::run([&] {
    auto values = rund::task::channel<int>::make(1);
    if (!values) {
      channel_status = rund::task::Status::fail(values.code());
      return;
    }
    rund::task::Status producer_status = rund::task::Status::success();
    rund::task::Status consumer_status = rund::task::Status::success();
    int received_value = 0;

    auto produce = [&]() -> rund::task::Task<void> {
      auto sent = co_await values.send(7);
      if (!sent) {
        producer_status = sent;
        co_return;
      }
      producer_status = values.close();
    };

    auto consume = [&]() -> rund::task::Task<void> {
      auto received = co_await values.recv();
      if (!received) {
        consumer_status = rund::task::Status::fail(received.code());
        co_return;
      }
      received_value = *received;
    };

    auto producer = rund::task::spawn("producer", produce());
    if (!producer) {
      channel_status = rund::task::Status::fail(producer.code());
      return;
    }
    auto consumer = rund::task::spawn("consumer", consume());
    if (!consumer) {
      channel_status = rund::task::Status::fail(consumer.code());
      return;
    }

    auto joined = rund::task::join(producer, consumer);
    if (!joined) {
      channel_status = joined;
    } else if (!producer_status) {
      channel_status = producer_status;
    } else if (!consumer_status) {
      channel_status = consumer_status;
    } else {
      channel_ok = received_value == 7;
    }
  });

  if (!result)
    return result.exit_code();
  if (!channel_status)
    return channel_status.exit_code();
  return channel_ok ? 0 : 2;
}
```

Group helper inside a coroutine:

```cpp fragment
std::array<rund::task::Handle, 2> slots{};
rund::task::Group tasks{slots};
const auto first = tasks.spawn("a", [] {});
const auto second = tasks.spawn("b", [] {});
if (!first || !second)
  co_return;
const rund::task::Status joined = co_await tasks.join();
if (!joined)
  co_return;
```
