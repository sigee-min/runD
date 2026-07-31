# Examples

These examples assume your target links `runD::sdk` and uses C++20.

API reference: [Run](./Run.md), [Compute](./Compute.md),
[Telemetry](./Telemetry.md), [Tasks](./Tasks.md), [Replay](./Replay.md),
[Network](./Network.md), [Math32](./Math32.md), [Math64](./Math64.md), and
[Cluster](./Cluster.md).

## Bit-Identical CPU And GPU Compute

The repository's [first-screen example](../README.md#see-it) declares one
typed Flow, executes it on CPU, Metal, and Vulkan, and compares the returned
bytes directly. The selected backend is explicit, and an unavailable
accelerator returns a typed failure instead of silently running on CPU.

For the complete API path, start with
[One Flow, Three Backends](./Compute.md#one-flow-three-backends), then move to
Program reuse, resident Jobs, dependent Pipelines, and Batch submission without
changing the canonical graph.

## Minimal Run

```cpp compile run source=package/tests/consumer/example/runtime.cpp
#include <rund/session.hpp>

int main() {
  const rund::Session::Result result =
      rund::run(rund::SessionConfig{.workers = 1u}, [] {});
  return result.exit_code();
}
```

## Observe A Run

Bind one lvalue observer to the Session. Basic is the normal evidence level;
select Detail only when phase timing is part of the diagnosis. The executable
[Telemetry example](./Telemetry.md#first-use) is generated from its package
source; this index does not keep a second copy.

## Typed Compute Flow

Flow is the only public graph-building language. For compile-once resident
reuse, immutable branches, multi-output records, and bounded operations, start
with the executable [Compute first success](./Compute.md#first-success).
That API page is the single copy generated from the package source.

## Task Group

```cpp compile run
#include <rund/session.hpp>
#include <rund/task.hpp>

#include <array>
#include <atomic>
#include <cstdint>

rund::task::Task<void> Add(std::atomic<std::uint64_t> *counter,
                           rund::task::Status *completed) {
  std::array<rund::task::Handle, 2> slots{};
  rund::task::Group tasks{slots};
  const auto first = tasks.spawn("first", [counter] {
    counter->fetch_add(1u, std::memory_order_relaxed);
  });
  const auto second = tasks.spawn("second", [counter] {
    counter->fetch_add(2u, std::memory_order_relaxed);
  });
  if (!first || !second) {
    *completed = rund::task::Status::fail(first ? second.code() : first.code());
    co_return;
  }
  *completed = co_await tasks.join();
}

int main() {
  std::atomic<std::uint64_t> counter{0u};
  rund::task::Status completed{};
  rund::task::Status joined{};
  const rund::Session::Result result = rund::run([&] {
    const auto task = rund::task::spawn("add", Add(&counter, &completed));
    joined =
        task ? rund::task::join(task)
             : rund::task::Status::fail(task.code());
  });

  if (!result)
    return result.exit_code();
  if (!joined)
    return joined.exit_code();
  if (!completed)
    return completed.exit_code();
  return counter.load(std::memory_order_relaxed) == 3u ? 0 : 2;
}
```

`rund::task::Group::join()` resumes directly with one
`rund::task::Status`; scheduler failure
and the first child failure never become a nested result that the caller must
unpack.

## Replay A Session

Open one Session, declare one canonical Input, and reuse one callback for
Record and Replay. The Replay call substitutes recorded bytes, so the bound
producer is not invoked again. Start with the executable
[Record and Replay journey](./Replay.md#bind-once), then add persistence
through [checkpoint continuation](./Replay.md#checkpoint-and-resume).
Both pages are generated from their package sources; this index does not keep
a second copy.

## Network

Start with the compile-checked [ordinary stream and datagram
operations](./Network.md#first-use). Task code lends a `SocketView` directly;
runD owns its one readiness decision and returns the typed byte result without
a child task or manual ticket plumbing.

For stream framing, bounded serving, many-socket readiness, and telemetry, see
[Network API](./Network.md). Those are byte and lifecycle helpers; protocol
and session meaning stays in the application.

## Bounded Server

Declare the listener lifecycle once, give `server::serve` the exact accept
budget, and write only the peer handler. The compile-checked [bounded server
example](./Network.md#bounded-server) returns one `PeerResult`; RAII owns every
terminal path without a manual close ladder. Exact counter, ordering, and
failure-priority verification stays in the runtime contracts.
