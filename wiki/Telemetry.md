# Telemetry API

Entry header: `<rund/session.hpp>`

Namespace: `rund::telemetry`

[Back to API Reference](https://github.com/sigee-min/runD/blob/main/wiki/API)

## When To Use

Attach one `Sink` to `SessionConfig` to observe terminal Compute and Replay
work without creating another result owner. `Level::Basic` is the default and
adds no telemetry-owned timing. Select `Level::Detail` only while diagnosing
phase timing; its operator route is exactly `telemetry:detail`.

## Key Types

| Type | Purpose |
| --- | --- |
| `Sink` | One borrowed callback and observation level for a Session. |
| `Event` | One borrowed terminal Compute or Replay observation. |
| `Finding` | One measured cost with typed cause, reference, and action. |
| `Findings` | Allocation-free list of at most five derived findings. |
| `Level` | `Basic` or `Detail`. |

The callback borrows its `Event` for that call. Copy the trivially copyable
value only when it must outlive the callback. `event.findings()` derives
actionable observations from the event's raw counters without changing the
operation result or creating a second counter authority.

Use `members(finding.cause)` and `members(finding.action)` because a critical
path can contain equal phases. `name(...)` returns stable allocation-free text
for diagnostics. Standalone Compute exposes the same decision owner through
`Job::profile().findings()`.

`Sink` keeps its callback, context, and level private. A default Sink disables
observation; `bind(lvalue, level)` is the only way to create a configured Sink,
and the observer must outlive every terminal operation and `Session::close()`.
Use `describe(finding, writer)` or `describe(event, writer)` to stream stable
cost, cause, and action text through a borrowed `std::string_view` writer. The
helpers allocate and retain nothing; manual `members`/`name` traversal remains
available when an application needs its own structured presentation.

## First Use

Bind one lvalue observer, run the operation normally, and act on the findings:

```cpp compile run source=package/tests/consumer/example/telemetry.cpp
#include <rund/replay.hpp>

#include <cstdio>
#include <string_view>

int main() {
  auto observe = [](const rund::telemetry::Event &event) {
    rund::telemetry::describe(event, [](const std::string_view text) {
      std::printf("%.*s", static_cast<int>(text.size()), text.data());
    });
    std::putchar('\n');
  };

  rund::SessionConfig config{.workers = 1u};
  config.telemetry = rund::telemetry::bind(observe);
  rund::Session session{};
  const rund::Session::Status opened = session.open(config);
  if (!opened) {
    return opened.exit_code();
  }

  const rund::replay::Live result =
      rund::replay::live(session, [](rund::replay::Context &) {});
  const rund::Session::Status closed = session.close();
  if (!result) {
    return result.exit_code();
  }
  return closed.exit_code();
}
```

`bind` is the sole configured-Sink construction path and retains no observer
copy. Callback exceptions are contained and cannot replace the completed
operation result. `describe` streams directly through the writer and creates no
owned string or dynamic container.

Basic reports stable identities, counters, hashes, and non-time findings.
Detail adds non-overlapping prepare, work, and finish durations. Their parity,
clock-read, allocation, bounded-finding, complete cause/action mask, and
Compute/Replay projection assertions belong to the
[Telemetry contract](https://github.com/sigee-min/runD/blob/main/node/docs/contracts/telemetry.md) and the exact
`telemetry:detail` contract test, not to this first-use example.
