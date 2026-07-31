# Node Guide

This page maps the source-tree implementation interface and its synchronous
Session product projection. SDK consumers enter through one focused
`<rund/*.hpp>` domain or the declaration-free `<rund/rund.hpp>` composition
and link `runD::sdk`; they do not include the internal owner
headers listed here. Stable behavior belongs to the contract pages:
Session rules in [`contracts/runtime.md`](./contracts/runtime.md), host API
rules in [`contracts/host.md`](./contracts/host.md), and
scheduler law in [`contracts/scheduler/README.md`](./contracts/scheduler/README.md).
Network socket bytes and readiness are routed by
[`contracts/net.md`](./contracts/net.md).

## Authority

1. [`README.md`](./README.md) owns the node subsystem boundary.
2. This page maps the implementation interface and product projection.
3. [`contracts/`](./contracts/README.md) owns stable behavior.
4. Source-internal headers under `/node/include/node` define the subsystem
   implementation API.
5. Contract tests under `/node/tests/contract` prove the API shape.

The installed consumer surface is owned by
[`/package/docs/surface.md`](../../package/docs/surface.md), not by this guide.

## Internal Interface Owners

Repository implementation and contract code include the smallest owning
`<node/...>` header. There is no internal node umbrella. SDK applications use
one focused `<rund/...>` domain, or `<rund/rund.hpp>` when the translation unit
intentionally consumes every runD domain.

| Header | Purpose |
| --- | --- |
| `node/resource/envelope.hpp` | Sole worker, topology, and backend resource evidence envelope. |

Numeric declarations are owned by the direct
`<rund/evidence.hpp>` entry. Distributed run identity is owned by
`<cluster/cluster.hpp>`; neither has a mirrored Node identity header.

## Session

`rund::run` executes one local scope:

```text
configure -> start -> scope -> close -> capture trace
```

Reusable Sessions use the same terminal UX: call `close()` directly from
Running. That one blocking call rejects new admission, waits for an active
scope, cancels and drains resident Compute work, and returns at Stopped after
the deterministic draining and stopped transitions.
`drain()` is only for applications that intentionally need a visible
admission-shutdown phase before the final close.

The Session product owns its worker resources, discovered resources, lifecycle,
trace, telemetry emission, Scheduler integration, and resident Compute
job admission. It owns no domain run proposal, workspace pool, or manual
dispatch ledger.

`Session::Status` and the move-only `Session::Result` use the single outcome
shape owned by the [Session contract](./contracts/runtime.md#product-surface).
They expose the same `ok()`, `code()`, `error()`, and `exit_code()` observers;
`Status::state()` adds the lifecycle decision. `Session::scope` and
`rund::run` both return that one result value.

Host-IO and network operation results add their operation payload to one
private authoritative `ReasonCode`, observed through `code()`. Timed waits
treat `IoTimedOut` as a completed non-error result. Completion does not mean
that a wait became ready, a drain wrote all bytes, or frame IO completed.
Check `ready()`, `timed_out()`, `events`, `all_written`, or `complete()` when
the caller requires that payload state. Timed and many-readiness calls return
move-only operations: coroutine code uses `co_await`, while synchronous code
uses explicit `.wait()`.

`Session::scope(callback)` installs the node-owned kernel parallel provider and
cooperative task scheduler for ordinary callback code. Without an active
Session scope, kernel `par()` fails closed. Without an active task scheduler
scope, task APIs fail closed using scheduler-owned reason codes.

Compute execution accepts an already prepared resident `rund::compute::Job` or
`rund::compute::Pipeline`. `session.compute(job).submit()` schedules that same
Job, while `session.compute(pipeline).submit()` schedules the Pipeline's frozen
declaration-ordered Program sequence; neither recompiles a graph or assembles
buffers. `poll()` reports admission,
backend submission, terminal state, and one typed terminal `Reason` without
changing them. Its `code()` and `error()` observers derive from that Reason.
`wait()` returns the same failure projection with Compute counters, and
`job.read()` and `pipeline.read(buffer, output)` remain the explicit
output/download boundaries. CPU work uses the Session's fixed worker pool.
Metal/Vulkan work submits its existing prepared backend command
asynchronously, so a Session worker does not wait for device completion. The
selected backend never changes and there is no CPU fallback. Code that performs
this admission includes `<rund/compute/session.hpp>`; the focused
`<rund/session.hpp>` entry intentionally carries lifecycle declarations without
importing Job, Pipeline, or coroutine implementation headers.

| Header | Purpose |
| --- | --- |
| `rund/session.hpp` | Public Session lifecycle, repeated scope, telemetry configuration, actual-state snapshots, configured-resource observation, trace, and resident Compute admission. |
| `rund/task.hpp` | Public cooperative task, group, channel, cancellation, and wait surface. |
| `rund/host.hpp` | Public deterministic host service surface. |
| `rund/net.hpp` | Public meaning-neutral network surface. |
| `rund/replay.hpp` | Public record, replay, scenario, checkpoint, and retention surface. |
| `rund/evidence.hpp` | Public numeric contract and evidence surface. |
| `rund/compute.hpp` | Public typed Compute surface. |
| `rund/compute/pipeline.hpp` | Focused public prepared dependent-Program surface. |
| `rund/compute/session.hpp` | Public Job/Pipeline Session admission and common Request/Submission/Poll/Completion surface. |
| `rund/session/config.hpp` | Public Session identity, resources, telemetry, scheduler, replay, and deterministic seed configuration. |
| `node/runtime/runtime.hpp` | Source-private compiled lifecycle and scope owner used by Session. |
| `node/runtime/compute/access.hpp` | Source-private resident Job-state access used by Compute contracts; it owns no Session request API. |
| `node/runtime/backend.hpp` | Source-private worker-pool construction and backend validation. |
| `node/src/runtime/resource/discovery.hpp` | Source-private Session resource admission and topology projection. |
| `rund/session/trace.hpp` | Public runtime trace event, record, and bounded snapshot values. |
| `node/runtime/replay.hpp` | Product replay evidence records and strict replay checks over scheduler operation, observation, canonical input, host event, payload, and runtime trace hashes. |
| `node/runtime/replay/host.hpp` | Host event replay evidence encode/decode API routed by the host contract. |
| `rund/task/` | Installed support owners reached through `rund/task.hpp`; they are not separate product entrypoints and no second aggregate exists. |
| `node/host/` | Source-internal deterministic host owners reached through `rund/host.hpp`; no second aggregate exists. Canonical workload input belongs to the Replay facade. Network semantics are owned by [`contracts/net.md`](./contracts/net.md). |

## Network

The product network namespace is `rund::net`. Move-only `Socket` owns the
lifetime, copyable `SocketView` borrows its identity, and `Address` carries the
canonical peer value. Applications never admit or expose native descriptors.

Use a caller-owned `rund::task::Group` when one coroutine fans out over dynamic
socket work. The slots bound task admission and remove a hidden vector from
the warm path:

```cpp compile
#include <rund/net.hpp>
#include <rund/task.hpp>

#include <array>
#include <cstddef>
#include <span>
rund::task::Task<void> ReadSocket(rund::net::SocketView socket,
                                  std::span<std::byte> buffer,
                                  rund::task::Status *status) {
  const auto received = co_await rund::net::receive(socket, buffer);
  *status = received ? rund::task::Status::success()
                     : rund::task::Status::fail(received.code());
}

rund::task::Task<rund::task::Status>
ReadSockets(std::span<const rund::net::SocketView> sockets,
            std::span<std::array<std::byte, 4096>> buffers,
            std::span<rund::task::Status> results,
            std::span<rund::task::Handle> slots) {
  if (buffers.size() < sockets.size() || results.size() < sockets.size() ||
      slots.size() < sockets.size()) {
    co_return rund::task::Status::fail(
        rund::ReasonCode::TaskCapacityExceeded);
  }
  rund::task::Group tasks{slots};
  rund::task::Status admitted = rund::task::Status::success();
  for (std::size_t index = 0; index < sockets.size(); ++index) {
    const auto task = tasks.spawn(
        "net.read", ReadSocket(sockets[index], buffers[index], &results[index]));
    if (!task) {
      admitted = rund::task::Status::fail(task.code());
      break;
    }
  }
  const rund::task::Status joined = co_await tasks.join();
  if (!admitted) {
    co_return admitted;
  }
  if (!joined) {
    co_return joined;
  }
  for (std::size_t index = 0; index < sockets.size(); ++index) {
    if (!results[index]) {
      co_return results[index];
    }
  }
  co_return rund::task::Status::success();
}
```

The caller retains the `Socket` owners until the group has joined. Views make
that lifetime visible without copying a close capability into each task.

Listener setup uses only typed lifecycle operations:

```cpp fragment
rund::net::OpenResult opened = rund::net::open({
    .family = rund::net::Family::IPv4,
    .transport = rund::net::Transport::Stream,
});
if (!opened) {
  report(opened.code(), opened.error());
  return;
}
rund::net::Socket listener = std::move(opened.socket);
const rund::net::Address local =
    rund::net::Address::loopback(rund::net::Family::IPv4);
const rund::net::BindResult bound = rund::net::bind(listener.view(), local);
if (!bound) {
  report(bound.code(), bound.error());
  return;
}
const rund::net::ListenResult listening =
    rund::net::listen(listener.view(), 64);
if (!listening) {
  report(listening.code(), listening.error());
  return;
}
```

No cleanup branch is required; `listener` closes at scope exit. Ordinary stream
and datagram tasks `co_await` their SocketView operation directly. Explicit
readiness is reserved for deadlines, many-socket fan-in, reusable sets, batch
operations, drains, accept, and connect completion.

Datagram and scatter/gather calls borrow caller spans. `accept::drain`,
`drain::{read,write}`, `frame`, and `server::serve` bound work before execution.
They do not allocate protocol queues or own client/session meaning.

The complete operation, ordering, complexity, result, replay, and telemetry
contract is [Network](./contracts/net.md). The user-facing map and examples
live in [Network API](../../wiki/Network.md); this guide does not mirror
that reference.

## Guide Rules

- Add a `/node/include/node` header here only when it owns the source-internal
  subsystem interface. Public SDK promotion happens only in the package
  surface authority.
- Keep validation rules, fail reasons, Session behavior, and scheduler law in
  the owning contract pages.
