# Network API

Entry header: `<rund/net.hpp>`

Namespace: `rund::net`

[Back to API Reference](./API.md)

## Purpose

Use `rund::net` for deterministic, bounded socket bytes: lifecycle,
readiness, stream transfer, datagrams, scatter/gather, connections, framed
bytes, and byte-level server helpers.

The API is intentionally protocol neutral. Your application owns clients,
sessions, schemas, authentication, queues, reliability, ticks, rollback, and
the meaning of every byte.

## First Use

Ordinary task code passes a borrowed `SocketView`; runD owns readiness and
consumes its one-shot capability internally:

```cpp compile source=package/tests/consumer/example/network.cpp
#include <rund/net.hpp>
#include <rund/task.hpp>

#include <cstddef>
#include <span>

rund::task::Task<void> Read(const rund::net::SocketView socket,
                            const std::span<std::byte> bytes,
                            rund::net::ReceiveResult &result) {
  result = co_await rund::net::receive(socket, bytes);
}

rund::task::Task<void> SendPacket(const rund::net::SocketView socket,
                                  const std::span<const std::byte> bytes,
                                  const rund::net::Address peer,
                                  rund::net::datagram::SendResult &result) {
  result = co_await rund::net::datagram::send(socket, bytes, peer);
}
```

Both awaits produce the operation result directly in the caller's task. They
add no child task, packet copy, or second result layer.

## Address And Ownership

`Socket` is the move-only RAII owner of one socket capability. `open` returns
that owner in `OpenResult`; move it into its final owner:

```cpp fragment
rund::net::OpenResult opened = rund::net::open();
if (!opened) {
  return opened.exit_code();
}
rund::net::Socket socket = std::move(opened.socket);
```

`SocketView` is the trivially copyable, 16-byte borrowed handle used by
operations. Obtain it with `socket.view()` and pass it by value. A view never
owns or closes the capability, so the owning `Socket` must remain alive until
all work admitted through that view has finished. Calling `view()` on a
temporary owner is rejected by the type system.

Moving `Socket` transfers ownership. Destruction retires its generation and
closes the native capability exactly once. Ordinary code therefore has no
cleanup branch. `socket.close()` exists for early release or when the caller
needs the typed `CloseResult`; it is not a routine end-of-scope step. Close and
destruction may wait for an already active native-operation lease to finish.

Native descriptors and native socket-address structures are not part of the
API.

Create addresses with:

- `Address::ipv4(bytes, port)`
- `Address::ipv6(bytes, port)`
- `Address::loopback(family, port)`

`Address::bytes()`, `family()`, and `port()` expose the canonical semantic
tuple. Platform padding and byte order never enter address identity.

## Lifecycle

| Operation | Result |
| --- | --- |
| `open(OpenOptions)` | `OpenResult` containing an owning `Socket` |
| `bind(view, address)` | `BindResult` |
| `listen(view, backlog)` | `ListenResult` |
| `local(view)` | `LocalResult` |
| `shutdown(view, mode)` | `ShutdownResult` |
| `nonblocking(view, enabled)` | `NonblockingResult` |
| `socket.close()` | `CloseResult` for explicit early release |

`OpenOptions` uses typed `Family` and `Transport`. Network work in a scheduler
task requires a nonblocking socket; `open` enables it by default.

## Bytes

An admitted non-empty stream or datagram operation combines one readiness wait
with one scalar native attempt. The dedicated operation stores the borrowed
view, caller span, and datagram peer when present. Entering `co_await` invokes
one compiled shape owner, creates the single readiness wait lazily for an
admitted non-empty operation, then delegates to the same prepared terminal
used by advanced code. The operation is move-only and single-await; moving or
awaiting consumes its view, so a moved-from or repeated operation fails before
native I/O. Span, datagram-capacity, and peer shape are checked before
readiness, so an invalid ordinary operation never parks. The prepared consumer
does not recheck that shape.

An empty stream span completes immediately without a reactor wait or native
transfer call, while preserving socket validation and its zero-byte host
observation. An empty datagram is a real packet and keeps the ordinary
readiness and native-attempt path.

The ordinary verbs return concrete public operation types. Retain one only
when its lifetime must cross a local expression:

```cpp fragment
rund::net::Receive operation = rund::net::receive(socket.view(), bytes);
rund::net::ReceiveResult result = co_await std::move(operation);
```

Stream operations are `rund::net::Receive` and `rund::net::Send`; datagram
operations are `rund::net::datagram::Receive` and
`rund::net::datagram::Send`. Awaiting them yields the corresponding
`ReceiveResult` or `SendResult`. Internal readiness templates are not the
ordinary verb's return type and never enter the application's stored type.

| Shape | Receive | Send |
| --- | --- | --- |
| stream in a task | `co_await receive(view, bytes)` | `co_await send(view, bytes)` |
| datagram in a task | `co_await datagram::receive(view, bytes)` | `co_await datagram::send(view, bytes, peer)` |
| synchronous outside a task | `direct::receive(view, bytes)` | `direct::send(view, bytes)` |
| prepared advanced path | `receive(Ticket&&, bytes)` | `send(Ticket&&, bytes)` |
| scatter/gather advanced path | `batch::receive(Ticket&&, buffers)` | `batch::send(Ticket&&, slices)` |

`direct` is only for synchronous work outside an active scheduler. Task code
uses the SocketView awaitable unless it is already coordinating explicit
readiness.

All buffers are caller owned. A successful byte result reports the exact
completed prefix in `bytes`. Datagram results also carry canonical peer
identity. Scatter/gather consumes `batch::Buffer` or `batch::Slice` spans and
does not create a temporary packet.

## Advanced Readiness

Explicit readiness is for timed waits, many-socket fan-in, reusable sets,
batch operations, drains, accept, and connect completion. A successful wait
returns one move-only `ready::Ticket` that authorizes exactly one matching
operation. Reuse, a stale generation, or the wrong interest fails before a
native call.

Single socket:

- `co_await ready::read(socket.view())`
- `co_await ready::write(socket.view())`

Deadline:

- `co_await ready::timed::read(socket.view(), timeout)`
- `co_await ready::timed::write(socket.view(), timeout)`

Cancellable deadline:

- `co_await ready::timed::read(socket.view(), timeout, token)`
- `co_await ready::timed::write(socket.view(), timeout, token)`

Ordinary scalar stream and datagram code does not assemble this pair itself;
their SocketView awaitables perform the same wait and consumption internally.

Many sockets:

- `co_await ready::many::wait(requests, events, budget)`
- `co_await ready::many::wait(requests, events, timeout, budget)`
- `co_await ready::many::wait(requests, events, timeout, token, budget)`

The token overloads borrow one `task::stop_token`. A requested stop resumes
the same wait with `ReasonCode::TaskCancelled`; it does not create a second
readiness result or retain a registration after completion.

Every `ready::Request` contains a `SocketView`; every completed `ready::Event`
owns the corresponding one-shot ticket. Many-socket output preserves
request-index order. `ready::many::Budget` bounds completed event work, and
the application supplies output storage.

Reusable set:

- `ready::create(Config)` returns `ready::Status`; use its `set` only after
  success.
- `ready::add`, `ready::remove`, and `ready::clear` return that same typed
  status family.
- `ready::destroy` is the single terminal release for the plain `Set` identity.
- `co_await ready::many::wait(set, events, budget)`
- `co_await ready::many::wait(set, events, timeout, budget)`

`Config::max_members` reserves the exact member bound. Clear retains that
storage for reuse; destroy retires it. Clear or destroy cancels a currently
parked set wait with `TaskCancelled`, and a stale Set fails instead of aliasing
a later generation.

The generation is validated when readiness is registered and again when the
ticket is consumed. The registration lease ends before parking so socket close
can invalidate a wait without blocking on it. The prepared operation takes its
own short lease across the native attempt; retaining one lease across both
phases could deadlock close against the parked task. Claiming carries the sole
socket view and interest into the operation. Both validations address the
stable slot directly in constant time, with no descriptor-to-entry search, and
the prepared operation is the only native descriptor and host-id projection
used by byte I/O and its host event.

## Connections

- `accept::one(Ticket&&)` performs one nonblocking accept.
- `accept::prepare(Result&&, options)` moves an accepted capability into its
  configured owner.
- `accept::drain(view, budget, handler)` performs bounded accept work; the
  ticket overload allows an already prepared readable ticket.
- `connect::start(view, address)` starts a nonblocking connect.
- `connect::finish(Ticket&&, address)` validates completion using the writable
  ticket.

`server::serve` builds serial or caller-slot-bounded peer task lifecycle over
the same accept authority. `server::Options::listener` is a borrowed
`SocketView`. `server::Peer` owns its accepted `Socket`, so a handler takes it
by value and scope exit completes every terminal path without a cleanup branch.
`server::PeerResult`, `server::Options`, and `server::Result` carry bytes and
lifecycle only; they are not player or session objects.

`drain::read(Ticket&&, bytes, Budget, callback)` and
`drain::write(Ticket&&, bytes, Budget[, callback])` share the single
`Budget::max_operations` bound. The two result types expose read- and
write-specific progress; there are no mirrored root-level drain entry points.

- `server::PeerResult::complete()` records success;
- `server::PeerResult::stop()` records an intentional stop;
- `server::PeerResult::fail(code)` preserves an exact valid failure.

These factories allocate nothing. There is intentionally no generic outcome
conversion: the handler must interpret operation-specific progress, including
partial bytes and successful timeout terminals, before choosing its peer
terminal.

The handler has one terminal shape. Parallel serving joins every admitted peer
and reports the non-success reason belonging to the lowest accept index, even
when a later peer completes first. Admission and join failures retain their
coordinator precedence.

Awaiting `server::serve` returns `server::Result` directly, so coordinator and
server failures use one typed check:

```cpp fragment
const rund::net::server::Result result =
    co_await rund::net::server::serve(options, handler);
if (!result) {
  log(result.code(), result.error());
}
```

`server::Task` is move-only and adds no hidden peer storage or second coroutine.

### Bounded Server

The handler receives the owning peer and returns one typed terminal. The
listener remains a borrowed view and the accept count is explicit:

```cpp compile source=package/tests/consumer/example/server.cpp
#include <rund/net.hpp>
#include <rund/task.hpp>

#include <array>
#include <cstddef>
#include <utility>

rund::task::Task<rund::net::server::PeerResult>
Handle(rund::net::server::Peer peer) {
  std::array<std::byte, 4096u> bytes{};
  const rund::net::ReceiveResult received =
      co_await rund::net::receive(peer.socket.view(), bytes);
  if (!received) {
    co_return rund::net::server::PeerResult::fail(received.code());
  }
  co_return rund::net::server::PeerResult::complete();
}

rund::task::Task<rund::net::server::Result>
Serve(const rund::net::SocketView listener) {
  co_return co_await rund::net::server::serve(
      rund::net::server::Options{
          .listener = listener,
          .accepts = {.max_accepts = 64u},
      },
      [](rund::net::server::Peer peer) {
        return Handle(std::move(peer));
      });
}
```

Exact counter, ordering, allocation, failure-priority, and capacity assertions
belong to the focused runtime server contracts rather than this first-use code.

## Options And Limits

`option::set` and `option::get` accept the closed `option::Name` set. Raw
platform levels, numeric option ids, and arbitrary option buffers are not
exposed.

`limits()` returns active network capacity and ready-set use.
Configured storage is the complete pool; capacity exhaustion returns its typed
failure.

## Framing And Flow

`frame::Header` carries the admitted payload byte count. `frame::Limit` bounds
that count and `frame::IoLimit` additionally bounds native read/write attempts.
`frame::encode_length(bytes, out, limit)` and
`frame::decode_length(in, limit)` encode and validate the canonical four-byte
big-endian header. Their `frame::Result` must succeed before its `bytes` value
is used. A buffer shorter than four bytes reports `NetFrameHeaderTooSmall`;
a decoded or requested payload above `Limit::max_bytes` reports
`NetFrameTooLarge`.

`frame::write` and `frame::read` have ticket-consuming overloads for prepared
work and `SocketView` coroutine overloads that obtain the ticket internally.
They transfer a bounded four-byte length prefix and payload. They provide byte
framing only, not a packet schema or delivery rule. `frame::ReadResult` and
`frame::WriteResult` expose separate header/payload progress, would-block and
budget exhaustion; `complete()` is true only after both portions finish.

`flow::reserve`, `flow::release`, `flow::record_send`, and
`flow::record_receive` provide pure bounded byte accounting through
`flow::State`, `flow::Limit`, and `flow::Result`. The folder and namespace are
the same authority; the `net` root has no mirrored accounting entry points.

## Results

Every network result has one private `rund::ReasonCode` status owner. Read it
with `code()`; consumers cannot mutate it. `ok()`, explicit truth conversion,
`error()`, and `exit_code()` derive from that same value. Progress and terminal
details stay separate:

- stream terminals are `ReceiveResult` and `SendResult`;
- datagram terminals are `datagram::ReceiveResult` and
  `datagram::SendResult`;
- all byte results expose completed `bytes`;
- timed tickets expose `ready()` and `timed_out()`;
- drains expose would-block and budget exhaustion;
- frame results expose complete and partial progress.

## Replay And Telemetry

Network observations participate in the same runD record/replay stream as
other host inputs. Operation order, socket identity, result, byte count,
canonical peer identity, and payload hash are compared without exposing a
native replay socket.

Use `Session::Result` `result.tasks().network()` for network calls, bytes,
lifecycle events, would-block results, and admission rejection. Use
`result.tasks().reactor()` for readiness registration, wake, cancellation, and
capacity evidence. A ticket rejection is preserved in the operation's typed
result without incrementing the matching native-call counter. Prepared
consumption removes socket registry lookup/resolution, not the required
generation and interest checks.

## Performance Contract

- operations borrow caller spans;
- ordinary stream and datagram awaitables add no coroutine frame, task
  admission, payload copy, or runD heap storage;
- each ordinary scalar call performs one readiness decision and one
  prepared-ticket consumption through the explicit advanced terminal; an
  immediately ready call parks no reactor registration and a suspended call
  parks exactly one;
- scatter/gather calls allocate no runD heap storage;
- accept and many-readiness work is bounded before execution;
- payload identity scans only completed bytes;
- `Socket` and `SocketView` are each 16 bytes; ownership is represented only
  by `Socket`, while both carry one stable entry projection and generation;
- readiness registration resolves the view once and validates its generation;
- consuming a prepared ticket performs zero registry lookups/resolutions and
  constant-time generation and interest checks before the native attempt;
- a consumed or wrong-interest ticket is rejected before acquiring a
  generation lease; a valid-interest stale ticket fails that constant-time
  lease check before any native call;
- for a single-attempt send or receive, a valid successful ticket increments
  its matching `send_calls()` or `recv_calls()` counter by exactly one, while
  wrong-interest receive, repeated-consume send, and retired-generation send
  increment both counters by zero;
- the focused warm-path contract executes 1,024 successful prepared sends and
  1,024 repeated-consume rejections with zero runD heap allocations; each
  successful byte is drained before the next send, so host socket-buffer
  capacity is not a hidden test input, and the peer receives only the 1,024
  successful bytes;
- close retires the generation before native release and may wait for active
  operation leases, preventing descriptor reuse from aliasing admitted work.

See [the owning contract](https://github.com/sigee-min/runD/blob/main/node/docs/contracts/net.md) for exact
ordering, identity, and complexity laws.
