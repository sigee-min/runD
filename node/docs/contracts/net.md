# Network

## Scope

This page owns the meaning-neutral network boundary exposed as `rund::net`.
Applications consume it through `<rund/net.hpp>` or the `<rund/rund.hpp>`
composition and link `runD::sdk`. The source
contract is split below `/node/include/rund/net/`; the directory hierarchy is
the complete API map. `rund::net` is the one public namespace,
`<rund/net.hpp>` is its focused entry. Move-only `Socket` is the sole lifetime
owner and `SocketView` is its non-owning identity projection. Ordinary stream
and datagram scalar I/O accept that view directly and own their one readiness
decision internally;
one-shot `ready::Ticket` remains the explicit prepared capability for
many-socket, reusable-set, timed, batch, and bounded-drain control.

runD owns socket identity, canonical addresses, bounded byte operations,
readiness ordering, host observations, replay comparison, resource limits,
and network telemetry. Applications own protocols, commands, clients,
sessions, authentication, reliability, retry, queues, ticks, rollback,
simulation state, and every interpretation of the bytes.

DNS, TLS, HTTP, WebSocket, packet schemas, retransmission, delivery policy,
and gameplay meaning are outside this contract.

## Authority

The authority path is:

1. this contract;
2. `/node/include/rund/net.hpp` and `/node/include/rund/net/`;
3. `/node/src/host/net/`;
4. `/node/src/runtime/task/scheduler/reactor/`;
5. `/node/tests/contract/runtime/task/net/`;
6. the installed SDK consumers under `/package/tests/consumer/`.

Platform socket conversion stays private below `/node/src/runtime/platform/`.
Public headers expose neither native descriptors nor platform socket-address
types.

## Product Surface

The top-level values and verbs are intentionally short:

| Area | Values | Verbs |
| --- | --- | --- |
| identity | `Socket`, `SocketView`, `Address`, `Family`, `Transport` | `Socket::view`, `Address::ipv4`, `Address::ipv6`, `Address::loopback` |
| lifecycle | `OpenOptions`, `OpenResult`, `BindResult`, `ListenResult`, `LocalResult`, `ShutdownMode`, `ShutdownResult`, `CloseResult` | `open`, `bind`, `listen`, `local`, `shutdown`, `Socket::close`, `nonblocking` |
| stream bytes | `Receive`, `Send`, `ReceiveResult`, `SendResult` | `receive`, `send` |
| datagrams | `datagram::Receive`, `datagram::Send`, `datagram::ReceiveResult`, `datagram::SendResult` | `datagram::receive`, `datagram::send` |
| scatter/gather | `batch::Buffer`, `batch::Slice` | `batch::receive`, `batch::send` |
| connections | `accept::Result`, `accept::Options`, `connect::Result` | `accept::one`, `accept::prepare`, `connect::start`, `connect::finish` |
| readiness | `ready::Interest`, `ready::Ticket`, `ready::Request`, `ready::Event` | `ready::read`, `ready::write` |
| timed readiness | `ready::timed::Result`, `ready::timed::Wait` | `ready::timed::read`, `ready::timed::write` |
| many readiness | `ready::many::Budget`, `ready::many::Result`, `ready::many::Wait` | `ready::many::wait` |
| reusable sets | `ready::Set`, `ready::Config`, `ready::Status` | `ready::create`, `ready::destroy`, `ready::clear`, `ready::add`, `ready::remove` |
| options | `option::Name`, `option::Value`, `option::Result` | `option::set`, `option::get` |
| limits | `Limits` | `limits` |
| bounded drain | `drain::Budget`, `drain::ReadResult`, `drain::WriteResult` | `drain::read`, `drain::write` |
| byte flow | `flow::State`, `flow::Limit`, `flow::Result` | `flow::reserve`, `flow::release`, `flow::record_send`, `flow::record_receive` |
| server | `server::Peer`, `server::PeerResult`, `server::Options`, `server::Task`, `server::Result` | `server::serve` |

`drain`, `frame`, `flow`, and `server` are bounded byte-level conveniences over
the same socket authority. Their folder and namespace names are identical;
they do not add protocol or domain meaning.

## Identity

`Socket` is an opaque move-only owner. It exposes validity, stable admitted
`id()`, an lvalue-only borrowed `view()`, and explicit early `close()`.
`SocketView` is trivially copyable but cannot close or transfer ownership.
Deleting `view()` on rvalues prevents a borrowed identity from being created
from an owner that dies at the end of the expression. Native descriptors and
generations remain private, so an SDK consumer cannot forge process-local
carriers.

The live identity is exactly `(stable registry slot, generation)`. A slot owns
one native descriptor while its generation is active. For a
nonnegative native descriptor, the public host id is derived as

```text
id = uint64(native) + 1
```

and an invalid descriptor derives id zero. Neither owner nor view stores that
id as a third field: doing so would admit disagreement with the registry slot
and copy eight redundant bytes at every boundary. The checked 64-bit product
layouts are therefore 16 bytes for `Socket` and 16 bytes for `SocketView`.
The public layout stores the slot as an opaque pointer; the private network
access owner is the only place that names or casts `SocketSlot`, so registry
implementation types do not enter the installed header dependency graph.
That friend owner is isolated below `rund::net::detail`; `rund::net` exposes no
top-level access or registry capability beside the product values and verbs.
This is a layout and copied-byte reduction; it is not by itself a wall-time
claim.

Admission stores one described native-fd snapshot in that slot. The snapshot
itself owns socket-object comparison: both sides must be described and their
device, inode, and object-type fields must match. Full mode remains available
to the separate reactor projection, but permission-bit differences do not
change socket identity. An inactive or closing slot stores the single
`Invalid` value with zero payload; there is no validity boolean or parallel
manual comparison authority in the registry.

The source-private admission result is one nondefault move-only ownership
value. `Success` contains one valid `Socket` and fixes its reason to `Ok`;
`Failure` contains no socket and requires a non-`Ok` reason. Moving the result
or taking its socket normalizes the source to `TaskInvalid` with no owner, so a
consumed value cannot still report success. The result cannot represent a
successful empty socket or a failed live owner, and no consumer reaches into
parallel socket/reason fields.

The registry has one active descriptor index and one address-stable slot pool.
The index contains only bound or closing descriptors. After close retirement
has waited for every operation reader, it extracts the descriptor-index node
into the same stable slot before that slot enters the intrusive free list. A
later bind changes the retained node key and reinserts it, so neither the slot
nor its hash node is allocated again. A stale borrowed view can read the slot
generation but can never dereference freed storage. The descriptor field is
lock-free atomic because a free slot may later represent another descriptor
while an already stale view is being rejected.

Every admission owns capacity before it can grow either the descriptor index
or slot pool. An active Session charges its declared
`net_socket_registry_capacity`. Admission outside a Session charges the single
default `SchedulerConfig{}.net_socket_registry_capacity` authority. Close
retains that charge through generation retirement, reader drain, native close,
and index erasure; only completed `FinishSocketClose` returns it. Let `L(t)`
be the concurrently bound slots and `H(t) = max(L(u), u <= t)`. Before any
generation is exhausted, retained slot count is exactly `S(t) = H(t)`. If a
slot reaches `UINT64_MAX`, that slot is burned, the pool stops growing, and its
capacity decreases permanently; generation never wraps. Thus `S(t) <= H(t)`
for the process lifetime, independent of close count and the number of
distinct retired descriptors. Also

```text
L(t) <= C_external + sum(C_session_owner_i)
S(t) <= max_u(C_external + sum(C_session_owner_i(u)))
```

Here `C_session_owner_i` includes a Session owner retained by any still-live
Socket after Session reset or destruction. Reset neither detaches those slots
nor zeroes their shared admission count; the last close releases the retained
owner. Thus the first line uses the same retained-owner set as the second line,
not merely currently running Sessions. Each `C` is an admitted envelope, not
an inferred OS limit. On a supported 64-bit ABI one stable slot is bounded by
128 bytes, including its retained index node handle, and one registry-owner
token is 24 bytes. The lock-free native descriptor, generation, reader count,
and close bit share one explicitly aligned private hot record of at most 32
bytes; its alignment does not depend on offsets inside a standard-library
container type. Each lock-free value is its direct `std::atomic` owner; there
is no plain storage plus `atomic_ref` adapter. These are checked layout bounds.
Hash-index nodes, buckets, and allocator bookkeeping are
implementation storage proportional to the concurrent high-water, not
retired-descriptor churn. Retaining one node handle adds eight slot bytes
on this ABI and removes one node allocation plus one node deallocation from
every close/re-admit cycle.

`Address` is the semantic tuple

```text
(family, canonical address bytes, host-order port)
```

IPv4 stores four meaningful octets; IPv6 stores sixteen. Address identity and
hashing never include native structure padding, platform family numbers, or
network-order storage. The canonical hash input has 19 bytes: one family byte,
sixteen canonical address bytes, and two port bytes. `sizeof(Address) == 20`
is a checked layout bound, not the identity encoding.

An invalid default address has `Family::None`, no byte span, and port zero.
Only `Address::ipv4`, `Address::ipv6`, and `Address::loopback` construct valid
public addresses.

## Lifecycle

`open` creates, registers, and transfers one `Socket` owner in one operation.
Accept transfers the peer owner the same way. Its typed
`OpenOptions` chooses only `Family`, `Transport`, and initial nonblocking
state. `bind`, `listen`, `local`, `shutdown`, and `nonblocking` borrow a
`SocketView` and validate its generation before reaching the platform
boundary.

All scheduler-network operations require a nonblocking socket. Blocking
sockets fail closed; runD does not hide a blocking syscall on a worker.
Accepted sockets pass through `accept::prepare` before application handoff.

Normal code does not author cleanup branches: owner destruction retires its
generation and performs the one native close attempt. `Socket::close()` exists
for explicit early release and consumes that owner even when the native close
reports failure. Retirement waits for already-acquired native-attempt leases,
so a destructor can wait for a concurrently completing native attempt and its
host event. A callback-bearing bounded helper releases that lease after event
completion and before arbitrary caller code; callback-driven close therefore
cannot wait on its own stack frame. Move-only ownership makes duplicate close
paths structurally unavailable. A stale view or ticket never becomes current
again when the operating system reuses its descriptor.
Recycling advances an active odd generation to an inactive even generation,
waits until the reader count is zero, and advances to the next odd generation
only on a later admission. The same slot and generation pair is therefore
unique until retirement. `UINT64_MAX` is absorbing and burns the slot instead
of admitting ABA through wraparound.

## Byte Operations

Inside a runD task, ordinary scalar I/O has one expression and one result:

```cpp fragment
ReceiveResult input = co_await receive(socket.view(), bytes);
SendResult output = co_await send(socket.view(), payload);
datagram::ReceiveResult packet =
    co_await datagram::receive(socket.view(), bytes);
datagram::SendResult sent =
    co_await datagram::send(socket.view(), payload, peer);
```

`receive` and `send` return the concrete public move-only operation types
`net::Receive` and `net::Send`; the datagram verbs return
`datagram::Receive` and `datagram::Send`. Awaiting them produces the matching
`ReceiveResult` or `SendResult`. Code that needs to retain an operation can
therefore name it without exposing the internal readiness composition:

```cpp fragment
net::Receive operation = net::receive(socket.view(), bytes);
net::ReceiveResult result = co_await std::move(operation);
```

The operations are dedicated awaitables, not nested `task::Task` values. Each
stores only the borrowed view, caller span, and any datagram peer. The public
operation owns no readiness or native-call policy; one internal prepared owner
validates the operation shape. An admitted non-empty operation creates its
single-socket readiness wait exactly once; construction before a runD task does
not capture a runtime. No internal readiness-composition type is the ordinary
verb's public return type or application storage vocabulary.
Resume yields exactly one move-only ticket to the existing scalar consumer.
The operation is itself move-only and single-await: moving transfers and
invalidates its view, and entering `co_await` consumes that view. A moved-from
or repeated operation fails as `IoFdInvalid` before native I/O.
Ordinary operations run the scalar shape owner before readiness. Invalid span,
datagram capacity, or datagram peer therefore fails immediately without
parking. A valid operation carries that admission to a private prepared
consumer, so it never repeats the shape predicate. Empty stream receive and
send complete through that consumer without a reactor wait or native transfer
call; they still validate the current socket and record the zero-byte host
observation. Empty datagrams remain real datagrams and therefore retain the
normal readiness and native-attempt path. The explicit Ticket terminal runs
the same shape owner after claiming the caller's capability, preserving its
claim-first one-shot contract.

The non-empty path takes one short generation lease to register readiness and
releases it before parking. After wake, the prepared consumer takes one short
generation lease across the native attempt. Merging those leases would make
`Socket::close()` wait on a parked operation and can deadlock the scheduler;
the close-invalidatable gap is therefore part of the lifetime contract, not a
duplicate lookup authority. Both leases address the stable slot carried by
`SocketView` directly in constant time; neither searches the descriptor index.
The prepared consumer is the sole native descriptor and host-id projection for
the syscall and its host event. The composition adds no child coroutine, task
admission, heap allocation, extra native attempt, result wrapper, or ordering
authority.

The POSIX syscall boundary has one buffer-shape predicate for stream,
positioned, and datagram calls: a zero-length span is valid regardless of its
data pointer, while a nonzero span requires a non-null pointer. Stream and
datagram adapters consume that same inline predicate; they do not carry local
copies whose empty-buffer semantics can drift. It adds no branch beyond the
predicate each native call already required.

Advanced code that already coordinates readiness obtains and consumes that
same capability explicitly:

```text
ready::read(socket.view())  -> Ticket&& -> receive / datagram::receive / batch::receive
ready::write(socket.view()) -> Ticket&& -> send    / datagram::send    / batch::send
```

`direct::receive` and `direct::send` are synchronous host operations for code
outside an active scheduler task. Calling them from a task is rejected. The
ordinary stream and datagram awaitables are the task-safe scalar path and
internally consume one matching `ready::Ticket&&`. Explicit ticket overloads
expose those same terminals for advanced coordination. A ticket is
move-only and single-use. Passing `Ticket&&` first claims and invalidates that capability,
including zero-budget, empty-buffer, or invalid operation-shape terminals.
Claim validates the ticket code and interest without acquiring the socket
generation lease; operation preflight then rejects invalid shape without a
registry lookup, lease, or native attempt. Only an admitted shape acquires the
ticket's direct entry/generation lease and produces one internal prepared
operation. That operation is the sole native descriptor and stable host-id
projection used by the syscall and its host event. Reuse, wrong interest, and
a generation retired before lease acquisition return typed failures without
attempting native I/O.

Every operation consumes caller-owned spans. Datagram results carry the
canonical peer address and its stable hash. Scatter/gather accepts a bounded
span of caller descriptors; runD does not assemble a temporary packet.

## Advanced Readiness

`ready::read` and `ready::write` return the scheduler's single-socket awaitable.
Timed waits live under `ready::timed`; many-socket waits and their budget live
under `ready::many`. Reusable membership lives in `ready::Set` and is managed
only by `ready::{create,destroy,clear,add,remove}`.

`ready::Set` is a 16-byte capability whose two public integers are opaque and
equality-only. Their exact values, numeric order, persistence, and use as a
replay branch are not contracts. A live capability belongs to its owning open
Session and may be reused across that Session's scopes. Process-wide slot ids
prevent a capability from another Scheduler or an earlier reset/open lifetime
from selecting a coincidentally numbered set in the active Scheduler; the
process issuer is not reset with a Session and is not part of trace, event,
ordering, or replay identity.
`ready/set/identity.hpp` is the sole storage-shape owner for that capability;
`ready/set.hpp` adds lifecycle and membership operations. A deferred
`ready::many::Wait` retains one complete `Set` value and never splits or
reconstructs its id and generation as parallel fields.

Create is the only operation that publishes a live incarnation. A new slot
starts at generation 1. Destroy returns a non-live tombstone at the next even
generation, and reuse publishes the following odd generation, so neither the
old live handle nor the destroy result authorizes the replacement. If the
maximum generation is retired, its numeric tuple remains a non-live tombstone
and the next create rekeys that storage with a fresh process slot id at
generation 1. Slot ids never emit zero or `UINT64_MAX` and never wrap; issuer
exhaustion fails a creation that needs a fresh slot id without publishing a
set. A reusable even-generation tombstone still succeeds without consulting
the exhausted issuer.

Many-readiness preserves request-index order. For equal observed readiness,
the output event order is the canonical input order rather than worker or
backend completion order. `Budget::max_events` bounds output work, and the
caller supplies both request and event storage.

A timed result distinguishes three states:

- `ready()` means an observation completed;
- `timed_out()` is a successful timeout terminal;
- any other non-OK reason is an error.

Cancellation and timeout cleanup remove their scheduler registration exactly
once. They do not leave a second polling or wake authority.

Cancellation-aware timed and many waits retain one complete internal
`StopIdentity` from defer through coroutine suspension. The scheduler validates
that value once at reactor admission and stores its complete scheduler-local
source projection; network awaitables do not mirror or reconstruct the four
identity components.

## Bounded Helpers

`accept::one` consumes one readable ticket and makes at most one native accept
attempt. `accept::drain` claims one readable ticket, validates interest, and
rejects a missing handler or zero budget before acquiring a generation lease.
Each admitted iteration reacquires that claim's generation, checks the current
listener's nonblocking state, performs one native attempt, and releases the
lease after accepted-result mapping and its host event but before the callback.
An admitted drain makes at most `accept::Budget::max_accepts` native accept
attempts. It stops on would-block, failure, budget exhaustion, or the callback;
the loop bound is the syscall bound. Scalar, server, and drain paths share that
single-attempt and accepted-result authority. The
caller owns peer-task slots for parallel serving; runD does not allocate a
hidden task vector. After configured scheduler and socket storage are warm,
drain and both server forms perform no runD heap allocation within their
declared bounds.

Sequential `server::serve` admits listener readability once per call and then performs
nonblocking accepts in canonical native order until the bound, an error, or
would-block. A connection that arrives while a peer handler is suspended joins
that same native backlog order; it does not create a second ordering authority.
Would-block after any completed handler is a successful short batch, while
`max_accepts` remains the exact upper bound. For `N` accepted peers the helper
therefore creates `N` handler tasks, not `N` additional accept-drain tasks, and
performs exactly `N` successful native accept attempts plus at most one final
would-block attempt.

Each server call owns one handler in its coordinator coroutine. Sequential
serving invokes it in order. Parallel serving lends a pointer to that same
owner to peer tasks and joins every task before destroying the handler, so a
move-only handler is not copied or allocated per peer. Parallel handlers may
run concurrently; mutable state captured by the handler must therefore provide
its own synchronization. Once handoff succeeds, the handler owns the peer
socket; its normal coroutine lifetime closes it automatically. The handler
returns
`Task<server::PeerResult>`: `server::PeerResult::complete()` records
completion, `server::PeerResult::stop()` requests a typed stop, and
`server::PeerResult::fail(code)` preserves an exact valid failure reason other
than the reserved `NetPeerHandlerStopped`; `Ok`, the reserved stop reason, and
invalid values normalize to `NetPeerHandlerFailed`. The three factories are
allocation-free and
store only the inherited reason. There is no generic outcome conversion:
operation-specific progress such as a partial frame or successful timeout must
be interpreted by the handler before it chooses the terminal. The server
closes only a prepared peer whose task spawn was rejected before the handler
could receive ownership.

The handler contract has one terminal classification owner:
`detail::classify_peer_terminal` maps a flattened `PeerResult` to exactly one
of completed, stopped, or failed. Sequential and parallel serving consume that
same classification; only their sequential update/early-return and atomic
update/lowest-index selection mechanics differ.

A failed peer coroutine reaches `flatten`, whose current compatibility behavior
preserves valid failure reasons other than `Ok` and
`NetPeerHandlerStopped`; those reserved reasons and invalid values normalize to
`NetPeerHandlerFailed` through `PeerResult::fail`. Whether a failed
`Task<PeerResult>` carrying `NetPeerHandlerStopped` should instead become a
stopped terminal is an unresolved public result/counter meaning decision. A
synchronously thrown handler invocation becomes `NetPeerHandlerFailed`, and an
explicit `PeerResult` preserves its already-normalized reason. Sequential
serving stops at that terminal. Parallel serving joins all admitted handlers,
then selects the non-success terminal with the lowest canonical accept index;
worker count, completion order, and stopped-versus-failed category cannot
change the selected peer reason. Admission/coordinator failure takes
precedence, followed by join failure, then the selected peer terminal.

`server::serve` returns the move-only `server::Task`. Awaiting it yields exactly
one `server::Result`:

```cpp fragment
const server::Result result = co_await server::serve(options, handler);
if (!result) {
  report(result.code(), result.error());
}
```

The scheduler-task terminal and server terminal are not nested public result
authorities. Failure to admit, run, or join the coordinator is mapped into the
same `server::Result` reason. The server coroutine, peer ordering, and
accounting path remain identical; `server::Task` only flattens the await
boundary and adds no coroutine, allocation, queue entry, or fallback path.

`drain::read` and `drain::write` share one `drain::Budget` and bound native
attempts by `max_operations`. Their distinct `ReadResult` and `WriteResult`
retain operation-specific progress without duplicating the budget authority.
Each iteration consumes the scalar byte path's single-attempt owner: it
reacquires the immutable claimed generation, performs the native call, and
releases the lease after event completion but before the callback. This keeps
the native attempt and event on one live host identity while excluding caller
code from its lifetime. A callback that returns false reports successful
accumulated progress with `handler_stopped`. If it returns true after closing
the source and more native work is required, the requested next iteration
fails its generation reacquisition with `IoFdInvalid`, makes no further native
attempt or event, and preserves the completed read/write/byte counts. EOF,
zero-progress, all-written, and the final budget attempt retain their existing
terminal rules because they request no next native attempt.

`frame::read` and `frame::write` carry a four-byte big-endian length followed
by bytes and reject frames above the declared limit. `flow` performs pure
bounded byte accounting. None of these helpers creates a packet schema, queue,
retry rule, or session.

Frame write claims one operation, validates its immutable header/payload shape
once, and retains one stack `WritePlan` plus one reusable `WriteBatch`.
Each native retry projects only the remaining header and payload into at most
two active slices. For `W` native attempts, configured native descriptor
capacity `C`, and active slice count `K <= 2`, descriptor initialization and
projection cost is `Theta(C + W*K)`: the bounded native array is initialized
once, and inactive entries are never cleared or examined between retries.
The completed-prefix payload hash consumes those same active slices after the
native call. No retry repeats shape validation, allocates storage, gathers
payload bytes, or changes the four-byte header order.

Drain consumes only a matching `Ticket&&`. A coroutine obtains it with
`co_await ready::{read,write}`; synchronous root-scope code may explicitly
call `.wait()` on that same readiness operation. There is no `SocketView`
overload that can hide blocking or accidentally select a task-invalid path.
The ticket claim and bounded drain loop are the only execution authority.

## Options And Limits

`option::Name` is the complete supported option set. Arbitrary integer option
levels and raw platform option buffers are not public. `option::Value` carries
only the boolean or byte-count meaning required by the selected option.

`limits()` returns configured network bounds and current ready-set usage.
Configured storage is the complete pool, and capacity exhaustion returns its
typed failure.

## Determinism And Replay

Network observations record the admitted socket identity, ordered operation,
native result, completed byte count, canonical peer identity where relevant,
and payload identity. Replay compares those observations in order. A replay
socket is an evidence carrier and never exposes or reconstructs a native
descriptor.

For a browser or remote client, this byte boundary ends before application
decode and canonicalization. The server-native execution authority is owned by
[Replay's server-native boundary](./replay.md#server-native-boundary); Network
does not carry a Compute backend selector or select a client-side device.

For vectored input with `K` admitted slices, `J` completed-prefix slices, and
`P` completed bytes:

```text
0 <= J <= K
descriptor work without raw capture = K + J
descriptor work with raw capture    = K + J
payload identity work               = Theta(P)
raw-capture copy work               = Theta(P)
```

Admission validates the `K` descriptors once. Hashing walks only the `J`
descriptors covering the completed prefix. One ingress owner selects hash-only
or retained capture before entering that walk. Hash-only performs the existing
single ordered hash pass. Retained capture visits each of the same `J` slices
once, copies it into the bounded ring, then updates StableHash from the exact
ring destination span in canonical wrap order. It has no preflight payload
traversal, gather buffer, or second source projection. The source is read
exactly `P` bytes, the ring is written `P` bytes, and StableHash reads those
`P` hot ring bytes. These are distinct `Theta(P)` byte operations: exact
identity must observe all `P` bytes and retention must write all `P` bytes.
The executable traversal contracts and the three-run retention-order
measurement are recorded in
[`Host Raw Byte Hash Evidence`](../../../docs/reference/host/hash.md).

The reactor owns one ordering and registration authority. Platform pollers,
network wrappers, and replay do not independently reorder ready events.

Raw host replay compares the recorded observation path; it does not invent a
causal edge between an external producer and readiness admission. If both ends
are owned by one runD schedule, the consumer admission must happen-before the
producer operation:

```text
admit(read readiness) < send(payload)
```

Without that edge, both `immediate read` and `park -> wake -> read` are valid
kernel traces, so their timer/reactor observations can differ even when the
payload is identical. The datagram replay contract therefore registers the
read before its spawned sender can run and awaits that sender asynchronously.
Uncontrolled remote arrival belongs at the canonical replay input boundary;
simulation replay consumes that ordered input instead of treating live packet
timing as deterministic state.

## Performance Freeze

The release surface fixes these hot-path properties:

- byte, datagram, and scatter/gather calls allocate no runD heap storage;
- ordinary scalar stream and datagram SocketView operations add no coroutine
  frame or task admission; their dedicated awaitables perform one readiness
  decision and consume its one ticket through the explicit prepared terminal;
  an immediately ready operation parks no reactor registration and a suspended
  operation parks exactly one;
- ordinary scalar shape rejection occurs before readiness and parks zero
  registrations; admitted direct operations do not repeat that shape check at
  the prepared consumer;
- readiness operations borrow caller buffers and configured scheduler storage;
- ready-set and accept work is bounded before observation;
- sequential serving performs one listener-readiness admission per batch;
- readiness registration resolves a view to its stable registry slot and
  validates the admitted generation;
- single-socket coroutine readiness does not project native descriptor or host
  id while constructing its deferred wait. The suspension boundary acquires
  the sole registration lease and derives both values from that lease;
- many-readiness validates request shape once and lets the registration lease
  perform the sole current-generation check on its normal execution path;
  runtime-missing and synchronous-misuse diagnostics check generation only on
  those rejected paths to preserve exact error precedence;
- ticket consumption performs no registry map search or descriptor-to-entry
  resolution; one `claim(ticket, interest)` transition is the sole consumed,
  status, and interest-validation owner, followed by one O(1) generation lease
  only after operation-specific shape admission; `prepare(claim)` produces the
  sole internal operation, and its captured descriptor derives both native
  work and host-event identity without rereading the socket slot;
- many-readiness acquires one bounded lease array and constructs its reactor
  request batch from those captured descriptors. It does not reload every
  slot's native descriptor after the lease pass. Park registration derives the
  host id arithmetically from the prepared descriptor rather than projecting
  the `SocketView` again;
- descriptor lookup, stable-slot allocation, and index growth occur only at
  admission; after a concurrent high-water is warm, close/re-admit churn uses
  one expected-`O(1)` index operation and the intrusive free list without slot
  allocation;
- scalar byte, datagram, scatter/gather, `accept::one`, and connect-completion
  consumers then make at most one native attempt;
- explicit read, write, and accept drains and server batches make at most their
  declared budget of native attempts and stop on would-block, error, callback
  stop, or the bound; every drain iteration uses one short generation lease
  that owns its host-event identity and ends before arbitrary caller code;
- generation is checked at registration and again at consumption because a
  close/reuse transition may occur between them;
- completed payload bytes are hashed in one ordered pass; ingress hashing and
  optional bounded retention share one canonical slice projection, while the
  disabled path selects hash-only before entering the payload loop;
- one 16-byte `(entry, generation)` `SocketView` is the complete borrowed
  carrier; host id is derived and never copied as a mirror;
- typed public operations call the owning implementation directly.

runD does not scan packet bytes for delimiters, message kinds, or schemas on
this path. Those protocol decisions belong above the domain-neutral byte
boundary. A scalar operation presents the caller span directly to one native
attempt. A vectored operation walks its bounded descriptors once before the
lease, simultaneously validating lengths, summing admitted bytes, and writing
the fixed native descriptor batch. The platform passes that prepared batch to
the syscall without a second caller-descriptor traversal. That prepared batch
is the sole admitted-byte count authority and remains alive through the native
call and host-event projection. The platform returns only the native call
outcome; it does not mirror the admitted byte count in a result wrapper. After
completion, the exact completed payload prefix is hashed in canonical slice
order; this is a byte-identity pass, not a protocol search or gather copy.

An admitted vectored operation whose aggregate byte count is zero still
consumes its readiness ticket, validates the current socket generation, and
records one zero-byte host event. The common operation owner then completes it
without a native syscall. This gives the same terminal meaning on kernels that
otherwise disagree on whether a zero-length `recvmsg` observes readiness.

For `N` admitted slices, pre-syscall descriptor work is `N` validation loads
and `N` native-descriptor stores. The platform consumes that prepared batch
directly, so pre-syscall metadata projection is exactly one `Theta(N)` pass
with no second caller-descriptor traversal. Storage remains the configured
fixed `net_iov_capacity`.

Each parallel peer handler that reaches an outcome performs exactly one
lock-free relaxed atomic counter RMW directly on its public `Result` field.
The supported GCC and Clang platform compilers provide that operation without
requiring the C++20 library `atomic_ref` adapter. Counter initialization occurs
before handler dispatch and non-atomic reconciliation occurs only after join,
so no atomic and non-atomic access overlaps. A non-success terminal
additionally publishes one packed candidate through an atomic-min CAS loop:

```text
p(i, r) = (uint64(i) << 16) | uint16(r)
0 <= i <= 2^32 - 2, 0 <= r <= 2^16 - 1
p(i, r) < 2^48 < UINT64_MAX
```

`i` is the canonical accept index and `UINT64_MAX` is the no-terminal
sentinel. Numeric minimum therefore selects the lowest accept index before
reason bits are considered. Atomic minimum is associative, commutative, and
idempotent, so the final value is independent of worker interleaving and
completion order. The release CAS and post-join acquire load publish that one
terminal authority without a per-peer outcome array.

The fixed caller-provided `span<task::Handle>` contains no separate outcome
storage. A scheduler-level missing completion is reconciled once after join
without an atomic operation. If no exact terminal was published,
reconciliation fails closed with `NetPeerHandlerFailed`; an OK result can never
coexist with a missing handler terminal. There is no second counter or mirrored
completion owner. For `N` handlers and `F` non-success terminals, accounting
performs exactly `N` counter RMW operations and `F` selector loads. If `A` is
the observed number of selector CAS attempts, its work is `Theta(N + F + A)`;
at most `F` CAS operations can successfully lower the selector. The shared
counters require at least one linearized write per outcome, meeting the
`Omega(N)` accounting lower bound; successful peers pay no selector load or
CAS.

These are structural bounds, not throughput claims. Throughput and latency
claims require measured benchmark evidence for the target platform.

For one successful `ready::{read,write}` followed by its matching scalar byte
operation, execution performs exactly two atomic native-slot loads: one in
each of the two generation leases. Generation is checked before and after each
reader acquisition, so no additional descriptor projection owns independent
validation. This is an exact operation count, not a wall-time ratio.

For `N` many-readiness requests, the admitted batch performs `N` native-slot
loads inside its `N` required generation leases and zero descriptor reloads
while constructing or parking the reactor batch. This meets the `Omega(N)`
validation lower bound with one descriptor projection per request. Storage
remains the configured lease and request scratch; steady-state admission adds
no allocation.

Payload identity still requires `Omega(P)` byte reads for `P` completed bytes;
no exact hash can distinguish every changed input byte without observing it.
runD uses the checked XXH3 owner for that one pass and performs no delimiter,
message-kind, or schema search. Optional raw diagnostic retention adds
`Omega(P)` destination writes because it owns persisted bytes rather than
identity, but it does not add a second source or slice walk. Disabled capture
selects hash-only before event commit and keeps its `Theta(P)` work outside the
scheduler evidence mutex. Enabled capture serializes its required ring mutation
and destination hash with host-event commit; this optional diagnostic path
therefore lengthens that critical section by `Theta(P)`. The measured ordering
evidence remains owned by
[`Host Raw Byte Hash Evidence`](../../../docs/reference/host/hash.md).

## Results And Telemetry

Every network result derives its status views from the one private
`rund::ReasonCode` owned by `net::Status`. Consumers read it only through
`code()` and cannot mutate it. `ok()`, truth conversion, `error()`, and
`exit_code()` derive from that same value. Byte
progress, timeout, would-block, budget exhaustion, and completion remain
explicit fields rather than a second success authority. Timed results interpret
`IoTimedOut` as a successful terminal while still exposing that same reason
through `code()`. The shared projections are owned beside public
`rund::ReasonCode`; installed network headers contain no Node-host result
adapter or duplicate status mapping.

`Session::Result` exposes public network counters through
`result.tasks().network()`: calls, bytes, lifecycle observations, would-block
outcomes, and admission rejection. `result.tasks().reactor()` owns readiness
registration, wake, cancellation, and capacity counters. Platform counters
remain implementation diagnostics and do not mirror the public totals.
The source-private scheduler host-event recorder is the single
`EventKind -> call/lifecycle slot` owner. Every additive Network field consumes
the repository `counter::Accumulate` law, so `UINT64_MAX` is absorbing for call,
lifecycle, would-block, admission-rejection, and byte evidence; no field wraps
to zero. Successful stream, datagram, and vectored events alone add their
completed bytes, and lifecycle events never add bytes.
Despite its category and spelling, the current compatibility meaning of
`would_block()` counts every admitted host event with `Status::WouldBlock`,
including non-network host I/O. Restricting that field to Network event kinds
requires an explicit public telemetry decision; the recorder and contract test
preserve the existing behavior until then.

Ticket rejection is observable through its typed result, not counted as a
native call. In the focused lifecycle contract, one accepted send and one
accepted receive increase `send_calls` and `recv_calls` by exactly one each;
wrong-interest, already-consumed, and retired-generation attempts increase
both native-call counters by zero. The warm loop performs 1,024 successful
sends and 1,024 repeated-consume rejections with zero runD heap allocations;
each successful byte is drained before the next send, so the proof has
constant kernel-queue occupancy on every host. The peer receives exactly 1,024
bytes and the next read would block. These deltas prove native-call
suppression and allocation behavior without importing an OS socket-buffer
capacity; generation validation remains mandatory.

## Verification

The focused product cases are:

```text
runtime.task.net
runtime.task.net-drain
runtime.task.net-nonblocking
runtime.task.net-readiness
runtime.task.net-ready-many
runtime.task.net-ready-set
runtime.task.net-datagram
runtime.task.net-options
runtime.task.net-vectored
runtime.task.net-accept-connect
runtime.task.net-accept-drain
runtime.task.net-write-drain
runtime.task.net-accept-handoff
runtime.task.net-server
runtime.task.net-server-parallel
runtime.task.net-lifecycle
runtime.task.net-listener-lifecycle
runtime.task.net-limits
runtime.task.net-timed-readiness
runtime.task.net-cancellation
```

The installed consumer independently compiles `<rund/net.hpp>` and proves that
the `<rund/rund.hpp>` composition reaches the same surface without source-tree
includes or native socket admission.

## Update Rule

A network surface change updates this page, the matching header, its focused
contract, the installed consumer, and any affected public site guide atomically.
Each public concept has one current namespace, path, name, and executable
example.
