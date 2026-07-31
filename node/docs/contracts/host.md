# Host Contract

## Scope

This page owns deterministic host services, evidence, and hashing under
`rund::host`.

This page and its routing links are the live docs authority for implemented
host surfaces. Every public, implementation, and verification owner must have
an existing repository path before this page can claim it.

Host APIs cover:

- deterministic random streams
- logical clock and thread-like scheduler facades
- typed admitted native fd handles
- low-level IO readiness and syscall wrappers
- byte-level admitted native socket send/recv wrappers routed by
  [Network](./net.md)
- environment input capture
- host event capture for later replay evidence

Host APIs do not cover:

- kernel packet order, partitioning, fold order, or workspace layout
- protocol-level networking, DNS, TLS, buffering, or retry policy owned above
  [Network](./net.md)
- arbitrary libc or std calls made outside admitted `rund::host` or
  `rund::net` APIs
- raw blocking syscalls inside generated scheduler lanes
- parallel root-level aliases for focused host services
- hardware cache discovery, CPU feature profiles, kernel capability types, or
  kernel tile planning authority

Raw network entrypoints are approved only in
`/node/src/runtime/platform/posix/net.cpp`, the feature-specific
datagram owner
`/node/src/runtime/platform/posix/net/datagram.cpp`, and the selected
socket-option owner
`/node/src/runtime/platform/posix/net/options.cpp`, the source-private
no-signal send policy
`/node/src/runtime/platform/posix/net/socket/call.hpp`, and the vectored IO
owner `/node/src/runtime/platform/posix/net/vectored.cpp`;
scheduler code routes native socket work through those platform owners instead
of calling raw network entrypoints directly. Raw IO entrypoints (`read`, `write`, `open`,
`close`, `fcntl`, and `fstat`) are approved only in
`/node/src/runtime/platform/posix/io.cpp`; scheduler code routes
native fd probing, fd identity lookup, and IO work through that platform
owner. Environment
reads through `std::getenv` are approved only in
`/node/src/host/env.cpp`. The
scheduler timer live wait implementation in
`/node/src/runtime/task/scheduler/timer.cpp` has a targeted exception only for
`std::chrono::steady_clock::now` and the scheduler-local `Clock::now` alias.
Test-only socket setup calls under `/node/tests` are outside this host runtime
contract. Host behavior is proved by runtime host, timer, host-IO, replay, and
network contracts. Source-internal boundaries are compiled by subsystem
contracts; the installed product projection is compiled through the package
consumer.
Internal scheduler and net helpers such as reactor wait bridges, cleanup
owners, socket-registry accessors, scheduler host-event record methods, and
host API event-record bridges must not appear in `/node/include/node` unless
the owning contract promotes them to the documented subsystem API.
Host and network source files may use source-owned scheduler access helpers
for active scheduler lookup, identity, limits, timed-ready completion, and
stop-token identity. Those helpers are not exported
through `Handle`, and public headers must not expose native reactor, socket
registry, stop-token identity, active scheduler, or coroutine completion
bridges as public `Handle` methods.
Host/network contracts verify the socket registry split, stack-memory platform
owner, and source-private host replay layout. Their evidence is the
grouped `node.*` run with ordered in-process case dispatch and the final
verification record in
[`/docs/architecture/verification.md`](../../../docs/architecture/verification.md);
that evidence is host boundary evidence only and does not add protocol,
session, retry, queue, rate-limit, or gameplay meaning.

## SDK Projection And Source Owners

SDK consumers reach selected host names through `<rund/host.hpp>` or the
`<rund/rund.hpp>` composition and link `runD::sdk`. The following paths own the
source implementation API and transitive support; they are not additional
direct-include entries:

- `/node/include/rund/host/hash.hpp`
- `/node/include/rund/host/status.hpp`
- `/node/include/rund/host/event.hpp`
- `/node/include/rund/host/random.hpp`
- `/node/include/rund/host/random/seed.hpp`
- `/node/include/rund/host/chrono.hpp`
- `/node/include/rund/host/timer.hpp`
- `/node/include/rund/host/io.hpp`
- `/node/include/rund/host/io/fd.hpp`
- `/node/include/rund/host/env.hpp`
- `/node/include/node/runtime/replay/host.hpp`

Host-IO operation results inherit one `io::Status`, whose `ReasonCode` is
private. Consumers observe it only through `code()` and cannot mutate it or
construct a result from a reason code. `ReasonCode::Ok` is the only completion
authority: `ok()` and explicit truth conversion derive from that equality,
`error()` is empty for `Ok` and otherwise returns `ReasonString(code())`, and
`exit_code()` is `0` or `1` from the same projection. Native OS errors use
`native_error`; byte progress and the opened descriptor remain payload, not a
second completion authority. Public default construction is fail-closed;
source-private `io::detail::Access` is the only writer of the private code and
maps an attempted `Ok` failure or negative success byte count to
`TaskInvalid`. There is no public status setter, stored success bit, reason
view, or second truth field.

`ReadResult`, `WriteResult`, and `CloseResult` remain trivially copyable
fixed-size values. On the supported 64-bit ABI their sizes are 24, 24, and 8
bytes respectively. `OpenResult` is a 32-byte move-only value because it owns
its returned `Fd`; copying an open result cannot duplicate close authority.
Status construction performs no allocation and adds no pointer or indirection
to blocking, task completion, or replay paths.

## Public API Meanings

| API | Identity | Determinism boundary |
| --- | --- | --- |
| `random::u64` | stream seed, stream id, draw id | Counter algorithm, no OS entropy |
| `random::below` | stream seed, stream id, draw id, upper bound | Rejection sampling, exclusive upper bound |
| `chrono::logical_clock::now` | scheduler logical time | No wall-clock value exposure |
| `timer::at` | task id, logical deadline, observed due event | Live wake timing is not semantic |
| `io::readable` / `io::writable` | admitted host fd id, native readiness observation | Only an lvalue owner's typed `io::FdView` enters the public readiness surface |
| `io::read_some` | host fd id, sequence, byte count, payload hash | Replay checks recorded event |
| `net::receive` | admitted socket id, byte count, payload hash | Network bytes only, no protocol meaning |
| `net::send` | admitted socket id, byte count, payload hash | Network bytes only, no delivery guarantee |
| `net::open` | admitted socket id, typed family and transport result | Socket carrier only, no session or protocol meaning |
| `net::bind` | admitted socket id, canonical address hash | Address only, no routing or protocol meaning |
| `net::listen` | admitted socket id, backlog value | Listener readiness only, no queue policy above native backlog |
| `net::local` | admitted socket id, canonical address hash | Local address only, no DNS or URL meaning |
| `net::shutdown` | admitted socket id, shutdown mode | Half-close carrier only, no reconnect or session policy |
| `net::datagram::receive` | admitted socket id, byte count, peer address hash, payload hash | One datagram observation only, no delivery or protocol meaning |
| `net::datagram::send` | admitted socket id, byte count, peer address hash, payload hash | One datagram attempt only, no retry, queue, ordering, or delivery guarantee |
| `net::option::set` | admitted socket id, selected option, normalized value | Selected option only, no arbitrary platform policy |
| `net::option::get` | admitted socket id, selected option, observed value | Selected option observation only, no protocol or session meaning |
| `net::batch::receive` | admitted socket id, byte count, completed-prefix payload hash | Caller-owned mutable slices only, no hidden receive queue or retry |
| `net::batch::send` | admitted socket id, byte count, completed-prefix payload hash | Caller-owned immutable slices only, no hidden send queue or retry |
| `env::get` | env name hash, value hash | Returns `task::Result<std::string>`; environment is external input |
| `replay::Channel::read(context)` | nonzero input id, nonzero schema id, source-returned or transcript-resolved sequence, exact ordered opaque payload | Canonical domain-free input; Live and Record call the bound Writer source once, while Replay and Scenario call it zero times |
| `host::hash_bytes` | byte count and ordered byte contents | Stable byte hash; structured owners encode explicit canonical framing |

The network host/reactor contract is owned by [Network](./net.md). That page
owns byte-level and readiness-level network behavior, the runtime/domain semantic
cut, would-block observations, and network replay evidence.

Hardware cache observation is not a Host product surface. CPU feature
detection and executable SIMD selection are private to the CPU accelerator in
`/node/src/accel/cpu/strategy.cpp`; Host public headers therefore expose no
Kernel type and runtime-only products do not link an accelerator capability
observer.

`/node/src/host/replay/layout.cpp` is the single source-private owner for host
event names and accepted replay event kinds.
The SDK exposes only the event value through `EventKind`; it does not expose a
schema object, field-mask API, iterable schema list, or count sentinel. The
private layout uses the last concrete event value and a compile-time table-size
check as its numeric bound. Replay decode rejects `EventKind::None`, every
unknown numeric kind, and noncanonical field presence before accepting the
replay hash. `/node/src/host/event.cpp` and
`/node/src/runtime/replay/host/codec.cpp` consume that same private owner.

Host persistence uses the shared binary artifact reader and writer. One
unsigned varint owns event count, followed by one fixed 64-bit event hash.
Each event stores sequence and logical-time deltas, kind, status, and one
12-bit presence bitmap; zero-valued optional fields consume no value bytes.
The decoder borrows the input span, admits the event count before allocating
the final event vector, fills that vector in one pass, and validates the
recomputed event hash before publication. It owns no per-field descriptor
table or intermediate encoded-event buffer.

Encoded chunk verification has one private owner in
`runtime/replay/host/payload/codec.cpp`. Raw chunks hash their immutable span
directly. RLE chunks walk encoded tokens once and feed the canonical decoded
byte order into the chunk, record, and optional joined-payload hash states.
An RLE repeat tag expands at most 130 bytes, so verification uses one fixed
130-byte stack window rather than a decoded vector. For `E` encoded and `P`
decoded bytes, verification is `Theta(E + P)` work and `Theta(1)` auxiliary
storage; it performs zero dynamic allocations and zero payload-size copies.

## Implementation Authority

- `/node/src/host`
- `/node/src/host/replay/layout.hpp`
- `/node/src/host/replay/layout.cpp`
- `/node/src/runtime/replay/surface/`
- `/node/src/runtime/replay/input/plan.hpp`
- `/node/src/runtime/task/scheduler/core/host.cpp`
- `/node/src/runtime/task/scheduler/core/replay.cpp`
- `/node/src/runtime/task/scheduler/core/time.cpp`
- `/node/src/runtime/task/scheduler/timer.cpp`
- `/node/src/runtime/task/scheduler/io.cpp`
- `/node/src/runtime/task/io/bridge.cpp`
- `/node/src/host/io/validation.hpp`
- `/node/src/host/io/access.hpp`
- `/node/src/runtime/task/scheduler/host/io/`
- `/node/src/runtime/platform/posix/io.cpp`
- `/node/src/runtime/platform/io.hpp`
- `/node/src/runtime/platform/net.hpp`
- `/node/src/runtime/platform/unavailable/io.cpp`
- `/node/src/runtime/platform/unavailable/net.cpp`
- `/node/src/runtime/platform/posix/address.cpp`
- `/node/src/host/net/registry/access.hpp`
- `/node/src/host/net/registry/state.hpp`
- `/node/src/host/net/registry/state.cpp`
- `/node/src/host/net/registry/admission.cpp`
- `/node/src/host/net/registry/validation.cpp`
- `/node/src/host/net/registry/lifecycle.cpp`
- `/node/src/runtime/platform/posix/net.cpp`
- `/node/src/runtime/platform/posix/net/socket/call.hpp`
- `/node/src/runtime/platform/posix/net/datagram.cpp`
- `/node/src/runtime/platform/posix/net/options.cpp`
- `/node/src/runtime/platform/posix/net/vectored.cpp`
- `/node/src/runtime/replay/artifact/format.hpp`
- `/node/src/runtime/replay/host/codec.hpp`
- `/node/src/runtime/replay/host/codec.cpp`
- `/node/src/runtime/replay/host/diff.cpp`

Replay host source map:

| Path | Owns |
| --- | --- |
| `/node/include/node/runtime/replay/host/bytes.hpp` | Source-private immutable `payload::Bytes` value shared by archive chunks and the memory backend. A producer freezes one completed byte vector; internal consumers receive read-only spans and pointers. |
| `/node/include/node/runtime/replay/host/archive.hpp` | Source-private `payload::Archive` record, piece, chunk, and snapshot metadata. No archive type is exposed by a direct SDK header. |
| `/node/src/runtime/replay/input/plan.hpp` | Immutable canonical Scenario input patches indexed by source, schema, and sequence. |
| `/node/src/runtime/replay/scope/plan.hpp` | One source-private Live, Record, Replay, or Scenario execution plan and its prepared expected-evidence owner. |
| `/node/src/runtime/replay/scope/session.hpp` | Session admission, evidence preparation, capacity query, and scope-generation lease. |
| `/node/src/runtime/replay/history.cpp` | Fixed-slot multi-segment retention, deterministic oldest eviction, checkpoint-chain continuation, and retention telemetry. |
| `/node/src/runtime/replay/artifact/format.hpp` | One bounded binary writer and one borrowed, canonical binary reader shared by every Replay artifact kind. |
| `/node/src/runtime/replay/host/codec.hpp` | Shared embedded host-evidence codec boundary. |
| `/node/src/runtime/replay/host/codec.cpp` | Standalone and embedded host binary encoding, count admission, delta/presence decoding, and event-hash validation. |
| `/node/src/runtime/replay/host/diff.cpp` | Host replay event equality, first mismatch, field diff, and context windows. |
| `/node/src/runtime/replay/host/payload/backend.hpp` | Private `Blob`, memory/spill backend, immutable `SpillGeneration` owner, read/append result, and storage report routing. |
| `/node/src/runtime/replay/host/payload/backend/memory.cpp` | In-memory payload blob storage and backend facade routing. |
| `/node/src/runtime/replay/host/payload/backend/spill/generation.cpp` | Unique generation creation, lease-based stale-generation scavenging, immutable lifetime, and Budget reserve/commit/refund. |
| `/node/src/runtime/replay/host/payload/backend/spill/segment.cpp` | One filesystem snapshot, exact allocation and headroom admission, fixed 41-byte little-endian header, and direct positioned segment I/O. |
| `/node/src/runtime/replay/host/payload/backend/spill/store.cpp` | Transactional Spill append/rollback, lazy reads, decoded-cache routing, reports, archive-load cursor reconstruction, and clear. |
| `/node/src/runtime/replay/host/payload/cache.cpp` | Bounded spill-read chunk cache. |
| `/node/src/runtime/replay/host/payload/chunk.hpp` | Fixed 64 KiB chunk bound shared by live storage and archive validation. |
| `/node/src/runtime/replay/host/payload/codec.cpp` | Private raw/RLE payload encoding and allocation-free encoded-chunk verification. |
| `/node/src/runtime/replay/host/payload/store.hpp` | Private `replay_detail::payload::Store` state and the archive-backed `Build` ownership boundary. |
| `/node/src/runtime/replay/host/payload/store.cpp` | Host replay payload append, chunk dedupe, storage counters, and clear. |
| `/node/src/runtime/replay/host/payload/store/archive.cpp` | Host replay payload archive export and archive-backed store construction. |
| `/node/src/runtime/replay/host/payload/hash.hpp` | Canonical payload byte, record, and archive hash framing over the shared stable hash state. |
| `/node/src/runtime/replay/host/payload/store/hash.cpp` | Host replay payload logical hash calculation over event bindings and uncompressed bytes. |
| `/node/src/runtime/replay/host/payload/store/load.cpp` | Host replay payload archive load orchestration into the store backend. |
| `/node/src/runtime/replay/host/payload/store/load/chunk.cpp` | Host replay payload archive chunk id, encoded byte, spill mode, segment, and blob/ref validation. |
| `/node/src/runtime/replay/host/payload/store/load/record.cpp` | Host replay payload archive record piece bounds, piece-byte accounting, and logical-byte validation. |
| `/node/src/runtime/replay/host/payload/store/resolve.cpp` | Direct indexed replay read, write matching, and materialization resolution over checked payload pieces. |
| `/node/src/runtime/replay/host/payload/materialize.hpp` | Source-private materialized payload values and `Materialize`, `Build`, `Equal`, and `MakeArchive` test helpers. |
| `/node/src/runtime/replay/host/payload/materialize.cpp` | Materialized payload ownership and store conversion without a parallel record-hash implementation. |
| `/node/src/runtime/replay/host/payload/validate.cpp` | Direct archive shape, binding, and canonical hash validation. |

## Verification Authority

- `/node/tests/contract/runtime/task/host.cpp`
- `/node/tests/contract/runtime/task/host/timer.cpp`
- `/node/tests/contract/runtime/task/random.cpp`
- `/node/tests/contract/runtime/task/host/io/`
- `/node/tests/contract/runtime/task/replay/`
- `/node/tests/contract/runtime/task/replay/run.cpp`
- `/node/tests/contract/runtime/task/net.cpp`
- `/node/tests/contract/runtime/task/net/registry/lifetime.cpp`
- `/node/tests/contract/runtime/task/net/datagram.cpp`
- `/node/tests/contract/runtime/task/net/options.cpp`
- `/node/tests/contract/runtime/task/net/vectored.cpp`
- `/node/tests/contract/runtime/task/net/limits.cpp`
- `/node/tests/contract/runtime/task/env.cpp`

## Invariants

Node is the local host authority. Host APIs expose std-like names only when the
runD meaning is explicit and deterministic.

Random values are derived from `(stream.seed, stream.id, draw_id)` through a
fixed-width counter algorithm. Worker count, OS thread pickup order, and task
completion order must not change the value returned for the same tuple.
`rund::host::random::Stream` is a public deterministic carrier, not an OS
entropy handle.

`random::stream(StreamId)` uses the active runtime scope's admitted
`random::RunSeed` with the caller-provided stream id. Outside an active
scheduler scope, the default run seed remains `0xC2B2AE3D27D4EB4F`.
`random::stream(RunSeed, StreamId)` exposes the explicit seed form for callers
that need a stable stream independent of the active scheduler scope.
`random::split(parent, child)` computes a child seed as:

`splitmix64(parent.seed ^ (parent.id * 0x9E3779B97F4A7C15))`

and stores `child.value` as the child stream id. `random::u64(stream, draw)`
computes:

```text
splitmix64(stream.seed ^
           (stream.id * 0x9E3779B97F4A7C15) ^
           (draw.value * 0xD1B54A32D192ED03))
```

where all arithmetic is unsigned 64-bit arithmetic. `splitmix64` is:

1. add `0x9E3779B97F4A7C15`
2. `value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9`
3. `value = (value ^ (value >> 27)) * 0x94D049BB133111EB`
4. return `value ^ (value >> 31)`

`random::u32` returns the high 32 bits of `u64`. `random::unit_f32` takes the
high 24 bits of that `u32` and maps them to `[0, 1)` by multiplying by
`1 / 16777216`. `random::fill_bytes` writes consecutive `u64` draw blocks
starting at `first_block`, low byte first within each block; `first_block`
addition wraps as unsigned 64-bit arithmetic. `random::below(stream, draw, 0)`
returns `0`; otherwise it uses rejection sampling for an unbiased value below
the exclusive upper bound. `random::range(stream, draw, lower, upper)` returns
`lower` when `upper <= lower`; otherwise it returns `lower + below(...)`.

The public clock is logical. Wall-clock reads may be used by the scheduler to
wait for a live timer, but wall-clock values are not public deterministic
values. `chrono::time_point + chrono::nanoseconds` saturates to the signed
64-bit time-point minimum or maximum on overflow. `time_point - time_point`
returns a `chrono::nanoseconds` duration and saturates to the signed 64-bit
duration minimum or maximum when the mathematical difference is outside that
range. Duration waits use the canonical `rund::task::sleep(duration)` task
operation. `timer::at(deadline)` is the sole deadline projection and computes
`max(deadline - logical_clock::now(), 0)` before that same sleep operation.
Live wake timing remains observed host behavior, not portable semantic
authority.

IO enters deterministic work through admitted handles. Native fd numbers are
process-local carriers, not durable replay identity. `host_id` is an
admitted carrier derived from the current native fd, so native fd reuse can
reuse the same `host_id`; it is not replay identity by itself. Later replay
evidence must combine admitted host handle id with operation kind, sequence,
result status, errno, byte count, offset, path hash, and payload bytes or
payload hash as specified by the event kind.
`rund::host::io::take_native_fd(fd)` transfers one non-negative native fd into
the move-only `Fd` owner, replaces the source integer with `-1`, and assigns
`host_id = uint64(fd) + 1`. An invalid native fd produces an empty owner. There
is no by-value admission fallback, so a successful transfer cannot leave a
second raw close authority in the passed variable. `Fd` is the sole close
authority and its destructor closes a live native descriptor exactly once.
`Fd::close()` provides explicit early release, invalidates the owner before the
platform close, and a second close reports `io_fd_invalid`.

`Fd::view()` is available only on an lvalue owner. It returns a 16-byte,
trivially copyable `FdView` that borrows operation identity without acquiring
close authority; `Fd{}.view()` is deleted. Readiness and byte operations accept
only `FdView`, so operation tokens cannot copy an owner. The source-private
scheduler bridge unwraps a view only after
`native >= 0 && host_id == uint64(native) + 1` holds. An empty or invalid view
therefore fails as `io_fd_invalid` before scheduler lookup.
The owner must outlive every borrowed view and the operation consuming it;
normal code therefore keeps the `Fd` in the scope that joins those operations.

Direct `read_some_blocking`, `write_some_blocking`, `pread_some`, and
`open_file` calls fail closed with `io_fd_invalid` for invalid handles,
`task_invalid` for empty paths or invalid non-empty buffers, and
`io_syscall_failed` with copied `errno` for syscall failures. The blocking read,
write, and positioned-read calls from an active scheduler task fail closed with
`task_invalid`. `Fd::close()` is valid in a task and synchronously invalidates
matching reactor state before the one native close attempt.

`rund::host::io::read_some` and `write_some` return move-only `ReadOp` and
`WriteOp` await tokens. A `Task<T>` must consume them with `co_await`;
discarding a token performs no syscall, reserves no payload bytes, and parks
no task. Synchronous callers outside a task use the explicitly named
`read_some_blocking` and `write_some_blocking` functions.

During live recording, awaiting an operation reserves the entire requested
byte count against `host_payload_capacity_bytes` before admission.
Configure prepares exactly
`rund::SchedulerConfig::host_io_capacity` retained operation slots. It allocates
that array only when the configured capacity changes; equal-capacity scope
reset retains the array and free list. Admission projects the borrowed fd once
into `(native, host_id)`, claims one slot in `O(1)`, stores one immutable
operation `(kind, native, host_id, data, size, sequence)`, and clears the
awaiter's fd and span before suspension returns. The slot is then the only
operation authority through execution, completion, and exact release.

The intrusive free list and FIFO each admit or remove one slot in `O(1)` and
perform no queue scan. A completed live operation performs exactly one native
`read` or `write`; it performs no readiness syscall and walks exactly the
completed byte prefix once for canonical payload hashing. A lazily started
scheduler-owned host-IO worker is the only owner that calls the native byte
operation; task lanes never perform those blocking syscalls. Slot and payload
pressure fail with `task_capacity_exceeded` before the native fd can observe a
read or write.

On the supported 64-bit ABI, one private slot is at most 104 bytes. The default
capacity of 64 therefore reserves at most 6,656 slot bytes. Configure performs
one bounded slot-array allocation, and the first live operation may allocate inside
`std::thread` startup. After that startup, an admitted would-block operation,
including park, native attempt, completion, event commit, and slot release,
performs zero process-heap allocations; `runtime.task.host-io` instruments this
warm boundary. Scope reset is `O(1)` in slot capacity: release already returned
every drained slot to the free list, while scheduler-lifetime monotonic
sequences prevent reuse from aliasing an earlier release generation.

The POSIX byte-write owner suppresses `SIGPIPE` without changing the
process-wide signal disposition. Where `F_SETNOSIGPIPE` exists, the owner
enables that descriptor-local bit before the native write and leaves the
transition monotonic; restoring it would race another writer. Other POSIX
targets block `SIGPIPE` in the executing thread, snapshot whether it was
already pending, and consume it only when it was absent at entry and pending
after the `EPIPE` write. This state-transition rule preserves a caller's
pre-existing pending `SIGPIPE` without claiming provenance for a concurrently
injected external signal. A caller that already blocked the signal does not
inherit an additional pending signal from runD. Because `Fd` owns the admitted
native descriptor, the descriptor-local suppression remains observable through
later operations on the same owner's view; it never changes the process-wide
handler.

The current deterministic authority is one FIFO host-IO worker. It executes
operations in scheduler-sequenced submission order and does not begin the next
native syscall until the preceding coroutine has resumed and committed its
`IoRead` or `IoWrite` event and payload. This is an ordering and lifetime
contract, not a throughput claim. The coroutine frame owns the caller buffer
for the full park-to-resume interval, and the scheduler retains the operation
slot until `await_resume` completes. Reset first closes admission and drains
that owner, then stops task lanes; an admission already in progress must either
enter the FIFO or release its slot before the worker can stop.
Each admission receives a monotonically increasing nonzero 64-bit submission
sequence. A worker waits for equality with that exact released sequence, not a
reusable slot's boolean occupancy, so immediate `read -> write` reuse cannot
lose the first release through an ABA transition. The largest issued sequence
is `UINT64_MAX - 1`; admission fails closed once the next sequence reaches
`UINT64_MAX`, so neither the submission nor release generation can wrap.
The slot lifecycle has one state authority:
`Free -> Admitting -> Queued -> Running -> Complete -> Free`. Read/write kind,
native outcome, and lifecycle are enums rather than correlated boolean fields;
an impossible mixed state therefore has no representable public path.
Scope drain waits until every external direct job has resumed and released its
slot. Reset then closes admission, joins a worker whose FIFO and admission
count are empty, and retains both sequence counters. Submission identity is
therefore monotonic for the Scheduler lifetime rather than the scope lifetime:
a newly claimed slot cannot erase a release generation that the worker has not
yet observed, and an equal sequence from an earlier scope cannot satisfy a new
release wait.

Host-IO await tokens do not accept a stop token and the scheduler does not
destroy a parked frame to cancel an accepted blocking syscall. Cooperative
task cancellation therefore does not interrupt an in-progress native
`read`/`write`; scope drain and shutdown wait for that syscall, event commit,
and exact slot release. Callers that need bounded cancellation latency must use
nonblocking descriptors plus readiness waits or arrange platform-level fd
shutdown outside the task-safe transfer contract.

During strict replay, awaiting does not start or enqueue the host-IO worker and
never calls the native fd. It consumes exactly the next `IoRead` or `IoWrite`
event and payload binding. Replay read validates that binding, then
decodes each archive chunk directly into the completed caller-buffer prefix.
The returned success status is the only consumption boundary: after any
failure, destination contents are unspecified and must not be consumed. In
particular, corruption in a later chunk may leave an earlier prefix written,
but can never return success. Replay write hashes and compares exactly the
completed caller prefix across every archive chunk without materializing a
second whole-payload vector.

`[[nodiscard]] Fd replay_fd(std::uint64_t host_id) noexcept;` is owned by the
transitive `<rund/host/io.hpp>` leaf and reachable through the sole direct
public aggregate `<rund/host.hpp>`.
`io::replay_fd(host_id)` creates a replay-only owner with no native close to
perform. Its `view()` is accepted only in replay mode and only when `host_id`
matches the next expected host event. Record mode still requires ownership
transfer through `take_native_fd(native_fd)`.
Raw libc `read` and `write` remain confined to the platform IO owner.

Network sockets are created only through `rund::net::open`. The public
move-only `Socket` owns lifetime, while its trivially copyable `SocketView`
borrows identity without gaining close authority. Neither value admits,
exposes, or reconstructs a native descriptor. The private generation guards
stale live readiness after operating-system descriptor reuse; ordered host
observations remain replay identity.

Socket admission proves native descriptor identity once at the lifecycle
boundary. Readiness registration acquires one generation-checked operation
lease from the view's stable entry, and the matching one-shot ticket performs
a second generation lease when consumed because close or reuse may occur
between those boundaries. Ticket consumption performs no registry map search
and neither boundary repeats `fstat` per packet. Arbitrary raw `close`,
descriptor replacement, or reuse outside the owning `Socket` remains outside
the host API contract above.

The process registry retains address-stable slots, while its active descriptor
index erases on completed close. Each owner and view stores one slot pointer
and one generation, so the byte-operation path performs no descriptor lookup.
Admission, recycling, capacity charging, the exact retained-slot bound, and
generation exhaustion are owned only by [Network](./net.md); this Host page
does not repeat a second formula or exhaustion policy.

The complete address, lifecycle, byte, readiness, replay, complexity, limit,
and telemetry laws are owned by [Network](./net.md). This host page owns only
the shared event and payload substrate and does not mirror those network laws.

Environment reads are external input. A deterministic scope may consume them
only through `rund::host::env::get`, which returns
`rund::task::Result<std::string>` and records name and value evidence. A missing
variable is a successful empty string. Empty names and names containing an
embedded NUL byte return `TaskInvalid` before calling the host environment or
recording an event. There is no parallel environment-specific result wrapper.

Canonical domain-free workload input uses one `rund::replay::Channel` formed
from `Binding::input(Input{id, schema}, source)` and stores the validated bytes
in the shared replay payload archive. The Live/Record source returns sequence;
Replay/Scenario resolve it from the transcript and `Value::sequence()` exposes
the canonical result. Applications must
decode and validate raw evidence into that canonical byte schema before
mutating domain state. The complete user contract is owned by
[Replay](./replay.md).

Host event ordering is owned by the scheduler sequencer. External wake timing
is not semantic authority; the recorded host event sequence is.
Canonical structured fields are streamed through a source-private hash state
with explicit little-endian integers and unsigned 64-bit length prefixes. The
owners do not dump platform object representations and do not allocate an
intermediate canonical byte vector. `host::hash_bytes` remains the sole public
stable hash for an already ordered raw byte span.
Runtime scope reports expose scheduler-owned host event records for timer sleep
wakes, IO readiness, and task-safe read/write completions. The
`host_event_capacity` limit bounds retained records and increments
`task::Stats::host_events_dropped()` after the retained vector is full while
`task::Stats::host_events()` counts every recorded event. The
`host_payload_capacity_bytes` run limit bounds retained host IO payload byte
records for scheduler-owned `IoRead` and `IoWrite` evidence. Retained payload
records are byte evidence for replay substitution only; they do not define a
file format, worker protocol, gameplay command, buffering policy, retry path,
hidden queue, packet schema, session, client, rollback, or gameplay meaning.
Scheduler-owned `IoRead` and `IoWrite` reserve enough payload capacity for the
requested byte count before native fd transfer; insufficient capacity fails
closed with `task_capacity_exceeded` before reading from or writing to the
native fd. `host_io_capacity` independently bounds admitted and queued task
operations; zero disables task host IO and fails before a native side effect.

### Host I/O Replay Payload Storage

Host I/O replay payload authority is the source-internal
`Archive`. Archive records bind event sequence, event kind,
or the canonical input `(source id, schema, sequence)`, plus completed byte
count, payload hash, and piece references. Host-role records keep every input
identity field zero; input-role records require nonzero source id and schema.
Archive chunks bind
chunk id, codec, uncompressed byte count, encoded byte count, uncompressed
hash, and either embedded encoded bytes or a spill segment location. The
logical payload hash is over the uncompressed payload bytes and their event
binding, not over the physical storage encoding.

Each input record also binds `source_event_offset/count`,
`source_payload_offset/count`, and `source_hash`. Record opens this range only
at a root input boundary; nested capture and simulation mutation fail closed.
Strict replay and scenario validate the whole range before consuming it and
project every adopted event through `CommitHostEvent`, so host-event retention,
network call/byte counters, ordering, and trace hashing have one owner.

The same Store owns a non-authoritative raw network diagnostic window for
successful `NetRecv`, `NetRecvDatagram`, and `NetRecvVectored` evidence. Its
fixed byte ring and fixed record ring are prepared once. Network admission is
the sole pointer, overflow, and available-byte authority and carries its exact
admitted-byte proof `A` into the source-private raw byte view. For completed
prefix size `P`, Store checks `P <= A` in `O(1)` and performs one ordered prefix
traversal while copying the `P` bytes directly into the ring; it has no second
slice-completeness pass, trusted public validity bit, warm allocation, or gather
buffer. A proof mismatch is dropped before any callback or eviction. Space
pressure removes oldest complete records; an input larger than the byte window
is dropped whole. Archive snapshot alone materializes the current logical bytes
into one source-private `payload::Bytes` owner. The diagnostic hash covers records, bytes, and
retained, evicted, and dropped counters. It is deliberately excluded from
`payload_hash`, input hash, transcript hash, replay hash, and checkpoint
identity.

The product facade exposes only the bounded diagnostic projection needed by
an operator: the borrowed `std::span<const Capture>` from `Record::captures()`,
`capture_hash()`, and `capture_report()`. Each Capture carries borrowed
immutable bytes; archive chunks, codecs, offsets, and backend placement remain
private instead of becoming a second SDK storage model. Acquiring the range
performs no allocation or copy; writing an exported capture is the caller's
explicit `Theta(N)` operation.

Private payload storage names live under
`rund::node::replay_detail::payload`; the namespace supplies the replay and
payload context, so its owners are `Store`, `Materialization`, `BuildResult`,
`Build`, and `Materialize` rather than parallel names that repeat that context.
`Store` and its backend/cache chain are move-only single owners; copying a
spill cache would duplicate iterator-bearing index state without duplicating
the external segment authority. Archive metadata is the explicit copyable
snapshot boundary. `payload::Bytes` is the sole encoded-byte value in both
`payload::ArchiveChunk::encoded` and the private backend `Blob::encoded`.
A producer completes a mutable `std::vector<std::byte>` and freezes it once;
the resulting value exposes only `span()`, const `data()`, `size()`, and
`empty()`. The value exposes no mutable container, and `Store::Archive()` is
the sole archive snapshot operation.

Every non-empty Memory-mode encoded chunk has one immutable shared byte owner.
`Store::Archive` copies record, piece, and chunk metadata and shares that owner
rather than copying encoded bytes. Spill-mode chunks instead retain
coordinates into one immutable shared `SpillGeneration` owner. A later Store
append, `Clear`, or destruction releases only that Store's references. An
already returned archive, including one held by an earlier `Session::Result`,
therefore remains valid and observes exactly its original record/chunk order
and bytes in either mode. The Spill owner also retains the committed storage
Reservations; final-owner destruction is the sole healthy-process removal and
Budget-refund boundary.

`Store` hides the backend behind one owner allocation, so scheduler translation
units do not parse or depend on blob, codec, spill-cache, or hash index
representation. That type-erasure boundary costs one allocation per Store and
one pointer load at backend calls; it adds no allocation per record or chunk.
Each chunk already requires a bounded `Theta(C)` hash/codec pass, so the
constant owner indirection does not change the byte-work bound while sharply
narrowing backend-only rebuild fanout. Freezing a distinct non-empty encoded
chunk adds one shared-owner control allocation while transferring the
producer's existing payload allocation. The control cost is independent of
payload size; its byte size is standard-library ABI dependent and is not a
product ABI constant.
`payload/hash.hpp` is the sole record-framing owner. Materialization, live
store hashing, and archive validation feed the same role, host event identity,
input source id, input schema, input sequence, completed byte count, declared
hash, materialized byte count, and ordered bytes into that owner. Changing
compression or spill placement therefore cannot change the logical identity,
while changing only an input schema necessarily changes the record, archive,
input-transcript, replay, and checkpoint transcript-prefix identities.

Append computes and stores each canonical record hash while the admitted input
span is already live. For a recorded Store, `payload_hash()` therefore folds
the `R` stored record hashes in canonical order, and `Store::Archive()` exports
only `Theta(R + Q + K)` metadata for `R` records, `Q` pieces, and `K` chunks.
It does not decode or rescan `P` payload bytes. A memory archive loaded from an
external value validates its decoded content in `Theta(P)` work before
admission; spill storage preserves its existing lazy fail-closed validation.

Store admission separates raw-byte and structured identity work. It verifies
the declared payload with one seeded XXH3 pass and feeds the same bytes to the
structured record-framing owner. For payload size `P` and
`K = ceil(P / 64 KiB)` chunks, `K = 1` reuses the verified payload hash as the
chunk hash, so no third raw-byte scan exists. For `K > 1`, the disjoint chunk
hashes require one additional total `P` bytes of XXH3 work because a whole
payload digest cannot derive its subrange digests. Codec and exact
deduplication traversals remain separate physical-storage work. The extra
structured scan is retained because record identity includes ordered bytes in
addition to the declared raw hash; it is not a payload copy and normally reads
cache-resident input.

The encoded-byte ownership path for one record followed by one replay is:

| Current boundary | Encoded payload bytes copied |
| --- | ---: |
| Recorded Store to retained archive | `0` |
| Expected replay record to run configuration | `0` |
| Replay Store to actual archive | `0` |

This table counts encoded payload bytes only. Metadata vectors copy or move in
`Theta(R + Q + K)`, each shared chunk owner is retained once per copied
archive, and archive admission performs the content reads required to
prove hashes and storage integrity.

Materialization consumes its record vector by value. An rvalue transfers its
payload buffers; an lvalue pays the one explicit ownership copy. Resolving a
store for materialization moves each resolved byte vector into its record. The
ownership transfer is constant time and copies zero resolved payload bytes.
Resolution validates physical chunks while writing its reserved destination
vector. A one-piece record whose chunk length and hash equal the record length
and payload hash needs no second joined-payload state: successful chunk
validation already proves the complete record byte identity. Multi-piece
records alone feed their ordered decoded spans to one joined state.
Resident archive validation passes encoded chunk spans directly to the shared
verifier. Raw spans feed hash states in place; RLE tokens feed the same states
through the fixed 130-byte repeat window. The required `Theta(E + P)` codec and
hash work allocates no decoded or encoded payload vector, while byte order,
hash framing, bounds, and fail-closed reasons remain canonical.

Materialization walks the store's contiguous record vector by position and
resolves that position directly. For `R` records and `P` payload bytes this is
`Theta(R + P)`; it does not perform an event-key search for every record. The
record vector remains the ordering authority, so the optimization cannot
reorder the canonical record hash or archive records.

Live replay uses that same canonical record position. The scheduler passes
`next_expected_host_payload` plus the committed sequence, kind, completed byte
count, and payload hash as one `Binding`; `Store` validates the selected record
before reading or matching bytes. There is no sequence/kind search API and no
second binding implementation in the scheduler. Across `R` replay payload
events and `P` bytes, ordered lookup and byte verification are therefore
`Theta(R + P)`.

Replay payload transfer has one owner and no whole-record aggregation buffer.
Let `P` be completed payload bytes and `K` the number of fixed-size chunks.
The current logical-payload movement at the Store boundary is:

| Replay path | Logical payload writes and transient storage |
| --- | --- |
| Memory raw/RLE read | Decode writes `P` bytes directly into the caller destination and validates each chunk. The common one-piece record reuses that proof; multi-piece records additionally hash their ordered output spans. |
| Spill read | A cache miss decodes each piece into one bounded chunk scratch, commits it once to the caller destination after validation, and transfers that allocation into the cache without another byte copy. A cache hit copies cached bytes directly into the destination. Only multi-piece records hash the joined order. |
| Memory raw/RLE write match | Exact comparison and chunk validation use caller-owned spans with no decoded payload vector. The common whole-chunk record needs no duplicate joined hash. |
| Spill write match and dedupe | A validated encoded segment feeds exact comparison and chunk validation; multi-piece order alone advances a joined state. Encoded segment/blob temporaries remain bounded by chunk storage. |

These are logical payload write counts, not total work. A memory replay read
writes exactly `P` destination bytes. A spill replay cache miss writes `P`
bounded scratch bytes plus `P` destination bytes; admitted cache storage takes
ownership of that scratch and adds no byte write. A cache hit writes exactly
`P` destination bytes and allocates no decoded temporary. Raw copy, RLE
expansion, and seeded
XXH3 validation are separate vectorizable operations over cache-local spans;
they do not allocate or copy a second whole-record payload. A chunk failure
can leave a partially advanced hash state, but that state and the destination
are discarded with the failed operation. Spill segment framing still parses
its bounded encoded record and owns its required encoded storage buffers.
Failed output is non-consumable, and a later-chunk corruption cannot be
reported as success.

Write matching does not need a preceding whole-record hash scan. Binding and
length are checked before byte access; then every piece is compared in order,
and the final offset must equal `P`. Each physical chunk is verified against
its declared hash. For one whole chunk, equality of chunk and record length and
hash makes that proof sufficient. For multiple chunks, the matching spans also
advance one ordered joined `ByteHash`, whose final value must equal
`record.payload_hash`. A corrupted spill segment therefore fails even when
caller bytes equal the corrupted physical bytes, and individually valid chunks
also fail when their concatenation has the wrong declared record hash.

Chunk deduplication computes the stable chunk hash once, then uses a
source-private hash-to-candidate index. Each hash bucket owns a vector of blob
indices in insertion order, and every candidate requires exact byte
equality. The hash table is only a lookup accelerator: archive serialization,
record pieces, and logical hashes continue to use the canonical record and
blob vectors and never iterate the table. For `B` chunks of at most `C` bytes,
ordinary distinct-hash construction is expected `Theta(B C)` byte work with
`Theta(B)` extra index memory. A deliberately colliding workload remains
bounded by the honest worst case `Theta(B^2 C)`; collision handling changes
neither identity nor insertion order.

When one record fits in one 64 KiB chunk, append reuses the payload hash already
validated at admission as the chunk hash, so the common single-chunk path has
no separate chunk-hash traversal. Multi-chunk records still hash each chunk
once because their chunk identities are not derivable from the whole-record
hash.

The upper layer chooses exactly one storage mode through
`rund::replay::Storage`:

- `StorageMode::Memory` keeps encoded chunks in process memory and requires an
  empty
  spill directory.
- `StorageMode::Spill` creates one uniquely named generation directory directly
  under the caller-provided root, then writes deterministic segment files named
  `host-replay-payload-000000.segment`,
  `host-replay-payload-000001.segment`, and so on inside that generation.

The caller owns the configured root and every unrelated entry under it. runD
creates the root path when absent but never treats that act as ownership of the
root itself. runD owns only a direct child whose name starts with
the private replay-spill prefix and whose `.rund-owner` file contains the exact
generation marker. Each live generation also holds an exclusive nonblocking
lock on its `.rund-lease` file. Generation names include process-local identity
and a monotonic sequence to prevent two live Stores from sharing or truncating
one segment namespace.

Healthy-process reclamation is RAII. The generation is shared by every Store
and immutable archive that can resolve its segment coordinates. When the last
reference dies, runD verifies direct-child containment, name prefix, and the
exact ownership marker before making one best-effort removal of that
generation tree; destruction also releases the generation's committed
Reservations and refunds its Budget charges. The removal path has no throwing
result channel. If the filesystem refuses removal, the recognized stale tree
can remain after the in-process charge is released. A first-use scan in a
future process may retry it; later generations in a process that already
scanned this root do not. The caller must keep the root itself usable while a
live Spill Record can be read. runD never removes the caller root or an
unrecognized sibling entry.

Crash reclamation is necessarily different because destructors do not run.
On the first generation-creation attempt that successfully registers each
absolute, lexically normalized root in a process, runD scans only direct child
directories with the recognized prefix and exact marker. It removes one only
after opening its lease and acquiring the exclusive lock without blocking.
The normalized root is then remembered process-locally; later generations
under that root do not rescan, which keeps repeated creation from multiplying
scan cost. If registration allocation fails, scavenging is skipped and
generation creation remains safe; an unregistered later attempt may retry. A
live locked generation, an unmarked directory, a mismatched marker, an entry
outside the root, or any failed inspection is preserved. Scavenging is best
effort: it is not an immediate crash cleanup guarantee, a recursive volume
sweep, or authority over artifacts created by callers or other producers.

Each accepted chunk creates one segment record with five 64-bit fields and one
8-bit codec field, so the fixed header is 41 bytes. For `E` unique encoded
bytes and `K` chunks, the exact logical segment-file content is `E + 41K`.
The header is materialized once in a fixed stack buffer in canonical
little-endian order and submitted with the borrowed encoded payload through
one positioned vectored write on the normal path. There is no stream buffer,
locale state, payload staging allocation, or payload-sized user-space copy.
Reads perform one fixed-header read, validate its bounds and identity, then
allocate the exact encoded owner and read the payload directly into it.
Replay's conservative allocated-byte charge, per-Session and shared ancestor
Budget limits, and advisory free-space headroom are owned by
[Replay](./replay.md); generic hierarchy laws are owned by
[Storage](./storage.md). Neither metric includes generation control files,
directory or filesystem metadata, journals, snapshots, or replicas.

The physical chunk codec and generation layout are private replay storage
representation. Callers should depend on canonical archive records and hashes,
not on segment paths, a gameplay protocol, or a worker framing format.

Only canonical archive payload records are accepted in the runtime replay
format. Unknown flags or malformed records fail closed during decode.
Encoding choices do not change replay semantics: replay `read_some`
substitutes recorded bytes and never reads the native fd to repair missing
payload bytes.

Replay mode remains fail-closed:

- a missing expected payload fails replay;
- an event/payload binding mismatch fails replay;
- corrupted blob bytes fail decode before replay starts;
- a record payload hash mismatch fails decode before replay starts.

Host-event status preserves native-substrate unavailability independently from
a syscall failure. Task-safe IO therefore records `Status::Unsupported` for
`IoUnsupported`, and replay returns the same reason without starting the
host-IO worker or calling the native fd.

The `runtime.task.host-io` registry row retains one runner in
`tests/contract/runtime/task/host/io/run.cpp`. Its semantic owners are
`surface`, `admission`, `replay`, `order`, and `signal`; `fixture` alone owns
temporary native descriptors and paths. Every owner creates and releases its
own Session, descriptors, and signal state, so no result depends on translation
unit initialization or the preceding semantic owner. The structure exposes no
product API or alternate IO oracle. Editing one semantic leaf compiles that
leaf and relinks the Runtime runner.

The `runtime.task.replay-spill-storage` row keeps its registry symbol in
`tests/contract/runtime/task/replay/spill/storage.cpp`.
`spill/local/model.cpp` is the single compiled owner for its temporary roots,
Store fixtures, generation and segment-path projection, 41-byte segment
oracle, allocation accounting, and `RunSegmentsContract()` first-failure
runner. That runner preserves the fixed `generation`, `lifetime`, `budget`,
`append`, `artifact`, then `layout` sequence. Each semantic leaf constructs and
removes its own root, Session, Store, Budget, and retained archive state, so
its result does not depend on a preceding leaf. Editing one segment leaf
compiles that leaf and relinks the existing Runtime executable; changing the
shared model intentionally invalidates every Spill contract consumer.

`node::EncodeHostReplayEvents` emits the HostEvents binary artifact kind.
After count and hash, every event writes sequence delta, kind, status,
presence bitmap, and only the nonzero stable values owned by `host::Event`:
task id, logical-time delta, stream id, draw id, host handle id, offset,
requested bytes, completed bytes, native errno, name hash, path hash, and
payload hash. Decoding rejects an invalid envelope, non-minimal or overflowing
integers, unknown enum values or bitmap bits, noncanonical zero/present
combinations, oversized counts, truncation, trailing bytes, and hash
mismatches. The public
`DecodeHostReplayEvents` bool wrapper clears its output vector on failure; the
decoded vector is valid only when the function returns true. On success the
wrapper consumes the decoder result and transfers its event-vector storage to
the caller. It must not allocate and copy a second `host::Event` array: after
the one-pass decode and hash check, wrapper overhead is constant-time ownership
transfer instead of `event_count * sizeof(host::Event)` bytes of element copy.
Runtime replay calls the same host codec directly on its borrowed reader and
reports the same host replay rejection reason when embedded host evidence is
malformed. Payload bytes are copied once from the borrowed artifact span into
their final immutable chunk owners. Finalization moves the decoded event
vector and payload archive into the replay record; it does not retain a second
evidence constructor, recompute the event hash at that transfer boundary, or
copy archive metadata.
The final record invariant independently verifies the adopted event hash
and archive. For `E` events, `Q` payload pieces, and `K` chunks, finalization
therefore adds zero element copies and zero destination-vector allocations;
artifact-to-archive assembly adds no second per-record piece allocation and no
shared chunk-owner increment. Parsing owns one piece vector per decoded
record plus the archive's required outer record and chunk vectors.

Runtime scopes internally admit host events for live recording or strict
replay. SDK callers do not select that mode or supply expected event/payload
vectors through `SessionConfig`; `rund::replay::live`, `record`, `run`, and
`scenario` are the sole product execution boundaries. The source-private
execution-plan owner carries their expected evidence to the Scheduler. Replay
consumes the next expected host event at the replay cursor; it must not skip
events to find a later match.
Each produced or substituted event receives its scheduler-owned sequence number
before comparison, then the replay checker compares sequence, kind, status,
task id, logical time, stream id, draw id, host handle id, offset, requested
bytes, completed bytes, native errno, name hash, path hash, and payload hash at
the same event index. Event kind, host id, status, count, hash, field, extra
event, or unconsumed expected-event mismatch fails the scope with
`host_replay_event_mismatch`. Host IO payload byte, payload count, payload
hash, missing payload record, or wrong payload-record binding mismatch fails
the scope with `host_replay_payload_mismatch`. Host IO payload records must
bind to the committed host event sequence and payload hash. When the mismatch
is owned by a current scheduler task, that task is failed with the same reason.
Recording retains host events according to `host_event_capacity`. Ordinary
`run` may report a bounded diagnostic prefix plus a nonzero drop count, but the
product replay facade never publishes that incomplete evidence as a successful
`rund::replay::Record`.

## Naming

Use these public namespaces:

- `rund::host::random`
- `rund::host::chrono`
- `rund::host::timer`
- `rund::host::io`
- `rund::host::env`
- `rund::net`
- `rund::host`

These focused namespaces are the complete public host authority; no parallel
product namespace is admitted.

## Update Rules

Changing host public names, event fields, random algorithms, replay encoding,
or raw API admission rules must update this page and the matching contract
tests in the same change.
