# Replay

This page owns the product replay boundary for one open `rund::Session`. It
defines canonical input capture, repeated record/replay/scenario execution,
scope-local evidence, checkpoint schema and lineage, persistence, and the
public surface. Host acquisition and payload storage mechanics remain
owned by [Host](./host.md); scheduler ordering remains owned by
[Scheduler](./scheduler/README.md).

## User Contract

The application owns domain meaning. It chooses the authoritative input
boundary, canonical byte schema, checkpoint state schema, and restore codec.
runD owns ordering, bounded storage, scope identity, record/replay selection,
checkpoint lineage, hashes, and telemetry. No replay type names a command,
player, body, ECS component, physics state, or game object.

One callback is the sole simulation graph for Live, Record, Replay, and
Scenario:

```cpp fragment
rund::Session session;
const auto opened = session.open(config);
if (!opened) {
  return opened.exit_code();
}

rund::replay::Binding replay{};
auto source = [&](rund::replay::Writer& out) -> std::uint64_t {
  const auto input = receive_commands();
  (void)out.append(input.bytes);
  return input.sequence;
};
auto commands = replay.input(
    rund::replay::Input{.id = 0x43u, .schema = 0x1001u}, source);
auto step = [&](rund::replay::Context& context, rund::Session& active) -> void {
  auto value = commands.read(context);
  if (!value) {
    return;
  }
  simulate(active, value.sequence(), value.bytes());
};

auto baseline = rund::replay::record(session, step);
auto checked = rund::replay::run(session, baseline, step);
auto changed = rund::replay::scenario(session, baseline, choices, step);

session.close();
```

The callback is borrowed for the call and is never copied, retained, or
invoked after the result is published. Its exact return type is `void`; a
discarded user result is not an error channel. It accepts either `Context&` or
`Context&, Session&`. The source bound once by `Binding::input` runs once per
read in Live or Record and zero times in Replay or Scenario. It returns the
canonical sequence that Replay later resolves from the same transcript row.

`SessionConfig::replay` owns only storage, diagnostic, and bounded-capacity
policy. Execution mode, expected evidence, scenario choices, cursors, and
continuation lineage are scope values. They are not configuration fields and
cannot be authored by an SDK consumer.

The installed reusable-Session example gives execution and shutdown separate
typed terminals while keeping one ordinary close path. It stores the Replay
operation's process result, calls `Session::close()` exactly once even when the
operation failed, then returns the operation failure first; the close failure
is returned only when Replay succeeded. This preserves the causal operation
diagnostic without skipping resource shutdown or asking callers to remember a
`drain(); close();` sequence. Process exit `2` remains reserved for an
application validation mismatch after every runD product result succeeded.

## Server-Native Boundary

Remote and browser clients stop at the application protocol boundary. They
send bytes with application meaning and never select a runD compute backend or
name a device. The server owns decode and
validation before binding the resulting canonical bytes once:

```text
network bytes
-> application decode and validation
-> Binding::input canonical row
-> one callback(Context&, Session&)
-> server-owned session-bound Device and Job
```

The server selects exactly one native `Cpu`, `Metal`, or `Vulkan` Device before
the Replay scope and uses that same prepared Job from the one callback. Record
invokes the bound input source and Replay substitutes the retained canonical
row; neither mode changes the callback or opens another backend. Backend
choice is server policy, not canonical input, so the wire schema and Replay SDK
contain no backend selector or execution fallback.

`rund::net` is the meaning-neutral byte transport, `Binding::input` is the
canonical input authority, and the callback's `Session&` is the bridge to
native Compute execution. Protocol framing, authentication, and normalization
are application-owned because runD cannot infer which pre-canonical bytes are
authoritative without acquiring domain meaning.

## Public Surface

Replay lives below `rund::replay`. Its durable public values are:

```text
Code          one typed replay outcome authority
Input         canonical input id and schema
Binding       input construction and optional checkpoint schema/restore owner
Channel       one Input plus borrowed Live/Record source
Choice        one Channel-owned Scenario replacement with explicit sequence
StorageMode   memory or caller-owned spill storage
Storage       bounded replay storage policy
StorageReport measured payload storage cost
Diagnostic    bounded raw-network diagnostic policy
DiagnosticReport measured diagnostic retention
Capture       one row in the borrowed raw-network capture range
Context       scope-bound canonical input access
Writer        bounded prepared source buffer
Value         immutable resolved input bytes
Live          move-only Live execution result and scope evidence
Record        immutable execution evidence
Check         strict replay comparison
Diff          record difference
Window        bounded mismatch context
InputPoint    canonical input identity, byte length, and hash
Scenario      scenario result
Checkpoint    immutable restore boundary
Resume        one checkpoint/schema/borrowed-restore execution binding
History       bounded rolling segment owner
Segment       retained record/checkpoint pair
Retention     History bounds
Save          streamed artifact-write outcome
Load<T>       bounded artifact-load outcome
```

Execution uses the short verbs `live`, `record`, `run`, and `scenario`.
Comparison uses `check`, `diff`, and `window`. Checkpoint chaining uses
`Binding::checkpoint`, `advance`, and `resume`; History publication uses
`Binding::append`. `Record` and `Checkpoint` own streaming `save()`
and bounded static `load()` so the source type selects the artifact schema
without a prefixed free function or a return-type-only overload:

```cpp fragment
std::vector<std::byte> artifact;
auto saved = record.save([&](std::span<const std::byte> bytes) {
  artifact.insert(artifact.end(), bytes.begin(), bytes.end());
  return true;
});

auto loaded = rund::replay::Record::load(artifact, limits);
```

The public replay authority is exactly the `rund::replay` namespace, the short
execution and comparison verbs above, and type-owned Record and Checkpoint
persistence. `<rund/replay.hpp>` is the sole consumer entry and a
declaration-free composition. Declarations live once in the support hierarchy:
`input.hpp` owns canonical input access, `record.hpp` owns immutable evidence
and comparison, `state.hpp` owns checkpoint persistence and restoration,
`history.hpp` owns bounded retention, and `run.hpp` owns scope execution.
Those support paths are dependency boundaries for the producer, not additional
consumer entries. The dependency direction is

```text
code/storage/limits -> binding -> input -> record -> state
                                      state -> history
                                      state -> run
```

and the aggregate includes each terminal owner exactly once. A producer source
includes the narrowest owner it implements; no producer object depends on the
aggregate merely to obtain one replay value.
`rund/replay/code.hpp` is the allocation-free transitive source leaf for the
outcome implementation, not a second direct consumer entry. It does not
duplicate the schema: the enum, validation, family, text, and internal Runtime
mapping all derive from the single ordered code definition and are reached by
SDK consumers through `<rund/replay.hpp>`.

## Session Scope

Replay requires an open, Running `Session`. It reuses the workers, scheduler,
reactor, prepared storage, compute host, and backend selected by `open()`.
Calling replay on a closed or unopened Session fails before user code. A
Session admits exactly one Live, Record, Replay, or Scenario scope at a time:

- a concurrent caller fails with `runtime_scope_busy`;
- same-Session callback reentry fails with `runtime_reentry_forbidden`;
- a successful scope drains every task admitted by that scope before it
  publishes a result;
- plan validation and installation happen only while the Session is quiescent;
- invalid expected evidence, choices, checkpoint lineage, or state schema
  fails before restore or simulation callback invocation.

Plan installation is transactional. A scope guard restores Live mode and
clears the expected owner, choices, replay cursors, and continuation identity
after success, callback exception, drain failure, or early rejection. A failed
scope cannot poison the next ordinary Session scope.

The expected `Record` owns one immutable prepared replay projection. Strict
replay installs a shared reference to that projection; it does not copy the
record's events or payload archive and does not rebuild its lookup structures.
Scenario validates and freezes only its choices against the prepared expected
index. Reusing the same Record never changes its bytes, hashes, or prepared
projection. A loaded Record builds that projection once under its cold
builder lock and publishes it with release ordering. Every warm Replay or
Scenario checks the stable pointer with one acquire load and never enters that
lock. The immutable owner remains unchanged for the Record lifetime.

## Canonical Scope Identity

Physical scheduler identities remain monotonic for the Session. Task, scope,
wait, timer, channel, observation, host-event, and trace identifiers are not
reset between scopes because a reset could make a stale handle name a new live
object.

At scope admission the scheduler captures a base for every physical identity
that enters replay evidence. The canonical projection is the checked offset

```text
canonical(x) = checked_sub(physical(x), scope_base(x)) + 1
```

for identities whose first scope-local value is one. Zero keeps its declared
sentinel meaning. Underflow or overflow is an internal scope failure. Physical
handles and registries continue to use physical identity; only replay
evidence, hashing, comparison, and persistence use the canonical projection.
Therefore two semantically equal scopes in one warm Session produce equal
records even though their physical identifiers differ.

Canonical input admission applies the same projection to its host-event
boundary. If `p` is the next physical host-event sequence and `b` is the
scope's captured event base, then the number already emitted in this scope is

```text
E = canonical(p) - 1 = p - b
```

Record input capture requires `host_events.size() == E`. Comparing against
`p - 1` is invalid in a reusable Session because it counts events retained by
earlier physical scopes after the current scope-local vector was cleared.

Outside an admitted scope plan, projection is an identity function. Scheduler
work such as product compute submitted directly against an open Session keeps
physical identifiers and cannot poison a later replay scope. Checked base and
handle-map failures apply only after scope-plan installation. Scope admission
flushes earlier physical batches before capturing canonical bases; previous
root work can never be projected through a new scope base.

Native descriptors are not monotonic and cannot use base subtraction. The
scheduler instead prepares one bounded open-addressed handle table at Session
configuration. Within a scope, the first observed nonzero handle receives the
next canonical handle, raw descriptor `fd` uses the same `fd + 1` key, and a
successful `IoClose` retires the active mapping. Reuse of that physical
descriptor therefore receives a new monotonically increasing canonical handle
without changing already published evidence. `NetAccept` resolves the listener
first and admits the accepted handle second. Handle-table exhaustion,
projection underflow, a zero base, or canonical-width overflow fails the scope
with `replay_scope_identity_invalid`; it never publishes a sentinel substitute
or continues hashing partial evidence. Scope admission advances an epoch and
resets only counters, so the warm reset is `O(1)` and allocation-free. Epoch
wrap clears the prepared table once without changing identity order.

`scheduler.host_handle_capacity` is the sole bound for this domain-neutral
mapping and is independent of the network socket registry. Zero disables
nonzero host-handle replay identity and rejects the first such event with
`replay_scope_identity_invalid`; handle-free events remain valid. The default
bound `H = 1024` prepares `2048` slots at load factor at most one half. On the
64-bit product ABI a slot is 24 bytes, so the flat table owns exactly `49152`
payload bytes, excluding allocator metadata.

Scheduler statistics, observations, host events, payload records, logical time,
random projection, and trace are scope-local replay evidence. A scope result
never combines cumulative task counters with scope-local vectors. The scope
captures counter bases and publishes checked deltas. Session lifecycle trace
such as configure, start, drain, and stop is not part of an individual replay
record.

The ten pure additive task totals selected by the semantic hash—spawned,
completed, yields, joins, timers, channel sends/receives/closes, observations,
and dropped observations—consume the repository saturating counter owner at
every producer. At mathematical overflow, a newly executed scope therefore
retains `UINT64_MAX` in its task evidence and semantic hash input instead of
wrapping; the persisted field order, codec grammar, and hash mixing algorithm
do not change. `Failed` is not claimed by this rule yet because scheduler
progress also reads its value change as a host-replay activity signal.
Separating that control responsibility is required before its additive
producers can safely adopt the same absorbing boundary.

Variable replay evidence has two distinct lifetime owners. The Session reuses
prepared capacity `C` while a returned result independently retains its `B`
published bytes. Publication therefore requires `Omega(B)` byte work and
independent `B` storage: reusing the mutable Session buffer cannot also give a
previous result immutable lifetime. The scheduler performs one deterministic
compact publication copy from the prepared buffer into the result owner. It
does not make a second `B`-byte copy, exchange the prepared vector with an
empty vector, discard its reserved capacity, or grow evidence storage during
the callback. Telemetry reports the exact published byte count. Detail's
`finish_ns` interval contains result capture, immutable-owner construction,
and publication, so for `B > 0`

```text
effective publication rate = B / finish_ns
```

is a conservative end-to-end diagnostic, not raw `memcpy` bandwidth. A
standalone copy microbenchmark would measure a different operation and cannot
become a second Replay performance authority. The frozen Telemetry route
therefore owns the directly observed `B` and `finish_ns`; derived rates remain
non-gating under the repository performance contract.

## Cost Model

Let `C_open` include discovery, backend selection, bounded scheduler/reactor
storage preparation, worker-lane start, and compute-host preparation. Let
`C_close` include the internal drain-and-stop transition, `C_scope` be warm admission and
evidence setup, `C_plan` be a prepared-plan install, and `C_sim` be application
work. Recreating a Session for `N` experiments costs

```text
T_reopen(N) = N * (C_open + C_plan + C_scope + C_sim + C_close)
```

while the required reusable Session costs

```text
T_reuse(N) = C_open + C_close + N * (C_plan + C_scope + C_sim)

saving = (N - 1) * (C_open + C_close)
```

The immutable expected-record projection is shared across scopes, so reuse also
avoids an evidence-plan copy. `C_open` is not constant in configured
capacity: the current scheduler prepares task/index, timer, observation,
host-event, reactor, ready-set, callable, coroutine-frame, and completion
owners, and starts `W` lanes. At minimum it writes the `2Q+1` task-index table
for task capacity `Q`, prepares reactor storage proportional to reactor bound
`R`, and reserves bounded evidence storage. Repeating it for a short thought
experiment is avoidable work.

For a configured simultaneous handle bound `H`, configuration writes a table
with load factor at most one half, requiring `Theta(H)` slots once. Expected
probe cost is `O(1)` under the fixed avalanche hash; admission, lookup, close,
and warm scope reset allocate zero bytes. Closed slots are reclaimed in place,
so table storage is bounded by simultaneous live handles rather than by the
total number of close/reopen cycles, while the emitted canonical counter
remains scope-monotonic.

For an expected record with `I` canonical input rows and `E` evidence rows, its
immutable replay projection is prepared once in `Theta(I + E)` work. A strict
scope installs it in `O(1)`: one acquire publication check and one shared-owner
retention, with no warm mutex acquisition. For `P` Scenario choices with
`B = sum(choice.bytes.size())`, lookup in the prepared expected index costs
`O(P log I)` and deterministic patch sorting costs `O(P log P)`. Admission
allocates one exact `B`-byte owner and one `P`-row patch array. Each borrowed
input byte is copied exactly once into that owner before the callback; every
patch is a slice of the same owner. There is no per-choice byte owner, pending
metadata array, or duplicate order vector. The implementation must not sort
all `I` expected rows again for every scenario. An identical frozen choice set
may reuse its fingerprinted plan, but cache reuse cannot change validation or
result identity.

Performance claims close only with measured cold Session, warm Live, warm
Record, warm strict Replay, and warm Scenario distributions. Report median,
p95, input/evidence cardinality, configured bounds, worker count, backend, and
allocation deltas. Wall-clock timing and cache-hit telemetry are diagnostics;
they never enter deterministic hashes.

## Canonical Input

`Input.id` and `Input.schema` are nonzero unsigned identities. Sequence zero
and an empty canonical byte string are valid. The schema is the identity of
the application's canonical encoding, not a domain type owned by runD. One
`Channel` retains the Input and a borrowed source thunk. Each Live/Record
`read(context)` invokes that source once; the source writes canonical bytes and
returns sequence. Each Replay/Scenario read invokes it zero times, validates
the next transcript Input, and returns the stored sequence and bytes through
the same `Value`. The caller cannot accidentally supply a different sequence
to Replay. There is no fallback to the source, network, a default value, or a
second graph.

For `N` logical reads of one schema, the application constructs one Channel
and calls it `N` times. Binding state remains `Theta(1)` instead of creating
`N` identity or callback adapters, and sequence contributes exactly one
unsigned word to each stack-local request and retained evidence row. Input,
Channel, and Choice allocate no storage.

Choice admission is deterministic. Choices sort by the complete canonical key,
must be unique, and must name exactly one expected input. Invalid id,
schema, duplicate, missing, ambiguous, or capacity-exceeding rows fail before
the simulation callback. Choice bytes may be shorter, longer, or empty; their
actual length and hash become Scenario result evidence and never become part of
the lookup key. A mismatch caused by a valid choice is a Scenario result, not
setup failure.

`Choice` is not an aggregate and has no public constructor. Only
`Channel::choice(sequence, bytes)` can form it, which prevents a replacement
from carrying an id or schema that disagrees with its bound Input.
`Choice::bytes()` is a borrowed span. The application only keeps its backing
storage alive through the `scenario()` call; runD validates, packs, and freezes
all choices before invoking the simulation callback. Mutating or releasing the
application buffer after `scenario()` returns cannot change a retained result.
The frozen owner retains exactly `B` payload bytes, excluding allocator
metadata; patch rows share it and do not copy their byte ranges.

The source has one bounded contract: it accepts `Writer&` and returns the
canonical `std::uint64_t` sequence.
`Writer::acquire(n)` borrows at most `n` writable bytes from the Session's
prepared input scratch, `Writer::commit(n)` publishes up to that acquired
prefix, and `Writer::append(bytes)` copies one chunk into that same owner.
Successful commits and appends are cumulative in call order, so the canonical
input is their exact concatenation and `size()` is monotonic. One or more
commits or appends are required. `acquire(0)` followed by `commit(0)`, or
`append(empty)`, explicitly records canonical empty bytes; returning without
either operation is an input failure.
Acquisition beyond the configured input/storage bound, commit beyond the last
acquisition, an operation while an acquisition is outstanding, or a write
beyond remaining
capacity fails the input without exposing partial bytes. `Writer&` is the sole
source output interface and the prepared Session scratch is the sole mutable byte
owner. Record commits the actual written length and canonical hash to evidence;
Replay and Scenario use stored or chosen bytes and invoke the Writer source
zero times.

A Channel is a noncopyable, nonmovable borrowed binding whose source must
outlive it. A resolved Value is scope-bound. Its internal reference uses a
stable Session-owned context plus a scope generation, so an escaped Value
fails closed instead of dereferencing a destroyed stack `Context`.

`window(expected, actual, context)` localizes the first canonical input
mismatch as an `InputPoint`: canonical row index, input id, input schema,
sequence, byte length, and content hash. `input_index()` is the zero-based
canonical input position;
`expected_inputs()` and `actual_inputs()` borrow only the bounded neighboring
rows selected by `context`. A missing row produces no point on that side while
an empty row retains a point whose `size` is zero. Because canonical input is
upstream of simulation evidence,
an input mismatch is localized before downstream observation, host, or trace
differences. The implementation makes one merge scan to locate the
first differing input and count both sides, then one scan per side to retain
the bounded window. For `I = I_expected + I_actual`, this is `Theta(I)` time
and `O(context)` result space; it never copies all payload bytes or rebuilds a
full input index merely to explain a mismatch. Equal canonical input count and
hash skip that scan in `O(1)` before downstream evidence localization.

`Diff::mismatch(i)` returns its stable field label as a borrowed
`string_view`. All labels are compile-time literals owned by the comparison
schema, so reading `M` mismatch rows performs `Theta(M)` field projections,
zero field allocations, and zero field-byte copies.
Window exposes all bounded context as paired borrowed spans:
`expected_observations()`/`actual_observations()`,
`expected_host_events()`/`actual_host_events()`,
`expected_inputs()`/`actual_inputs()`, and
`expected_trace()`/`actual_trace()`. These ranges have one owner and require no
caller-side reconstruction. Trace rows retain `TraceCode` plus snapshot
`ReasonCode`; `error()` and `snapshot_error()` derive stable text without a
per-row allocation, copy, or retained string owner.
Mismatch and range views remain valid while their Diff or Window value lives.

## Storage And Cost Evidence

`SessionConfig::replay.storage` has one public policy owner:

```cpp fragment
rund::storage::Budget application_storage{8ull * 1024ull * 1024ull * 1024ull};
rund::replay::Storage storage{
    .mode = rund::replay::StorageMode::Spill,
    .directory = replay_directory,
    .cached_bytes = 4u * 1024u * 1024u,
    .segment_bytes = 64u * 1024u * 1024u,
    .max_bytes = 1024u * 1024u * 1024u,
    .max_allocated_bytes = 2ull * 1024ull * 1024ull * 1024ull,
    .minimum_free_bytes = 256u * 1024u * 1024u,
    .budget = application_storage,
};
```

`Memory` requires an empty `directory`; only its nonzero `max_bytes` logical
bound applies. `Spill` requires a non-empty caller-owned root `directory`,
nonzero `segment_bytes`, and nonzero `max_allocated_bytes`. `cached_bytes` may
be zero to disable decoded spill caching. `max_bytes` remains the admitted
logical replay payload bound. The two limits are independent: compression may
make encoded storage smaller than logical storage, while segment framing and
filesystem allocation rounding make the conservative disk charge larger than
the encoded bytes.

For Spill, Session admission derives one child Budget whose capacity is

```text
min(storage.max_allocated_bytes, storage.budget.capacity_bytes)
```

when the application supplied a valid root. Sibling Sessions derived from the
same root therefore compete atomically for that root's remaining capacity. If
`budget` is left invalid, Session admission creates one private root with
capacity `max_allocated_bytes`; that isolates the Session rather than claiming
an application-wide or volume-wide limit. The application owns which other
in-process disk producers receive children of the same root. Hierarchy,
reservation, refund, and concurrency laws have one authority in
[Storage](./storage.md).

Spill has two distinct disk-byte metrics. For `E` unique encoded chunk bytes
and `K` stored chunks, every segment record has a fixed 41-byte header, so the
exact logical length of runD's segment files is:

```text
P = E + 41K
```

Before reserving and writing a record of `r` bytes, runD takes one `statvfs`
snapshot. That single observation owns both the allocation unit and available
space: `u = f_frsize` when nonzero, otherwise `u = f_bsize`, and
`V = f_bavail * u` with checked multiplication. `s` is the current logical
length of that target segment. With
`ceil_u(x) = u * ceil(x / u)`, it reserves the conservative increment:

```text
a = ceil_u(s + r) - ceil_u(s)
```

An append that stays inside the target segment's already charged final block
therefore reserves and commits `a = 0` while still adding its exact positive
`r` to the physical metric. Summed over one generation,

```text
A = sum over segments ceil_u(segment logical length)
```

and each accepted record commits `Usage{physical_bytes = r,
allocated_bytes = a}`. `A <= max_allocated_bytes` for one Session generation,
and the same `A` is charged through every ancestor Budget. `A` is a
reservation-safe upper-bound metric derived from the filesystem allocation
unit; it is not a query of the filesystem's observed block count. The metric
covers segment-file content only. Generation marker and lease files, directory
entries, filesystem metadata and journals, copy-on-write snapshots, and
replicas are outside both `P` and `A`.

`minimum_free_bytes` is evaluated from that same snapshot rather than issuing
a second capacity query. runD computes `V >= a + minimum_free_bytes`, acquires
the in-process Budget reservation, then consumes that recorded headroom
decision before writing. This preserves rejection precedence: a Budget
rejection is `HostPayloadCapacityExceeded`; otherwise an overflow, query
failure, or insufficient observed space is
`HostPayloadSpillWriteFailed`. It also halves filesystem-capacity observations
per unique chunk while preserving the exact allocation equation and one
admission state. The headroom remains advisory: unrelated writers can consume
space immediately after the snapshot. It is not a volume reservation or
quota.

The fixed 41-byte header is encoded once in a stack buffer and submitted with
the borrowed encoded payload by one positioned vectored write on the normal
path. Thus one accepted chunk performs `Theta(encoded bytes)` unavoidable
kernel transfer, `O(1)` temporary user-space storage, zero payload-sized
staging copies, and one filesystem-capacity query. A lazy read performs one
fixed-header read, one exact encoded-owner allocation, and one direct payload
read; validation and codec work remain `Theta(encoded bytes)`.

A Budget rejection maps to `HostPayloadCapacityExceeded` before the record
write. Allocation-unit or free-space query failure, insufficient observed
headroom, generation creation failure, and segment I/O failure map to
`HostPayloadSpillWriteFailed`. Store append is transactional across those
paths: failed or rolled-back work publishes no logical record. A successful
filesystem rollback releases its staged Reservations immediately. If file
cleanup itself fails, the backend is poisoned and keeps those charges owned
until final generation destruction instead of making possibly retained bytes
available for reuse. A rejection is never reclassified as a successful
checkpoint, History eviction, or diagnostic-only event.

`Record::storage_report()` publishes the cost of the immutable record:

| Field | Exact meaning |
| --- | --- |
| `logical_bytes` | Canonical uncompressed payload bytes referenced by records. |
| `encoded_bytes` | Unique encoded chunk bytes after deterministic deduplication and codec selection. |
| `retained_bytes` | Bytes retained by the immutable in-memory publication owner; zero for spill archives. |
| `copied_bytes` | Bytes copied to form this publication; a repeated publication of the same unchanged memory Store reports zero. |
| `cached_bytes` | Current backend memory retained for encoded memory chunks or decoded spill-cache entries. |
| `physical_bytes` | Exact Spill segment-file content length `P = E + 41K`; zero for Memory. |
| `allocated_bytes` | Committed conservative Spill charge `A`; zero for Memory. |
| `reserved_bytes` | Accepted but not yet committed Spill charge; an immutable Record has no in-progress append and reports zero. |
| `growths` | Saturating count of actual post-preparation container-capacity transitions in Store metadata, backend chunk/ref storage, and spill-cache indexing. |
| `chunk_count` | Unique physical chunks. |
| `segment_count` | Physical spill segments; zero for memory storage. |
| `cache_hits`, `cache_misses`, `cache_evictions` | Spill decoded-cache activity; zero for memory storage. |

For prepared memory execution with admitted bounds, `growths = 0`. A positive
value is diagnostic evidence that a container crossed its prepared capacity;
it is never synthesized from payload size. `copied_bytes` owns exact
publication byte copying and is deliberately separate: allocating the final
immutable `B`-byte owner is required for result lifetime but is not a
container growth. Counts saturate at `2^64 - 1` and never enter replay hashes.

The publication invariant is `retained_bytes = encoded_bytes` for a non-empty
memory archive and `retained_bytes = copied_bytes = 0` for spill. The first
memory publication may copy exactly `B = encoded_bytes`; unchanged repeated
publication reuses the exact owner and copies zero bytes. If prepared storage
retains capacity `C` while the result needs `B`, publication retains `B`, not
`C`. Thus `C >> B` cannot pin the Session scratch allocation in a returned
Record.

A Spill publication instead shares one immutable process-local generation
owner with its segment coordinates. A later Store append, clear, or
destruction cannot invalidate a previously returned Record. The generation
directory and its committed Budget charges remain live until the final Store,
archive, or Record owner releases them; normal final-owner destruction removes
that runD-owned generation when filesystem removal succeeds and refunds its
charges. Removal has no throwing result channel, so a filesystem failure can
leave a recognized stale directory after the in-process charge is released.
Abrupt process termination cannot run that destructor at all. The first
generation-creation attempt that successfully registers each normalized root
in a process performs one best-effort scan of recognized, unlocked stale
generations. That root is then remembered process-locally, so later generations
in the same process do not rescan it. The marker, lease, path-containment, and
deletion mechanics are owned by [Host](./host.md).

## Raw Capture

Canonical input remains the authoritative simulation boundary. A separately
bounded raw ingress capture exists only to reproduce decoder, framing, or
normalization defects before that boundary. `SessionConfig::replay.diagnostic`
sets its byte and record windows. A zero bound disables capture; there is no
unbounded mode.

`Record::captures()` returns one borrowed `std::span<const Capture>` over every
retained ingress. Its `size()` is the retained row count; each `Capture`
contains the host-event sequence, receive kind, content hash, and a borrowed
immutable byte span. `capture_hash()` identifies the complete retained archive,
while `capture_report()` reports retained bytes and rows plus evicted and
dropped rows. Both the Capture range and each byte span remain valid while the
Record evidence owner lives. Acquiring the range is `O(1)` and creates no
allocation or copy; iterating `R` rows is `Theta(R)`, and exporting `N` bytes is
`Theta(N)` only when the application writes them. Raw bytes never enter the
canonical input hash or alter Replay and Scenario selection.

The ingress owner chooses hash-only or hash-and-retain once before traversing
the completed prefix. Hash-only preserves the one StableHash byte pass.
Hash-and-retain visits every canonical slice once, copies it into the prepared
ring, then updates that same StableHash from the exact ring destination spans
in canonical wrap order. The returned hash is written directly into the host
event before Replay comparison; the ring accepts no caller-supplied parallel
hash. For `J` completed-prefix slices and `P` bytes, capture therefore costs
`Theta(J + P)` with exactly `J` source projections, exactly `P` source-byte
reads, zero warm allocation, `Omega(P)` hot ring hash reads, and `Omega(P)`
retained-byte writes. Disabled capture hashes outside the scheduler evidence
mutex. Enabled capture mutates the ring and hashes its destination while
host-event commit owns that mutex, preserving event/capture order at the cost
of an optional `Theta(P)` critical section.

## Checkpoint Schema And Lineage

Checkpoint state is domain-neutral bytes with one explicit nonzero schema and
one borrowed restore authority:

```cpp fragment
auto restore = [](std::span<const std::byte> bytes) {
  return restore_world(bytes) ? rund::replay::Restore::Restored
                              : rund::replay::Restore::Failed;
};
rund::replay::Binding replay{state_schema, restore};
auto saved = replay.checkpoint(record, bytes);
auto next = replay.advance(saved, continued, next_bytes);
```

`Checkpoint::schema()` exposes the stored schema and `Record::start_hash()`
exposes the boundary from which a segment began. Let canonical
domain-separated hash be `H`, and define

```text
G        = H("rund.replay.start", 0)
start(C) = H("rund.replay.start", C.hash)

state(S) = H("rund.replay.checkpoint.state",
             S.schema, |S.bytes|, S.bytes)
```

A fresh `record(session, step)` has `start_hash = G`. A continuation recorded
from checkpoint `C` has `start_hash = start(C)`. The Record hash commits
`start_hash` before semantic, operation, observation, host, input, transcript,
and trace identities. The checkpoint hash commits the state schema through
`state(S)` and keeps the existing checked segment, input-position, boundary,
prefix, and transcript-prefix chain.

`binding.checkpoint(record, bytes)` accepts only a successful record whose
start is `G`. `binding.advance(C, record, bytes)` accepts only a successful
record whose start is `start(C)`. `binding.append(history, record, bytes)`
applies the same rule. This prevents an
independent or differently restored segment from being spliced into a valid
chain merely because its local evidence is valid.

A continuation declares the application state schema and one borrowed restore
lvalue exactly once, uses that same binding to form snapshots, and binds the
immutable checkpoint without restating either authority:

```cpp fragment
rund::replay::Binding replay{state_schema, restore};
auto saved = replay.checkpoint(record, bytes);
auto resume = replay.resume(saved);
auto next = resume.record(session, step);
auto check = resume.run(session, expected, step);
auto branch = resume.scenario(session, expected, choices, step);
```

`Binding` accepts only an lvalue restore callback and is the sole schema,
restore, checkpoint, resume, and History-publication authority.
`Binding{}` is a valid input-only binding; its checkpoint, advance, resume,
and append paths reject with `StateSchemaInvalid`. A stateful binding requires
a nonzero schema and valid borrowed restore callback before it can create a
Channel or a state boundary, so one malformed policy cannot partially run.
`Binding::resume(checkpoint)` validates that binding against the checkpoint
once and returns `Resume`. An rvalue restore is rejected at compile time, so
the binding cannot retain a temporary.
The restore callback receives only the checkpoint-owned immutable byte span and
returns `Restore::Restored` or `Restore::Failed`; schema is bound once and is
neither repeated per execution nor passed back for callback interpretation. A bad
binding, failed restore result, or exception prevents the simulation callback.
For strict Replay and Scenario, expected-record start lineage is validated
before restore. `Resume::record`, `run`, and `scenario` all enter the same
internal execution authority; there is no direct Checkpoint execution adapter.
Binding is `Theta(1)`: it copies the checkpoint's shared immutable owner and
stores one object pointer, one function pointer, and one validated Code. It
allocates no memory and copies no checkpoint bytes. Each execution therefore
restores from the same immutable span without rebuilding schema or callback
adapters.

## Rolling History

`History` assigns one checked, strictly increasing sequence to every accepted
segment. Eviction removes only the oldest segment, so the retained identities
are always the contiguous suffix

```text
[oldest, oldest + size)
```

and never a sparse set. `find(sequence)` therefore performs no scan or binary
search. For `d = sequence - oldest`, checked only after
`sequence >= oldest`, `d >= size` is absent; otherwise the ring slot is

```text
slot = head + d                  when d < capacity - head
slot = d - (capacity - head)     otherwise
```

This is `O(1)` time and `O(1)` additional space, independent of retention
length. The implementation retains an exact stored-sequence equality guard so
an invariant violation fails closed instead of returning the wrong segment.
Sequence overflow rejects append; it cannot wrap and alias an older segment.

History retention does not choose application checkpoint policy. The
application owns when to capture a checkpoint, the state schema and bytes,
which Record/checkpoint pairs to append, and any durable retention cadence.
Storage pressure does not synthesize a checkpoint, change that cadence, or
evict a History segment early. A record or History append that cannot satisfy
its configured logical, allocated, or retention bound fails with its typed
capacity result.

## Persistence

`Record::save(sink)` and `Checkpoint::save(sink)` are the sole write boundary.
The sink is called synchronously with ordered spans of at most 4096 bytes and
returns `true` only after consuming the complete span. `Save::bytes()` and
`Save::writes()` report successfully accepted output. Sink rejection or an
exception becomes `ArtifactWriteFailed`; no exception crosses `save()`.
Because an earlier span may already have reached the sink, durable callers
write a temporary artifact and publish it only when `Save` succeeds.

The internal artifact Writer is the sole save-failure state authority. The
first non-`Ok` code wins and later writes become inert: an invalid sink reports
`ArtifactOutputInvalid`, an admitted-byte or write-count overflow reports
`ArtifactCapacityExceeded`, sink rejection reports `ArtifactWriteFailed`,
missing spill payload reports `ArtifactPayloadMissing`, and malformed
in-memory evidence reports `CodecInvariantInvalid`. Every semantic encoder
closes a false result through that authority; a partial artifact can never
finish with `Code::Ok`. The sink's internal `max_bytes` admission is checked
before each callback, so capacity rejection neither calls the sink nor counts
the rejected span. This state machine does not preflight or traverse evidence
a second time.

Save emits one compact binary schema. Every artifact starts with the six-byte
binary magic `72 75 6e 44 1a 0a`, schema byte `1`, and one kind byte. Fixed
64-bit identities are little-endian. Counts and nonnegative deltas use the
shortest unsigned varint; signed fields use zigzag encoding. Load rejects a
non-minimal varint, an unknown schema or kind, overflow, truncation, and
trailing bytes. There is no alternate persistence grammar.

Internal host, accelerator, and Kernel evidence encoders that target a
contiguous byte vector all use one artifact append callback. It is the sole
owner of vector growth failure translation: a complete span is appended or
the callback returns `false`, with no exception crossing the artifact writer.
The three schemas keep separate field-order owners; only their identical sink
mechanism is shared.

All Replay byte, range, sequence, retention, and spill-size composition
consumes the Kernel [checked arithmetic](../../../kernel/docs/contracts/checked.md)
owner directly. Replay has no private arithmetic namespace, header, overflow
predicate, or duplicate boundary test. Addition succeeds iff
`left <= UINT64_MAX - right`; multiplication succeeds iff `right == 0` or
`left <= UINT64_MAX / right`; subtraction succeeds iff `value >= base`.
Callers consume either the checked value or the negated success predicate, so
decode, validation, storage, and spill paths cannot drift on wraparound
semantics. Delta encoders reject descending sequence, time, or range
identities, and their decoders reject any addition that would wrap. Payload
chunk references use the schema-1 modular contract: their delta is the canonical
representative in `Z/(2^64)`, because deduplication may refer to an earlier
chunk. Decode applies the inverse modular projection and then admits the result
against `archive.chunks.size()`. This preserves the one established grammar
without treating accidental overflow as valid range arithmetic. Writer
fail-closed state changes do not alter this algebra, serialized field order, or
valid schema-1 bytes. The functions are inline, stateless, allocation-free, and
introduce no extra data traversal.

The bounded raw-capture ring advances offsets with one overflow-free wrap
projection. For `0 <= base < capacity` and `0 <= offset <= capacity`, it uses
`offset >= capacity - base ? offset - (capacity - base) : base + offset`.
The two branches are algebraically equal to `(base + offset) mod capacity`,
but the chosen expression cannot overflow and does not require integer
division.

The schema stores semantic source values once and derives redundant values
during load before validating the final identity. Observation deadlines,
observation sequence, host logical time, host sequence, and trace sequence are
delta encoded. Host fields use a presence bitmap and omit zero values.
Payload chunks carry Raw or RLE bytes directly. Payload records use an
exception bitmap: unchanged source and schema, consecutive sequence,
contiguous empty source ranges, one unchanged chunk, completed byte count,
source hash, payload hash, and host kind are omitted whenever they are
derivable. An exceptional multi-piece row writes its aggregate payload hash
before its piece count and chunk deltas; load consumes that same canonical
order. Checkpoint state bytes are emitted directly and its derived boundary and
prefix identities are reconstructed during load.

The internal payload record has one semantic value authority in
`node/include/node/runtime/replay/host/record.hpp`. `ArchiveRecord` composes that
`Record` with serialized piece references, `StoredRecord` composes it with the
record hash and a bounded piece range, and `MaterializedRecord` composes it with
owned decoded bytes. Conversion copies or moves the `Record` once as a value;
none of the three representations mirrors its thirteen fields. Serialization
and record hashing read that same value in the frozen schema-1 order, so this
boundary preserves artifact bytes, record hashes, source-range hashes, and
Replay diff identity. It adds no installed SDK type or public projection.

The Store's `Binding` and `InputBinding` remain separate execution-request
views. They validate a host operation or canonical input lookup respectively;
they do not own persisted role, source-range, byte-storage, or full record
identity. Merging either view with `Record` would add irrelevant state to the
hot lookup boundary and create a second construction authority, so no
conversion accessor or field forwarding exists.

Save never materializes a second artifact or a payload-sized staging value.
For a memory-backed Record it adds 4096 bytes of fixed writer storage beyond
the already retained archive. For a spill-backed Record it adds that writer
plus at most one loaded encoded chunk, whose hard bound is 64 KiB. Thus
save-side transient payload memory is

```text
memory: 4096 bytes
spill:  4096 + max(chunk.encoded_bytes) <= 69632 bytes
```

and not `B + ceil(4B / 3)` for encoded payload size `B`. A spill artifact
embeds every encoded chunk and normalizes its persisted storage projection to
Memory. It therefore remains loadable and replayable after the caller-owned
spill directory is removed; segment paths and cache state are never artifact
dependencies.

The sink and every byte it accepts are caller-owned persistence. Replay's
Budget does not account for temporary or published artifacts, replicas,
snapshots, remote copies, or sink-side filesystem metadata. Applications that
need one aggregate limit must account those producers explicitly under their
own shared root and must choose artifact naming, atomic publication, cleanup,
replication, and retention policy. A successful `save()` proves only that the
sink accepted the ordered spans; it does not prove durability, replication, or
remaining free space.

`Record::load(bytes, limits)` and `Checkpoint::load(bytes, limits)` are the
sole read boundary. `Load<T>` owns the typed result and `Code`; the caller owns
the contiguous artifact bytes for the duration of the call. Load is
`noexcept`, enforces `Limits` before publication, catches allocation failure,
and recomputes every identity before exposing an immutable value. Successful
loads use the same generic `*loaded` and `loaded->` access for Record and
Checkpoint. There are no type-specific `record()` or `checkpoint()` result
adapters and no duplicate load state machines.

Record and Checkpoint each accept exactly their binary kind and schema. Load
walks the borrowed artifact once, admits declared entry and byte counts before
allocating, copies each retained payload or checkpoint byte range directly
into its final immutable owner once, reconstructs derived values, and
recomputes the complete identity before publication. No parsed field table,
second payload vector, or second decode state machine exists.

For a memory payload with `K` chunks and `B` unique encoded bytes, Record load
records each borrowed range's artifact offset in the already admitted chunk
metadata, then allocates one exact `B`-byte owner and copies those ranges into
consecutive slices. The unpublished parse offset is cleared before the archive
is published. There is no second `K`-element range table and no `K` byte-owner
allocation: auxiliary payload-load space beyond the final archive and owner is
`Theta(1)`. The first cold replay preparation adopts those same slices, so the
loaded Record and prepared Store share one `B`-byte allocation and preparation
copies zero payload bytes.

Memory-archive validation hashes and verifies each unique chunk's decoded bytes
once. A repeated whole-chunk record uses the already verified chunk identity and
does not walk its bytes again. A repeated chunk inside a multi-piece record is
decoded only into that record's aggregate byte hash; it does not recompute the
chunk hash. If `D_u` is the sum of unique decoded chunk bytes and `D_a` is the
sum of repeated decoded bytes required by multi-piece aggregate hashes, byte
work is `Theta(D_u + D_a)`, plus `Theta(K + P + R)` metadata work for `P`
piece references and `R` records. For the usual one-piece event/input record,
`D_a = 0`; validation scales with unique physical bytes rather than logical
bytes repeated through deduplication. Spill validation remains metadata-only.

Default `Limits` admit at most 2 GiB of artifact bytes, 4,194,304 aggregate
entries, 1 GiB of payload bytes, and 64 MiB of checkpoint state bytes. Callers
may lower any bound per load. Every declared count contributes to the single
`max_entries` aggregate before its destination allocation.

The checked 30-minute compact-input model records 18,000 canonical input
events at 10 events per second using a repeated 16-value, one-byte command
alphabet. It saves to 64,394 bytes, which is 38,006 bytes or 37.1% below the
100 KiB limit, and the same contract test loads and validates all 18,000
events. This is an evidence-backed workload bound, not a universal bound:
unique high-entropy payload bytes still require proportional storage because
no lossless codec can encode arbitrary information below its entropy.

Trace persistence is numeric and canonical. The root writes `trace.code` as
one validated `ReasonCode`. Each record writes `event`, then `code.domain` and
`code.value`, followed by snapshot state, active-job count, scope flag,
`snapshot.code`, and sequence. Trace hashing mixes that same fixed order:
Runtime/Compute kind and exact 16-bit value are separate words. No reason text,
pointer address, allocation shape, or locale participates in the artifact or
hash. Trace and replay identity are derived exclusively from this numeric
canonical form.

The hash is deterministic integrity evidence, not an authentication signature.
Untrusted persistent data still requires an application-owned authenticated
transport or signature.

## Failures And Telemetry

`rund::replay::Code` is the sole Replay outcome authority. Every public Replay
result exposes `code()`. `Code::Ok` is the only success value; `ok()`, truth
conversion, `error()`, and `exit_code()` derive from it. `error()` is empty for
`Ok` and otherwise returns the stable Code-to-text projection. No public or
facade result stores, accepts, or compares a raw reason pointer or string.
The free `exit_code(Code)` function is the sole process-status formula and
every result member delegates to it. `Live`, `Record`, `Check`, and `Scenario`
use this identical completion UX; `Live` additionally preserves the ordinary
scope, task, memory, observation, event, and trace evidence without exposing a
second `Session::Result` authority.

Codes retain the admitting boundary instead of collapsing failures into a
generic error. Result containers for `Record`, `Diff`, `Window`, and
`Checkpoint` hold failure codes without allocating. Public result publication,
comparison, minimization, and checkpoint state adoption catch allocation
failure and return `AllocationFailed`; an exception is never the fallback
outcome channel:

`node/src/runtime/replay/exception.hpp` is the sole C++ exception-type
projection for Replay `noexcept` boundaries. Each boundary supplies its three
semantic outcomes for `bad_alloc`, `length_error`, and an unexpected exception;
the common projection inspects the active exception once and selects exactly
one of those typed `Code` values. Checkpoint load, Record codec load, and
Scenario preparation therefore keep their own domain-specific codes without
repeating or drifting three independent catch ladders. Calling the projection
without an active exception fails closed to that boundary's unexpected code.

| Family | Required distinctions |
| --- | --- |
| Lifecycle | unopened, busy, reentry, not prepared, moved value, expired scope |
| Input | id, schema, missing input, corrupt input, writer state, commit, uncommitted source |
| Load | envelope, schema, kind, canonical integer, field, invariant, and immutable-value adoption |
| Capacity | encoded bytes, entries, input bytes, state bytes, payload bytes, index slots, retention weight, and arithmetic overflow |
| State and order | checkpoint state, lineage, start identity, input sequence, choice duplicate, choice ambiguity, and position overflow |
| Integrity | state, boundary, prefix, operation, observation, host event, input, transcript, trace, and complete record hash |
| Callback | callback failure, callback not run, restore rejection, restore failure, and restore exception |
| Storage | invalid policy, unavailable backend, spill failure, missing payload, payload mismatch, allocation failure, and retention rejection |
| Comparison | outcome and each canonical evidence mismatch without parsing field text |

When a Session failure crosses the Replay facade, its exact `ReasonCode` maps
bijectively into the reserved Runtime range of `replay::Code`. The mapping and
stable text derive from the Node reason schema; Replay does not copy that
schema or replace an exact reason with `RuntimeFailure`. Replay-native codes
derive from one ordered code schema. Code family is a derived classification,
never a second stored status.

The 32-bit numeric layout uses disjoint 16-bit domains:

```text
raw(Code) = (domain << 16) | value
```

`Code::Ok` is zero. Replay, Runtime, host artifact, Accel artifact, and Kernel
artifact failures have distinct nonzero domains, and each domain value is
below `2^16`.
The mapping is injective: equality of two raw codes implies equality of their
upper 16-bit domains and lower 16-bit values. Runtime's lower value is the
exact underlying `ReasonCode` value, so conversion is lossless and requires no
lookup or allocation. Load rejects a domain or value absent from the ordered
schema.
Unknown persisted numeric codes and known codes presented through the wrong
domain loader fail with that loader's typed code before an immutable value
is published. Tests compare enum values; stable text remains a derived human
projection and never participates in control flow or canonical hashes.

Replay telemetry reports Session id, scope id, mode, built/reused plan state,
canonical input rows and bytes, source-emitted rows, choice count, evidence
rows and bytes, retained bytes, physical publication copies, prepared-storage
growths, exact Spill file bytes, conservative allocated-budget bytes, in-flight
reserved bytes, typed result code, and published record hash when present.
Error text derives from that code. Detail timing covers the complete public
verb from validation and plan work through immutable result publication.
Telemetry callbacks run outside runtime locks, cannot
reenter the same Session, and cannot affect ordering, admission, hashes, or
success. The sole detailed-observation command is `telemetry:detail`.

## Implementation Authority

The implementation cut is owned by these paths:

- public entry composition: `node/include/rund/replay.hpp`;
- single declaration owners: `node/include/rund/replay/{input,record,state,
  history,run}.hpp`, with `binding.hpp`, `code.hpp`, `limits.hpp`, and
  `storage.hpp` as their leaves;
- Session admission: `node/include/rund/session.hpp`,
  `node/include/rund/session/config.hpp`, `node/src/runtime/session.cpp`,
  `node/src/runtime/task/scope/frame.cpp`, and
  `node/src/runtime/runtime/scope/`;
- replay product surface and values: `node/src/runtime/replay/surface/`,
  `node/src/runtime/replay/binding.cpp`,
  `node/src/runtime/replay/record/`, `node/src/runtime/replay/checkpoint.cpp`,
  `node/src/runtime/replay/history.cpp`, `node/src/runtime/replay/hash.cpp`,
  `node/src/runtime/replay/exception.hpp` for the one exception-type-to-Code
  projection, and `node/src/runtime/replay/codec/`;
- transactional input and admission plan: `node/src/runtime/replay/input/` and
  `node/src/runtime/replay/scope/`;
- scope identity and evidence: `node/src/runtime/task/scheduler/core/identity.cpp`,
  `node/src/runtime/task/scheduler/core/snapshot.cpp`,
  `node/src/runtime/task/scheduler/core/host.cpp` for the narrow active-host
  bridge, `node/src/runtime/task/scheduler/core/replay.cpp` for canonical input
  capture and replay,
  `node/src/runtime/task/scheduler/core/record/`,
  `node/src/runtime/task/scheduler/progress/scope.cpp`, and
  `node/src/runtime/task/scheduler/state/storage/`;
- persistence and payload projection: `node/src/runtime/replay/host/payload/`
  and `node/include/node/runtime/replay/`; the canonical internal payload-record
  semantics are owned only by
  `node/include/node/runtime/replay/host/record.hpp`.

`Runtime` configuration prepares replay capacity. One Session scope admission
owner selects execution, installs expected evidence, captures canonical input,
and publishes the terminal result for ordinary and replay scopes alike.

The source-private product surface is partitioned by responsibility:
`surface/input.cpp` owns canonical input routing, `record.cpp` immutable result
projection and persistence, `compare.cpp` comparison views, `scenario.cpp`
choice admission, `run.cpp` ordinary scope execution, `resume.cpp` restored
execution, `state.cpp` checkpoint construction, and `telemetry.cpp` terminal
observation. `surface/data.hpp` owns the shared private value layout;
`surface/local.hpp` adds only execution helper declarations for the owners
that require them.

Persistence is partitioned without duplicating its grammar:
`codec/value.cpp` owns observation and trace pairs, `payload.cpp` owns both
directions of the payload grammar and the one-owner load projection,
`save.cpp` owns the Record write entry, and `load.cpp` owns the bounded Record
read entry and final identity validation. `codec/local.hpp` is the single
private owner of admission arithmetic and payload-record field bits.

Strict Replay and Scenario, including their resumed forms, call one
`Access::expected` authority. It preserves the fixed admission order:
Record readiness, start lineage, Session input capacity, then immutable
expected-plan reuse or construction. Resume validates its own binding before
that sequence and restores state only after expected and Scenario choice
admission. All rejection paths publish through one typed terminal per result
kind, so refactoring cannot give telemetry a different failure code or order.

Scenario retains the required exact `B`-byte owner and `P` patch rows. Each
choice hashes its caller-owned immutable span before copying it once into that
owner; the hash is therefore identical to the canonical source bytes while the
destination is never rescanned. Mismatch-window public projection releases
its private input and trace vectors after conversion, so the result does not
retain two representations of those rows.

## Verification

P0 closes only when the following contracts pass on the same source manifest:

1. one open Session records and strictly replays the same callback; the Record
   and actual hashes match, the Record source runs once, and Replay runs it
   zero times;
2. Scenario reuses that Session, applies only admitted choices, and invokes
   every source zero times;
3. repeated scopes have different physical identities but identical canonical
   evidence and hashes; a close followed by physical descriptor reuse receives
   a new canonical handle, while Record and Replay with different physical
   handles still compare equal;
4. every task statistic and trace record is a scope-local delta;
5. 100 warm scopes cause no scheduler/evidence capacity growth or callback-time
   runtime allocation after prepared bounds; result-owned retained bytes are
   reported separately;
6. busy, reentry, callback exception, prepare failure, and restore failure all
   leave the next Live scope usable;
7. wrong checkpoint/record start fails before restore and callback;
8. zero or mismatched state schema fails before restore, and schema changes the
   state and checkpoint hash;
9. Record and Checkpoint canonical save/load round trips preserve
   start/schema/lineage; load rejects every value missing `start_hash` or
   `state_schema`; sink rejection and sink exceptions return
   `ArtifactWriteFailed` without throwing;
10. a spill Record saves through spans no larger than 4096 bytes, loads after
    its spill directory is removed, and replays without invoking its source;
11. genesis checkpoint, advance, History append, eviction, loaded anchor, and
    continuation enforce the same start relation;
12. the installed SDK consumer performs record, strict replay, Scenario,
    checkpoint persistence, and continuation through one open Session and the
    `rund::replay` namespace;
13. `Input` contains only id and schema; the bound source returns sequence once
    in Live/Record, while `Value`, `Choice`, and immutable evidence expose it.
    Record infers the produced byte size,
    Replay uses the stored size, and a Scenario choice may replace it with a
    different admitted size without invoking a source;
14. Choice is constructible only through `Channel::choice`, borrows application
    bytes only through admission, and the frozen plan
    retains one exact `B`-byte owner, does not alias the application buffer,
    and remains unchanged after that buffer is mutated;
15. the installed-SDK current-surface contract accepts Binding/Channel input
    and state ownership while rejecting rvalue source/restore lifetimes,
    void-return sources, caller-supplied read sequence, direct Choice
    construction, and direct Checkpoint execution; the surface equals the
    declared namespace, verbs, canonical rows, and direct-header registry;
16. checked base projection and bounded handle admission fail with
    `replay_scope_identity_invalid`, publish no post-failure evidence, and the
    prepared handle table performs one configuration allocation and zero warm
    scope allocations;
17. every Replay outcome has exactly one `Code`, all runtime mappings are
    total and bijective, stable text is derived, unknown persisted codes fail
    before publication, and source/tests contain no stored or compared outcome
    reason strings. Trace root and snapshot retain exact `ReasonCode`; each
    record retains one exact Runtime-or-Compute `TraceCode`, and persistence
    rejects unknown domains or values before publication. The source-private
    exception projection distinguishes allocation, length, and unexpected
    exceptions according to the calling Replay boundary's supplied codes;
18. input mismatch windows identify id, schema, sequence, byte length, and
    hash while retaining only bounded context; raw diagnostic captures expose
    exact immutable bytes after Record load without becoming simulation
    input or a second outcome authority;
19. generic Storage contracts prove ancestor-atomic capacity admission,
    reserve/commit/refund arithmetic, move-only exactly-once release, reporting,
    and concurrent sibling safety;
20. Spill reports exact `P = E + 41K`, commits the documented conservative
    allocated increment, never exceeds its per-Session child or shared root,
    refunds failed and rolled-back writes, and rejects a capacity boundary
    without accepting the logical Record;
21. a Spill Record keeps its immutable generation readable after its source
    Store and Session are released, normal final-owner release removes only
    that generation and refunds its Budget charge, and the single first-use
    scan per normalized root removes an unlocked marked stale generation
    without rescanning later or touching live, unmarked, and unrelated root
    entries;
22. filesystem headroom success, insufficient-space, overflow, and query
    failure paths preserve Budget and Store invariants, while evidence names
    the free-space check as advisory rather than volume-global authority.

Focused ownership belongs in `node/tests/contract/runtime/task/replay/` with a
dedicated reusable-Session case, plus storage configuration, Spill accounting
and generation lifetime, checkpoint, history, codec, negative,
evidence-capacity, and mismatch cases. Generic Budget ownership belongs in
`node/tests/contract/runtime/storage.cpp`. Installed coverage belongs in
`package/tests/consumer/example/replay.cpp`,
`package/tests/consumer/example/scenario.cpp`,
`package/tests/consumer/example/checkpoint.cpp`,
`package/tests/consumer/example/history.cpp`, `package/tests/consumer/sdk.cpp`,
and `package/tests/consumer/blackbox.cpp`. Every documented example fence is
byte-identical to its compiled source.

The payload Store contract keeps one registry symbol in `payload/store/suite.cpp`.
Its fixed first-failure sequence is `memory`, `archive`, `publish`, `input`,
`diagnostic`, then `materialization`; every semantic leaf creates its own Store
and byte owners, so its result has no dependency on an earlier leaf.
`payload/store/local/model.cpp` is the sole compiled owner of fixture
construction, temporary payload construction, the canonical archive-identity
oracle, and the ordered runner; `model.hpp` declares that boundary without
defining another implementation. Editing one semantic leaf therefore compiles
that leaf and relinks the existing Runtime contract executable. Editing the
shared model intentionally invalidates its consumers.

The reusable-Session run contract keeps `CheckReplayRunContract` as its one
registry entry and delegates in the fixed order `capacity`, `surface`,
`scenario`, `lifetime`, then `history`. `run/local/model.cpp` is the sole
compiled owner of its borrowed restore/source callbacks and shared live
Session state; the semantic leaves own no alternate runner or fixture.
Assertions and results follow that canonical order, and a leaf-only edit
dirties one compile plus the Runtime executable relink.

The Spill segment contract likewise keeps `RunSegmentsContract` as its one
ordered runner: `generation`, `lifetime`, `budget`, `append`, `artifact`, then
`layout`. `spill/local/model.cpp` alone owns temporary generation paths,
storage construction, segment accounting, the canonical segment-byte oracle,
and that runner. Each `spill/segments/` leaf owns one semantic boundary, so
changing a boundary does not recompile the other five contracts.

The P0 product cut consists of reusable Session execution, transactional
scope plans, scope-local canonical evidence, prepared immutable Record
projection, state schema binding, start lineage, current-schema persistence,
the namespace authority, inferred input length, variable-size Scenario
choices, typed Replay outcomes, installed consumers, and one documentation
authority. A wider or authenticated persistent identity is a separate product
change outside this verified scope and remains unclaimed.
