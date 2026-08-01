# Public API Authority

The checked-in direct-header surface is the sole SDK source authority. Every
public name and contract is defined by that current surface. Direct-header
ownership is defined by [`../surface.md`](../surface.md).

## Direct Headers

- `<rund/rund.hpp>`
- `<rund/session.hpp>`
- `<rund/task.hpp>`
- `<rund/host.hpp>`
- `<rund/net.hpp>`
- `<rund/replay.hpp>`
- `<rund/storage.hpp>`
- `<rund/evidence.hpp>`
- `<rund/compute.hpp>`
- `<rund/compute/pipeline.hpp>`
- `<rund/compute/async.hpp>`
- `<rund/compute/math.hpp>`
- `<rund/compute/session.hpp>`
- `<math32/math32.hpp>`
- `<math64/math64.hpp>`
- `<cluster/cluster.hpp>`

Support-only transitive headers are not consumer API. Downstream projects must
not include Kernel, Accel, direct Node, private, tooling, test, replay-detail,
or platform SDK headers directly. Product names are reached through their
focused `rund` entry or `<rund/rund.hpp>`; support-header path stability is not
implied. Cooperative tasks are owned only by `rund::task`; task implementation
namespaces are not product namespaces. Session evidence is owned by `rund`,
Session-host Compute operations and outcomes by `rund::compute`, host evidence
by `rund::host`, network operations and outcomes by `rund::net`, Replay
storage policy and evidence by `rund::replay`, and shared hierarchical byte
admission by `rund::storage`. Scheduler and Runtime implementation names
remain private.
`rund::ReasonCode` is the one shared typed runtime failure authority.
Physical presence is not a versioned-entry promise: C++ may
allow a consumer to spell the path of a staged support header, but only the
direct headers above are versioned SDK entries.

`<rund/task.hpp>` is a declaration-free composition of the task owner headers.
It introduces no forward-declaration mirror: `Handle`, `Status`, `Task`, and
the spawn/scope verbs each have one declaration authority in their owning task
header. Template-to-compiled bridges and bounded task storage have one
non-product authority under `rund::detail::task`; the installed-package gate
rejects any public header that declares or references implementation-only task
authority or includes an implementation header.
Scheduler operation identity is likewise non-product. Session evidence exposes
only the documented observation records and the derived task trace hash.

`task::Task<T>` is a move-only coroutine return and RAII value before
admission, not a public coroutine-frame capability. Consumers cannot name its
raw handle type, construct it from a raw handle, inspect or release its frame,
or call frame destruction. The C++ coroutine protocol still reaches the one
`rund::detail::task` promise implementation through `Task<T>::promise_type`;
no promise-support type is a product name. Admission transfers the frame once
to the scheduler, and exactly one of scheduler retirement or failed-admission
cleanup destroys it.

## Storage

`<rund/storage.hpp>` owns copyable shared `storage::Budget` values and the
move-only `storage::Reservation` refund obligation. The short operations are
`child`, `reserve`, `commit`, `refund`, and `report`. `storage::Usage` keeps
physical and allocated bytes distinct; `storage::Report` exposes coherent
current, available, peak, reservation, commit, refund, and rejection evidence.
Every Budget capacity is denominated only in allocated bytes; physical usage
is producer evidence and need not be ordered relative to it. Every failure
uses `rund::ReasonCode`; there is no storage-local code enum, allocator, pool,
TTL, eviction callback, or background owner. Exact hierarchy and lifetime
semantics are owned by the
[Storage contract](../../../node/docs/contracts/storage.md).

## Telemetry

`<rund/session.hpp>` exposes `rund::telemetry::{Sink, Level, Source, Mode,
Preparation, Compute, Replay, Detail, Event, Cost, Unit, Accuracy, Reference,
Cause, Action, Finding, Findings}` plus `bind`, `members`, `name`, and
`describe`. A configured Sink can be created only by binding an lvalue
observer; its callback, context, and level storage are private. `describe`
streams the bounded findings through a borrowed writer without allocating or
retaining text. One Sink and one nested Event shape remain the complete
telemetry authority. Selection, event meaning, emission order, and parity are
owned only by the
[Telemetry contract](../../../node/docs/contracts/telemetry.md).

## Session And Replay

`<rund/session.hpp>` owns Session lifecycle and telemetry configuration.
It declares the resident Compute admission template using incomplete product
types but does not import the Compute execution surface. The definition and
the `rund::compute::{Request,Submission,Poll,Completion}` product sequence are
reached through `<rund/compute/session.hpp>` by code that actually calls
`Session::compute`; the coroutine bridge is nested as `Request::Awaiter`, and
there is no Session-side fallback definition or root awaiter vocabulary. The
transitive `rund/compute/session/{request,submission,poll,completion,await}.hpp`
hierarchy provides one physical owner per declaration or bridge, but those
support leaves are not direct SDK entries.
`SessionConfig::workers` is the complete public worker-selection input;
Session constructs and owns its worker resources internally.
`<rund/replay.hpp>` is the sole replay consumer entry and a declaration-free
composition of the exact support owners for input, record evidence, state,
history, storage policy, and execution. Replay `Storage` accepts one
`rund::storage::Budget` but does not create a second reservation state machine.
The support paths are producer dependency boundaries, not direct consumer
entries, and this page does not maintain a second type inventory. Its
execution verbs are
`live`, `record`, `run`, and `scenario`; evidence operations are `check`,
`diff`, and `window`. `Binding` owns `checkpoint`, `advance`, `resume`, and
History `append`. Record and Checkpoint own their
streaming `save` and bounded `load` methods, and every fallible Replay value
exposes `rund::replay::Code` directly. `Load<T>` uses `*loaded` and
`loaded->member()` for both supported artifact types; type-specific load-value
adapters are not part of the surface.

Live, Record, Replay, and Scenario use one already-open Session and the same
callback shape. Applications create one `Binding`, combine each
`Input{id, schema}` and borrowed source into one `Channel`, and call
`channel.read(context)`. The source returns canonical sequence in Live and
Record; Replay and Scenario recover that sequence from the transcript and
publish it through `Value::sequence()`. Checkpoint continuation creates one
`Binding{schema, restore_lvalue}`, captures through
`binding.checkpoint(record, bytes)`, and creates
`Resume = binding.resume(checkpoint)`, then selects `record`, `run`, or
`scenario` without repeating schema or restore. No parallel input
identity, Context-owned bind, aggregate Choice, direct Checkpoint execution,
per-verb Session configuration, caller mode branch, byte-count argument,
source-owned byte return, root replay facade, or prefixed persistence verb is
part of the surface.

Every terminal replay value exposes the same direct completion shape:
`code()` is the sole outcome authority, truth conversion and `ok()` identify
success, `error()` is empty on success, and `exit_code()` is `0` or `1`.
Installed examples return that exact projection for every product failure;
process exit `2` denotes only an application assertion evaluated after all
product outcomes succeeded.
Replay semantics, source counts, ordering, identity, persistence, retention,
and failure behavior are owned only by the
[Replay contract](../../../node/docs/contracts/replay.md).
`Record::captures()` is the bounded pre-canonical debugging surface: one
borrowed `std::span<const Capture>`, whose elements borrow immutable raw ingress
bytes while the Record lives. `Window` localizes a canonical input mismatch
with id, schema, sequence, byte length, and hash. Its expected/actual
context is exposed only through the paired spans `expected_observations()` and
`actual_observations()`, `expected_host_events()` and `actual_host_events()`,
`expected_inputs()` and `actual_inputs()`, and `expected_trace()` and
`actual_trace()`. Neither surface creates a second simulation input path.
Diff field labels remain borrowed `string_view` values. Window trace rows own
one `TraceCode` and one snapshot `ReasonCode`; `error()` and
`snapshot_error()` are stable derived projections, not stored strings. Reading
these ranges performs no per-row allocation; retain the Record, Diff, or Window
while using its spans.

## Session Evidence

`Session::Result` is the one move-only completion value returned by one-shot
run and reusable Session scopes. It exposes `code()`, `memory()`, `tasks()`,
`observations()`, `events()`, `trace()`, and derived `trace_hash()`; there is no
parallel per-scope result, compact completion value, or public payload-archive
view. `tasks()` returns `const task::Stats&`, and `observations()` returns
`std::span<const task::Observation>`; both are borrowed from the owning result.

Prepared-memory evidence uses one `rund::PreparedMemory` with a
nested `Capacity` proof. Capacity owns one `rund::ReasonCode code` and derives every
success or diagnostic observer from it; no independent boolean or borrowed
reason pointer is part of the SDK. Its `valid()` observer and the recording
boundary share the single reason-schema category: only `Ok` and defined
prepared-memory failure reasons are admissible. `rund::record_memory`
is its sole writer, succeeds only from the root/sequencer context, and rejects
task-worker mutation. `task::Stats` carries no partial prepared-memory
mirror. No public projection function accepts internal Kernel telemetry.

## Task, Host, And Network Outcomes

Task, channel, host, and network outcomes keep one `rund::ReasonCode` as their
completion authority. `task::Status`, `task::Result<T>`, `task::YieldOp`,
`task::SleepOp`, `task::IoResult`, `task::IoOp`, and
`task::ReceiveResult<T>` expose truth conversion, `ok()`,
`code()`, `error()`, and `exit_code()` from that same value. Value-bearing
results add dereference and arrow operators but no second success flag or
failure owner. Awaiting a status-valued
`task::Task<task::Status>` returns that direct status; await failure and its
returned status are flattened instead of exposing
`task::Result<task::Status>`. Operation-specific records may add progress,
readiness, or native-error fields; those fields do not replace the code.
The concrete `task::Result<T>` and `compute::Result<T>` families share one
generic value-or-failure implementation. Their domain Status policies select
`TaskFailed` and
`compute::Reason::ValueInvalid`, respectively, if a throwing user move leaves
the active alternative invalid. Checked code, reason, error, pointer, and exit
observers remain total in that state. Move exception specifications derive
from the complete storage operation, including cross-alternative value
construction, rather than from value assignment alone.

`rund::net::Socket` is the sole socket-lifetime capability. It is move-only;
open and accept transfer it once, destruction retires its generation and makes
one native close attempt, and `Socket::close()` exists only for explicit early
release. Retirement may wait for an already-acquired operation lease, so no
bounded destructor latency is promised. `SocketView` is the trivially
copyable, non-closing borrowed identity and can be created only from an lvalue
owner. Ordinary code needs no cleanup wrapper or repeated close branch.

Inside a task, ordinary stream and datagram scalar I/O accept `SocketView`
directly. Stream verbs return the concrete move-only `net::Receive` and
`net::Send` operations and await to `net::ReceiveResult` and
`net::SendResult`. Datagram verbs likewise return
`net::datagram::{Receive,Send}` and await to
`net::datagram::{ReceiveResult,SendResult}`. No internal readiness template is
the public return type. Each dedicated awaitable creates no nested task and
internally transfers exactly one matching move-only `ready::Ticket` to the
existing operation. Vectored I/O, accept, connect completion, reusable
readiness, and bounded drains retain explicit Ticket control. Readiness
registration resolves the view and validates its generation. Ticket
consumption performs no registry map search or descriptor-to-entry resolution;
it still performs the required O(1) interest and generation checks. Scalar
consumers then make at most one native attempt. Explicit read, write, and
accept drains consume one ticket but make at most the attempts declared by
their public budget, stopping on would-block, error, callback stop, or the
bound. Read and write drains expose no direct `SocketView` overload: coroutine
code must await readiness, while synchronous root code makes any blocking
`.wait()` explicit before transferring the same ticket. Wrong-interest,
consumed, and retired-generation tickets preserve their exact typed result and
perform no native attempt.

Server handlers return only
`task::Task<rund::net::server::PeerResult>`. `complete`, `stop`, and `fail` map
handler terminals into the same exact network reason; there is no generic
outcome conversion or out-of-band result mirror. The handler first interprets
operation-specific progress, then chooses the terminal. Parallel serving
deterministically reports the non-success reason from the lowest accepted peer
index after every admitted handler has joined. The server folder and
`rund::net::server` namespace are the sole product authority.

## Numeric Evidence Outcome

`rund::evidence::Numeric::Code` is the single validation and decoding outcome.
`Ok`, `NotBuilt`, `BadHeader`, `DuplicateField`, `MissingField`, `BadValue`,
and `HashInvalid` are its complete values. Truth conversion, `ok()`, `error()`,
and `exit_code()` derive from that code; the value stores no parallel boolean
or error owner. `rund::evidence::{make, encode, decode}` are the only build and
codec verbs, and `rund::evidence::Id` is the identity carried by Cluster run
keys. `Numeric::strict_float()` is the sole strict-floating-point observer and
derives its answer from the admitted contract.

## Artifact Identity

An installed binary artifact records its producer SDK version, platform,
compiler, standard library, build type, compile definitions, and external
backend dependency versions. The sealed
`share/runD/release/artifact-identity.tsv` inside the unpacked prefix is the
single authority for that producer tuple; Darwin and Linux use platform-specific
schemas behind the same validator. Package discovery enforces the
requested exact SDK version and word width, resolves every declared backend
dependency, and exports the C++20 and strict floating-point requirements. It
does not guess that arbitrary compiler, standard-library, or dependency
versions form a valid ABI tuple; artifact selection remains constrained to a
published supported tuple whose recorded identity matches the consumer
environment.
The CMake version file uses exact-version matching. `1.0.1` is the current
artifact identity; a major-only version range is not a supported consumption
contract.

The `1.0.1` Alpha extends by-value Compute reports and the inline `Run` receipt
for nested Pipeline evidence. It deliberately uses a new exact SDK identity
because those headers and libraries are not binary-compatible with the
published `1.0.0` tuple. Consumers must build and consume headers and libraries
from one matched exact artifact; `1.0.0` remains a historical identity and may
not be mixed with `1.0.1`. The current checked 64-bit tuple has a 184-byte
`PipelineStats`, 632-byte `Stats`, and 1,152-byte inline `Run`; the sealed
repetition fields are part of that `1.0.1` by-value ABI rather than an extension
that an older header may safely ignore.

## Compute Contract

`<rund/compute.hpp>` owns the basic `Flow`, `Target`, `Backend` observations,
`Device`, `DeviceInfo`, `Buffer`, typed `View`,
`Program`, `Job`, `Batch`, `ProgramCache`, `Bounded<T, Count>`,
`graph::Fingerprint`, `graph::MemoryPlan`, `graph::Info`, `Stats`,
`Status`, `Result<T>`, `Reason`, derived `Code`, the explicit `Compile`
resource envelope, the `input::{Bound,Deferred}` Flow identity markers, and the
nested `compute::telemetry` values.
`graph::Resource::{active,parent,source}` and `graph::Info::memory` expose the
same count-lineage, pointwise destructive-alias, and arena plan used by CPU,
Metal, and Vulkan. Indexed Maps never publish `source` without an identity-read
proof; they are evidence, not another planner or backend switch.
`<rund/compute/async.hpp>` completes the existing deferred Flow's
`compile_async()` terminal and standard future result. It adds no target,
graph, cache, or compilation authority, and the basic Compute entry does not
parse the future/task construction templates.
`<rund/compute/math.hpp>` extends that same Flow with composite expression
functions and matrix, transform, factor, solve, and spectrum stages. It owns
no second graph, target, compilation, or execution authority.
`<rund/compute/pipeline.hpp>` is the opt-in focused direct owner of `pipeline`,
`read`, `write`, `tile_repeat`, `PipelineBuilder`, `Pipeline`,
`PipelineSealedRepetitionCapacity`, copyable
`StateSnapshot`, and the bounded profile vocabulary `PipelineProfile`, `StepClock`,
`StepTimingRelation`, `StepTiming`, `PipelineStepStats`,
`PipelineStepProfile`, and `PipelineProfileSnapshot`; the basic
`<rund/compute.hpp>` entry deliberately excludes it, while `<rund/rund.hpp>`
composes it. The fluent builder binds compiled Programs and resident Buffers
but does not add a graph language.
`<rund/compute/session.hpp>` alone owns the `Session::compute(Job&)` and
`Session::compute(Pipeline&)` templates,
`Request`, `Submission`, `Poll`, `Completion`, and the nested
`Request::Awaiter` bridge.
Kernel IR and Node backend objects remain internal; `Flow` is the single
graph-building language. Fixed-point storage uses only `Fixed<I, F>` and
`quantize<T>()`.
`Stats::available()` and `MemoryStats::available()` are the observation-state
authorities. Invalid and moved-from observations use the non-selectable
`Backend::Unavailable`; installed consumers never infer availability from
zero counters or mistake a missing owner for CPU.
`Stats::control` is the device-generated count, indirect-work, bounded-loop,
conflict, and overflow projection. `Stats::publication` is the transactional
generation, commit/discard, snapshot/restore, and device-loss projection.
Neither creates another execution or failure authority.
`Stats::pipeline::{sealed_repetition_count,coalesced_repetition_count}`
reports an admitted Pipeline's frozen input-sealed repetition count and the
successfully removed suffix. A default or non-Pipeline `Stats` reports both as
zero; an ordinary prepared Pipeline reports one and zero. Physical work
counters are never multiplied by the sealed repetition count.
`Stats::{reset_bytes,reset_commands}` is runtime reset evidence;
`graph::MemoryPlan::{reset_bytes,reset_count}` remains the canonical compiled
range and first-write frontier plan.
Program-cache observations are owned by the cache itself: `ProgramCache::Stats`
is the exact value returned by `ProgramCache::stats()` and has no independent
writer.

`Batch` is the fixed-capacity, move-only accelerator submission owner. It
accepts at most 64 prepared Jobs from one exact Device and exposes only
`add`, `run`, `size`, `capacity`, and the shared `stats` snapshot. It has no
CPU fallback, implicit timer, or dynamic growth.
Batch admission and execution failures use their own `Reason` members; Job
results and hashes remain Job-owned.

`PipelineBuilder` is the transient move-only declaration phase produced by
`pipeline(device)`. `then(program, read(inputs...), write(outputs...))`
composes one prepared step. `sealed_repetitions<N>()` requests one terminal
observation for `N` identical input-sealed evaluations, with
`1 <= N <= 1024`. The cold exact strided-range planner must prove that no
caller-owned write intersects any caller-owned next-evaluation read, and
transactional state is ineligible; otherwise `plan()` and `prepare()` fail with
`PipelineTemporalDependency`. Success still owns one claim, physical
execution, publication, and generation, while failure records zero collapsed
evaluations. This is a throughput contract for frozen inputs, not `N` public
ticks, intermediate checkpoints, or a single-tick latency claim.
`repeat<N>(program, read(inputs...), write_final(outputs...))` is the
terminal-only fixed-count resident recurrence spelling.
`write_each(history...)` selects lossless caller-owned iteration-major output;
plain `write` is rejected on recurrence declarations. The Program
outputs must be the exact prefix of its inputs, that prefix is loop-carried,
and later inputs are invariant. The body is compiled once and preparation
freezes one Program-internal workspace, the seed route, and either the
terminal two-bank route or exact caller-owned history slices for the positive
compile-time bound. `write_each` adds no private carried bank. Bounded resident
windows add one selector that alone names the current route; an inactive
occurrence leaves it unchanged instead of propagating payload through every
remaining bank. The first stop disables later payload commands and performs
at most one parity seal into the immutable final-bank binding used by
downstream steps and publication. Reset, View transfer, and body dispatches
share that one gate; fixed-width status, telemetry, and selector control do
not become a second payload authority.
Program internal Buffers are not retained once per authored occurrence.
`host_feedback(pipeline, count, callback)` is the separate host-control
boundary. Its `HostIteration` callback runs after every successful public run,
may read published outputs, and may write exact typed inputs only before a
next run. Successful prefixes remain committed on a later callback or run
failure. Callback identity and host values are not Pipeline fingerprint
inputs, and GPU execution pays one submit/completion per requested host
iteration.
`windows<Max, Tile>(...)` is the bounded resident-stream recurrence spelling
and accepts terminal-only `write_final(...)`; it does not infer a history
shape from a runtime count.
The body derives its canonical `base`, active `count`, and ordinal through
`resident<Max, Tile>` and therefore authors tile-sized intermediates while
keeping the full input Buffer resident. It is not a request to resize an
already authored full-capacity Program.
`tile_repeat<N>(seed_program, action_program, fold_program)` is the
hierarchical body spelling for a fixed tile-local recurrence inside those
windows. It is a declaration value with no independent `prepare`, `run`,
binding, or observation surface. The enclosing Pipeline retains each compiled
Program once. For flattened tuples `S`, `T`, `P`, and `O`, Seed has signature
`T(S..., U32 total_count, U32 outer_ordinal)`, Action has signature `P(T...)` with `P`
an exact prefix of `T`, and Fold has signature `O(O..., T...)`.
`windows` consequently reads `(O..., S...)` and writes `write_final(O...)`.
The `T` suffix after `P` is one invariant tile bank and Action alternates
exactly two `P` banks for its positive compile-time bound `N`.
Preparation retains `O(ceil(Max / Tile) + N)` compact routes rather than the
outer-times-inner product. It never duplicates the three Program graphs,
Jobs, workspace/View/scratch envelope, or banks per outer/inner pair.
CPU executes one Pipeline semantic order, and Metal/Vulkan retain one
submission with no warm allocation, binding-identity mutation, count readback,
or fallback. `rebinding_count` names post-prepare mutations; cold encoding of
frozen descriptors is not a mutation, and the contract fixture independently
compares all retained owner and View identities across warm executions.
Metal cold preparation records the admitted command graph in one reusable ICB.
Warm execution creates the API-required single-use outer command buffer, makes
one bulk resource-residency declaration, executes the full ICB range, commits,
and observes fixed control. It walks no command, range, binding,
indirect-grid, or recurrence-state descriptor table and performs no rebind;
outer command-buffer/encoder lifecycle and completion remain real host work.
Seed derives the active tail count through `resident<Max, Tile>`. Zero,
partial-tail, overflow, terminal, and first-failure behavior is the
resident-window contract; evidence names phase, outer window, and inner
iteration as separate coordinates instead of flattening them.
`plan()` returns the immutable `PipelinePlan` before allocation. It separates
referenced `persistent_bytes` from Pipeline-owned `peak_bytes`, publishes their
checked `total_bytes`, and identifies the largest single workspace with
`largest_bytes`, `largest_step`, `largest_iteration`, and `largest_chunk`.
For `tile_repeat`, the plan also distinguishes outer-window count, tile
capacity, inner-iteration count, route-template count, native
command-reference capacity, and separate outer/inner coordinates for
largest, peak, and View locations. Observed dispatches remain runtime
statistics and are not inferred from prepared-route cardinality.
`budget(MemoryBudget)` compares the Pipeline-owned `peak_bytes` before
Pipeline-owned Buffer materialization. `peak_bytes` is exactly
`state_bytes + transient_bytes + prepared_bytes`; the prepared term contains
the shared dense View and primitive scratch arenas. `scratch_bytes` and
`scratch_count` expose the scratch payload and page count without exposing
backend handles. Backend allocation granularity, host container capacity,
staging, and opaque native owners are observed after prepare through
`Pipeline::memory()`; their failures retain the owning typed reason.
`state(published, pending)` declares an explicit
double-buffered field. `profile(PipelineProfile::Steps)` is the sole explicit
cold opt-in for bounded per-declared-step evidence; `PipelineProfile::None` is
the default. Optional `restore(snapshot)` seeds a replacement owner and
freezes further step and state declarations. The orthogonal profile mode may
still be selected in either fluent order until `commit()` requests one
all-fields publication point and seals the builder. `prepare() &&` returns the
move-only durable `Pipeline`. `Pipeline` exposes `valid`, bool conversion,
`poisoned`, `run`, `read`, `stats`, `memory`, `memory_snapshot`, caller-storage
`profile`, `fingerprint`, `generation`, `snapshot`, and `restore`.
`StateSnapshot` exposes `valid`, bool conversion, `generation`, `fingerprint`,
and `hash`.
Buffer and typed View use the same `read`/`write` vocabulary; a View adds only
element-unit `offset`, `size`, `stride`, `alignment`, and byte diagnostics.
Declaration order plus exact range hazards is the dependency authority, so a
cycle is not representable in the SDK type surface. State publication uses
`state(...).commit()` and execution uses the prepared `Pipeline` owner.
Pipeline fingerprint identity policy version 3 includes the frozen sealed
repetition count. An omitted declaration and `sealed_repetitions<1>()` are
identical; different positive counts are different identities. Contents,
addresses, backend objects, and profile observations remain excluded.

`PipelineStepProfile::index` is the logical declaration-order correlation
authority. A `repeat<N>` step emits N rows with that same index; `iteration`
is the zero-based physical iteration and `iteration_bound` is N. Its Program
fingerprint is identity rather than a semantic stage name.
`PipelineProfileSnapshot` captures aggregate execution, step rows, whole
Pipeline memory, the disjoint shared-memory remainder, non-owning canonical
resource bytes, run-side instrumentation commands/bytes, and a separately
sampled host-side snapshot-observation duration in one epoch.
Unavailable timing has an explicit clock and zero sample count; a measured
zero duration remains available. `Stats`, `PipelineStats`, `MemoryEntry`,
`MemoryStats`, and `compute::telemetry::Profile` retain their meanings, while
the recurrence release adds the two explicit iteration fields to
`PipelineStepProfile` and the sealed-repetition release adds the two fixed-width
fields to `PipelineStats`. The Pipeline terminal-control ABI occupies 128 bytes.
Profiling mode and observations never enter the Pipeline fingerprint, Replay,
output, snapshot, publication, or failure identity.

`PipelineStepCapacity` is the 64-step logical declaration envelope and
`PipelineIterationCapacity` is the independent 1,024-entry prepared execution
envelope. `PipelineSealedRepetitionCapacity` independently bounds input-sealed
repetition at 1,024. The maximum 32 typed leaves therefore admit at most 32,768 frozen
binding occurrences under one checked product bound. Public
`PipelineStats::step_count`, `verified_step_count`, and
`failed_step_index` always use logical declaration indices; native control and
profile rows retain physical iteration evidence internally without leaking a
second public step authority.

For body graph size `G`, internal workspace bytes `I`, carried bytes `S`,
binding width `L`, body dispatches `D`, and bound `N`, retained body payload is
`Theta(G + I + S)` independent of `N`; frozen occurrence metadata is
`Theta(N * L)` and necessary execution remains `Theta(N * D)`. Fixed
occurrence order and the cross-occurrence visibility frontier make shared
workspace reuse bit-equivalent to explicit recurrence expansion.

The independent Program graph envelope is 16,384 ordered nodes with at most 64
buffer refs per node and 64 public graph outputs. It is not the 1,024-node
ComputeIR envelope: large Programs stay one logical graph while deterministic
fusion forms only maximal regions that fit one IR. Graph and fusion planning
storage is proportional to the authored graph, and no maximum-size graph table
is retained by Program, Job, or Pipeline.
Each occurrence also consumes its mutable device telemetry before the next
route reuse. One prepared open resets status identity, verified prefix, and the
telemetry suffix before any Program command. Telemetry accumulates immediately
after its owning occurrence, and each status-bearing primitive canonicalizes
and folds its local status before a later primitive can reuse native storage.
The earliest failure remains the sole terminal reason. One terminal close
records generation and verified prefix from the folded state.
Replayed and explicit recurrence therefore expose the same per-iteration and
aggregate control evidence.

Node-host `rund::compute::Request` is the inert move-only admission value and
`rund::compute::Submission` is the sole admitted owner. Its
`rund::compute::Poll` snapshots and terminal `rund::compute::Completion`
retain one `Reason` and expose `reason()`, derived `code()`, and derived
`error()`. Poll admission, backend, and completion flags are observations, not
parallel failure authorities. `rund::compute::Completion::exit_code()` is
derived from the same Reason. `Submission::wait_for(duration)` returns the same
Poll snapshot after a bounded wait and never cancels or detaches the owner.

The [Compute reference](../../../docs/reference/compute.md) owns Flow,
selection, compilation, caching, numeric, Job, and shared telemetry semantics;
the [Pipeline contract](../../../node/docs/contracts/compute/pipeline.md) owns
dependent Program execution, claims, poison, status, memory, and one-submit
semantics. Package target reachability and collision rules belong to
[SDK Surface](../surface.md) and [SDK Consumption](../consumption.md); they are
not repeated as API behavior here.

## Verification

The release route installs the artifact, configures it through
`find_package(runD 1.0.1 EXACT CONFIG REQUIRED)`, builds all consumers, and runs them
against `runD::sdk`. Those consumers compile the current runtime and Compute
usage from the installed package. They cover
explicit no-fallback CPU, Metal, and Vulkan selection and execution,
runtime replay codec/diff/re-execution, archive-backed host I/O replay with no
valid native descriptor, numeric evidence, and cluster retry identity.
The positive installed consumer is configured as `Release`, matching the
artifact producer. A separate expected-failure configure requests a nonexistent
required component and proves that rejection occurs before `runD::sdk` is
imported.
The installed task surface also compiles ordinary `Task<T>` coroutine return,
move, and await use while compile-time contracts reject raw frame construction,
handle observation, release, destruction, and public promise-support names.
