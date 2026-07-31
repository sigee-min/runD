# Compute API

Default header: `<rund/compute.hpp>`

Focused Pipeline header: `<rund/compute/pipeline.hpp>`

Opt-in async header: `<rund/compute/async.hpp>`

Opt-in math header: `<rund/compute/math.hpp>`

Opt-in Session header: `<rund/compute/session.hpp>`

Namespace: `rund::compute`

The basic Compute header deliberately excludes Pipeline binding templates.
Include the focused Pipeline header for dependent Program execution, or use the
all-domain `<rund/rund.hpp>` composition.

Compute has one public construction language: typed, lazy `Flow`. Applications
do not author Kernel IR, internal Accel graphs, adapter selections, or raw
backend resources. Every fallible operation returns `Result<T>` or `Status`.
Compiler diagnostics name an input-bound Flow with `input::Bound` and a
compile-once Flow with `input::Deferred`; no implementation-only type is part
of Flow's public identity. These are compile-time markers only and occupy no
runtime storage.

## First Success

`on(target, input)` borrows any lvalue contiguous range that forms a
`std::span`, including `std::vector`, `std::array`, `std::span`, C arrays,
PMR containers, and custom contiguous ranges. Its element type must exactly
match the Flow value type after removing `const`; implicit scalar conversion is
not an input contract. Rvalues, `std::initializer_list` (even when named), and
proxy-element containers such as `std::vector<bool>` are rejected, so the
borrowed view cannot dangle or silently materialize another representation.

```cpp compile run source=package/tests/consumer/example/compute.cpp
#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

int main() {
  std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto output = rund::compute::on(rund::compute::Target::cpu(), input)
                    .map("twice", [](auto value) { return value * 2 + 5; })
                    .collect();
  if (!output) {
    return output.exit_code();
  }
  return *output == std::vector<std::int32_t>{7, 9, 11, 13} ? 0 : 2;
}
```

`collect() &&` is the expensive terminal: it opens only the selected backend,
lowers and compiles the recipe, allocates, uploads, runs, downloads, and returns
the typed host value. It never retries on CPU when an explicitly selected
accelerator is unavailable.

## Performance Model

Element count alone does not determine whether an accelerator wins. Submission,
synchronization, transfer, arithmetic intensity, and device-memory traffic all
participate in the boundary. Use
[GPU Performance](https://github.com/sigee-min/runD/blob/main/wiki/Performance) to choose
between `collect()`, reusable Program execution, resident Job, Pipeline, and
Batch without changing the canonical graph.

## Compile Once, Reuse

For repeated execution, open one explicitly bounded backend, build one Flow,
compile one Program, then run that immutable Program with each changed input.
Each `Result` keeps one typed `Reason`; this first-success path returns its
derived `exit_code()` without translating failure into another authority.
Program reuse skips graph construction and compilation but always executes
again, so it is not result memoization.

```cpp compile run source=package/tests/consumer/example/program.cpp
#include <rund/compute.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <utility>

int main() {
  constexpr std::size_t Count = 4u;
  auto device = rund::compute::open(rund::compute::Target::cpu(1u));
  if (!device) {
    return device.exit_code();
  }

  auto flow = rund::compute::on(*device).map<std::int32_t>(
      "affine", Count, [](auto value) { return value * 2 + 5; });
  auto program = std::move(flow).compile();
  if (!program) {
    return program.exit_code();
  }

  constexpr std::array<std::int32_t, Count> Initial{1, 2, 3, 4};
  constexpr std::array<std::int32_t, Count> Changed{10, 20, 30, 40};
  auto first = program->run(std::span<const std::int32_t>{Initial});
  auto second = program->run(std::span<const std::int32_t>{Changed});
  if (!first) {
    return first.exit_code();
  }
  if (!second) {
    return second.exit_code();
  }

  constexpr std::array<std::int32_t, Count> First{7, 9, 11, 13};
  constexpr std::array<std::int32_t, Count> Second{25, 45, 65, 85};
  return std::ranges::equal(*first, First) &&
                 std::ranges::equal(*second, Second)
             ? 0
             : 2;
}
```

## Composition

The primary chain supports `map`, `filter`, `scan`, `reduce`, `sort`,
`argsort`, `compact`, `histogram`, `window`, `gather`, `scatter`, `partition`,
`segmented_scan`, `segmented_reduce`, bounded `expand`, bounded `join`, integer
`group_by`, and explicit graph `unroll<N>`.

- `branch` exposes immutable typed stage references.
- `record` creates ordered structure-of-arrays fields; it never uploads a C++
  object layout.
- `outputs` fixes terminal order.
- `zip` creates an equal-cardinality heterogeneous element stage.
- `combine` is the explicit sequence/scalar broadcast path.
- `pipe(function)` reuses an ordinary C++ recipe builder without retaining it
  as an execution callback.

Cardinality is carried in the C++ type. Sequence-only operations are absent on
scalar stages and bounded operations retain a resident logical count. Include
`<rund/compute/math.hpp>` for composite expression functions and matrix,
factor, solve, spectrum, and complex transitions; they extend the same Flow
authority and expose only valid next operations.

`compact({.capacity = N})` emits the original indices of nonzero U32 flags in
strictly increasing source order. The capacity is a hard selected-count bound:
an actual count above `N` fails the whole run with
`compute_compact_capacity_insufficient` instead of truncating or publishing a
partial result. CPU, Metal, and Vulkan obey the same rule in Standalone
and Node-native execution. Its public type is
`stage::Bounded<std::uint32_t>` with capacity `N` and a resident U32 logical
count. Underfilled results are not padded: downstream bounded operations and
terminals consume only the logical prefix, and readback transfers the count
followed by that prefix rather than the unused tail.
An empty source is valid and returns an empty sequence with resident U32 count
zero through the canonical empty Program path. The host-materialized count is
hashed as one U32 value without recording backend transfer traffic. If a
nonempty input selects
nothing, the terminal reads the count but performs no value-buffer transfer.
Empty-Program graph inspection reports the zero-element value output and
one-element count output even though there are no executable backend nodes.

A nonempty Program admits up to `graph::NodeCapacity == 16384` canonical
ordered graph nodes after liveness and identity removal but before fusion.
`Program::graph().authored_nodes` reports that admission count, not the raw
number of fluent calls; `lowered_nodes` and `nodes.size()` report the selected
executable count. `graph::ValueCapacity`, `graph::BuffersPerNode`, and
`graph::OutputCapacity` expose the related checked bounds. One Map expression
and its ComputeIR remain bounded to 1,024 nodes, so this larger schedule
envelope does not create a giant shader. The compiler keeps one Program and one
submission, then deterministically ends each maximal Map-fusion region before
the 64-binding or 1,024-IR-node bound. Authored graph order, Fixed policy,
fingerprint, dependencies, barriers, and output bits remain the single
authority on CPU, Metal, and Vulkan.

`group_by` reuses compact's resident count as its only group-count authority.
Before gathering group keys across capacity, inactive head-index slots are
masked to source index zero, so a many-group to fewer-group resident rewrite
cannot dereference stale compact tail storage. A 64-bit grouped value Map that
returns the public U32 `mask(...)` form reuses that canonical U32 head stream
and canonical active-count mask for segmented follow-up operations; it does
not reinterpret, narrow, or add a second conversion authority to the 64-bit
value-domain boundary marks.

`sort()` and `argsort()` use one canonical stable order. A square Fixed Matrix
may solve from its raw matrix and RHS with
`solve(rhs, FactorOp, rhs_cols)` or
`solve<FactorOp, RhsCols>(rhs)`; LU, QR, and Cholesky return values plus one
status per batch. Matrix and RHS must share `(I,F)`, rounding, and overflow,
but keep independent approximation provenance; an Exact RHS is valid beside a
Deterministic matrix and the Solve result is Deterministic.

The public scalar domain is closed to `int32_t`, `uint32_t`, `int64_t`,
`uint64_t`, and `Fixed<IntegerBits, FractionBits>`. Stored Fixed formats have
exactly 32 or 64 bits. Floating-point inputs are compile-time errors; ratio
construction and every precision-losing storage boundary are explicit through
integer rounding and `quantize<T>()`.

## Element Expressions

Element functions build one canonical expression graph. Arithmetic, bitwise,
comparison, predicate, selection, saturation, shift, Fixed nonlinear, and
explicit quantization operations are available directly. The public helper
families cover range and interpolation, dot/convolution, statistics,
metric/geometry, small matrix and polynomial algebra, activations and signal
windows, complex arithmetic, hashing, and deterministic noise. Every legal
helper lowers through the same opcode graph on CPU, Metal, and Vulkan;
none is a retained host callback.

Mode values are grouped under one concise selector per helper family, for
example `FixedOp::Half`, `MetricOp::Squared`, `GeometryOp::Projection`, and
`ActivationOp::HardSwish`. Geometry helpers share `GeometryOp` as their sole
mode authority. Dispatch-only tag types stay nested under their selector, so
they do not pollute the `rund::compute` namespace or form a second mode
vocabulary.

Fixed intermediate expressions keep widened precision. A stored Fixed result
must use `quantize<T>()`, whose target format, rounding, overflow, and
approximation policy participate in graph and Program identity. Implicit
mixed-format Fixed arithmetic and conversion to or from floating point are not
part of the language. Public expression constraints and the Standalone Compute
contracts own the current opcode, helper, numeric-policy, and backend behavior
directly.

Flow expression inputs always use the referenced resource's stored Fixed
format. Nondefault policy therefore survives map, filter, bounded/scalar map,
combine, branch, and zip instead of reverting to the C++ type default. Complex
imaginary input must share the real side's `(I,F)`, inherits its rounding and
overflow, and retains independent approximation provenance.

`mul_fixed`, `mul_fixed_scaled`, and `mul_unsigned_fixed` are distinct stored
operations. They use the declared `F` to scale the widened product and differ
only in signed/signed, signed/unsigned, and unsigned/unsigned lane
interpretation; rounding occurs before the matching signed or unsigned
overflow law. `mul_add_fixed` keeps its product and aligned addend widened and
stores only at the caller's terminal `quantize<T>()`.

## Compile Once And Run Resident

Use a shape-only Flow, `compile()`, and `resident()` when one fixed graph runs
repeatedly:

```cpp compile run
#include <rund/compute.hpp>

#include <array>
#include <cstdint>
#include <vector>

int main() {
  const std::array<std::int32_t, 4> input{1, 2, 3, 4};
  auto program =
      rund::compute::on(rund::compute::Target::cpu(4))
          .map<std::int32_t>("twice", input.size(),
                             [](auto value) { return value * 2 + 5; })
          .compile();
  if (!program) {
    return program.exit_code();
  }

  auto job = program->resident(input);
  if (!job) {
    return job.exit_code();
  }
  const auto first = job->run();
  if (!first) {
    return first.exit_code();
  }
  const auto second = job->run();
  if (!second) {
    return second.exit_code();
  }

  auto output = job->read();
  if (!output) {
    return output.exit_code();
  }
  return *output == std::vector<std::int32_t>{7, 9, 11, 13} ? 0 : 2;
}
```

`resident()` prepares fixed buffers, bindings, pipelines, and scratch before
the warm loop. A successful warm `Job::run()` performs no compile, SDK
allocation, input upload, or output download. `write()` is an
explicit input update and `read()`/`read<I>()`/`read_all()` are explicit host
download boundaries.

## Chain Dependent Programs

Use `Pipeline` when several already compiled Programs share resident Buffers
and must execute in one fixed order on every tick. The API reads in execution
order: select one Device, append each Program with its typed read and write
bindings, then prepare once.

```cpp fragment
auto prepared =
    rund::compute::pipeline(*device)
        .then(*integrate,
              rund::compute::read(*position, *velocity, *force),
              rund::compute::write(*next_velocity))
        .then(*advance,
              rund::compute::read(*position, *next_velocity),
              rund::compute::write(*next_position))
        .prepare();
if (!prepared) {
  return prepared.exit_code();
}

rund::compute::Pipeline tick = std::move(*prepared);
const rund::compute::Status ran = tick.run();
if (!ran) {
  return ran.exit_code();
}
```

`PipelineBuilder`, `Pipeline`, and the transient `read(...)`/`write(...)`
binding packs are move-only. Buffers must be lvalues, and write bindings must
be mutable. Program signature expansion checks the exact leaf order, type,
count, and Fixed policy before backend preparation. Pipeline retains the bound
Buffer owners after preparation.

Declaration order is the only step order. There is no dependency handle, DAG,
topological reorder, or cycle reason; the type surface cannot express a cycle.
Preparation infers forward RAW, WAR, and WAW hazards between steps and rejects
unsupported same-step aliases. A nonempty Metal or Vulkan tick submits one
prepared command stream; CPU executes the same steps in declaration order and
submits no native GPU command.

Use `repeat<N>(body, read(...), write(...))` for fixed-bound resident
recurrence. The body's outputs must match the prefix of its inputs: that prefix
is loop-carried state and later inputs are invariants. The body Program is
compiled once; preparation creates one private ping-pong bank, one shared
Program-internal workspace, and freezes the seed plus two alternating binding
routes. Internal buffers are not copied per iteration. CPU, Metal, and Vulkan
execute the same fixed occurrence order and visibility frontiers, so workspace
reuse preserves every result bit while retaining body payload independently of
N. There is no host callback, readback, graph rebuild, warm allocation, or
fallback, and a GPU Pipeline performs one native submit. It consumes one of the 64
declared steps and up to N of the separate 1,024 prepared-iteration envelope.
`PipelineIterationCapacity` is the single public authority for that bound.
Step-profile rows report the common logical `index`, zero-based `iteration`,
and `iteration_bound`.
Mutable count, predicate, and primitive telemetry is consumed immediately
after each occurrence and before the next route reuse. The immutable stream
resets and folds that evidence on every submission, so warm runs cannot inherit
or overwrite another occurrence's counters.

Use `windows<Max, Tile>(body, window(count), read(...), write(...))` when the
same recurrence consumes a device-resident bounded stream. The body receives
the count and canonical window ordinal after its authored inputs and derives
`base` and the active window count with `resident<Max, Tile>(count, ordinal)`.
The body is compiled once, all occurrences reuse one scratch workspace, and
the complete Metal or Vulkan stream is submitted once.
The body itself must express tile-local work through that `resident` value.
`windows` does not reinterpret or resize an already compiled full-capacity
Program, because the runtime cannot infer which intermediates are window-local
without changing graph meaning.

```cpp fragment
auto prepared =
    rund::compute::pipeline(*device)
        .windows<516096, 8192>(
            *fold,
            rund::compute::window(*count).until<1>(7u),
            rund::compute::read(*accumulator, *terminal, *geometry),
            rund::compute::write(*result, *stopped))
        .prepare();
```

The optional `until<Index>(expected)` names a recurrent U32 scalar leaf.
The recurrence owns one device-resident selector over the seed and two private
banks. Once its selected input equals `expected`, later occurrences leave that
selector unchanged; they do not copy the inactive payload through alternating
banks. On the first stop, runD disables the remaining recurrence-owned command
ranges, including reset, View transfer, and body dispatches, and only when the
selected bank differs from the prepared final bank seals the selected bytes
there once. Fixed-width status, telemetry, and selector control may continue
without writing recurrent payload. Downstream Programs and final publication
then consume that one canonical bank. The host never reads the count or
terminal, and terminal does not select another graph, executor, submit path,
or CPU fallback.

Call `builder.plan()` before `prepare()` to inspect admission. `peak_bytes` is
the exact Pipeline-owned planned payload
(`state_bytes + transient_bytes + prepared_bytes`), `persistent_bytes` is
referenced caller storage, and `total_bytes` is their checked sum.
`prepared_bytes` is the aligned backing payload of the typed dense View and
primitive scratch arenas borrowed by prepared GPU commands; logical slots and
collective temporary requests are suballocated from shared owners instead of
being allocated once per binding, primitive, Program, or recurrence phase. A
raw-word slot can serve sequential uses of different scalar types and is
aligned to the strongest scalar and backend offset requirement. Pipeline steps
are serial, so `scratch_bytes` and `scratch_count` report the maximum
deterministic Program page envelope rather than the sum of every occurrence.
`MemoryBudget` admits the complete planned payload before Pipeline-owned Buffer
creation. The largest Program workspace is reported by `largest_*`; the largest
normalized View is reported by `view_bytes`, `view_step`, `view_iteration`, and
`view_binding`, with exact span/backing/offset/stride/element/count/alignment in
the remaining `view_*` fields. `DeviceInfo::storage_alignment` and
`storage_bytes` expose the selected backend's descriptor boundary. Backend
allocation rounding, driver metadata, and transient transfer pools are measured
after prepare by `memory()` instead of being mixed into a false preflight
scalar. `memory_snapshot()` labels scratch separately in Resident and Device
categories, making logical capacity and physical allocation rounding directly
comparable.

After success, call `tick.read(buffer, output_span)` for a declared write
Buffer. Pipeline never performs an implicit payload download. `stats()` and
`memory()` remain allocation-free observations, `fingerprint()` identifies the
ordered Programs and alias topology without object addresses, and
`generation()` advances only after successful publication. A failure after
writes become possible poisons the Pipeline and all declared write Buffers;
recovery replaces or restores them from the application's checkpoint instead
of silently retrying on CPU. The exact contract is
[Compute Pipeline](https://github.com/sigee-min/runD/blob/main/node/docs/contracts/compute/pipeline.md).

## Submit Through A Session

Include `<rund/compute/session.hpp>` only when a resident Job or Pipeline must share a
Session's bounded scheduler and lifecycle. The four public types form one
direct sequence:

Open a Session-bound Device with `compute::open(session, target)`. It binds to
the one compile service owned by `SessionConfig::compile`; it never creates a
Device-local service or falls back to a standalone target.

```cpp fragment
auto device = rund::compute::open(session, rund::compute::Target::cpu(4));
if (!device) {
  return device.exit_code();
}
```

| Step | Type | Operation |
| --- | --- | --- |
| admission value | `rund::compute::Request` | returned by `session.compute(job)` or `session.compute(pipeline)` |
| admitted owner | `rund::compute::Submission` | returned by `request.submit()`; owns `poll()`, `wait_for()`, `cancel()`, and `wait()` |
| progress snapshot | `rund::compute::Poll` | returned by `submission.poll()` or `submission.wait_for(duration)` |
| terminal publication | `rund::compute::Completion` | returned by `submission.wait()` or direct await |

```cpp fragment
rund::compute::Request request = session.compute(*job);
rund::compute::Submission submission = request.submit();
rund::compute::Poll progress = submission.wait_for(std::chrono::seconds{1});
rund::compute::Completion completed = submission.wait();
```

Inside a runD task, `co_await session.compute(*job)` returns that same
`Completion`; `co_await session.compute(tick)` does the same for Pipeline. The
coroutine bridge is nested as `Request::Awaiter`; there is no
second root awaiter or host-result vocabulary. Construction alone does not
submit work, and Standalone `Job::run()` remains the separate synchronous
terminal. `Pipeline::run()` is the corresponding standalone synchronous
terminal. `Submission::cancel()` requests cancellation through the same Job or
Pipeline owner and returns a typed `Status`; `wait()` remains the sole terminal
`Completion`. Cancelling an already terminal Submission reports
`compute_already_completed` instead of manufacturing another terminal state.
`Poll::submitted`, `backend_submitted`, and `completed` separately report
admission, actual backend start, and publication; a pending admitted request
keeps `Reason::Ok`. A `wait_for()` timeout leaves the Submission and native
command owned and returns `completed == false`; it is not cancellation.

## Batch Small Accelerator Jobs

Use `rund::compute::Batch` when up to 64 independent resident Jobs are ready on
the exact same Metal or Vulkan Device. Jobs may have different Programs,
signatures, shapes, and pipelines. `Batch` shares only one native command
submission; it does not fuse their graphs or delay work to discover a future
batch.

```cpp fragment
rund::compute::Batch batch;
const auto first_added = batch.add(*first);
if (!first_added) {
  return first_added.exit_code();
}
const auto second_added = batch.add(*second);
if (!second_added) {
  return second_added.exit_code();
}
const auto completed = batch.run();
if (!completed) {
  return completed.exit_code();
}
auto first_output = first->read();
auto second_output = second->read();
```

Batch has fixed inline capacity, no CPU fallback, no dynamic spill, and no
implicit timer. It claims every Job before executing any; empty, full,
duplicate, cross-Device, CPU, busy, and invalid prepared inputs retain distinct
typed reasons. Metal commits one command buffer and Vulkan performs one
`vkQueueSubmit`; results and overflow reasons remain Job-owned and finish in
admission order.

`Batch::stats()` owns that one execution submit and its timing. The Job
snapshots taken immediately after `Batch::run()` contain no duplicate shared
submit or timing. A later explicit accelerator `read()` may own a separate
transfer command and publishes the Job output hash. See the complete
[Compute Batch contract](https://github.com/sigee-min/runD/blob/main/node/docs/contracts/compute/batch.md).

## Execution Evidence

`Job::stats()`, `Run::stats()`, and Node-host `rund::compute::Completion::stats()` expose
one product `Stats` snapshot. Counter names state their unit and owner:
pipeline and buffer creation, public download events, logical dispatches,
physical command submits, transfer and intermediate round-trip bytes, cache
and reuse events, fusion-plan counts, backend timings, graph/output identity,
and CPU worker/tile/SIMD evidence. In particular, `dispatches` and
`command_submits` are not aliases: a multi-pass algorithm may encode several
dispatches into one queue submission.

For a Pipeline, `stats().pipeline` keeps its evidence visibly nested:
`step_count`, `resource_count`, `barrier_count`, `claim_conflict_count`,
`verified_step_count`, `failed_step_index`, `status_entry_count`,
`control_byte_count`, `control_command_count`, `claim_ns`, and `control_ns`.
`failed_step_index == PipelineStats::no_failed_step` means success or an
unknown failure step; the terminal `Reason` distinguishes those outcomes.
Control bytes are exactly 80 when the GPU control record is observed and zero
for CPU, zero-work, or control-lost execution. They are not payload download
bytes.

`Stats::available()` distinguishes an execution observation from an invalid or
moved-from owner. Before a Job's first run the snapshot is available, names the
selected backend, and has zero execution counters. A moved-from Job instead
returns `Backend::Unavailable` with `available() == false`; it never looks like
a zero-work CPU run. `MemoryStats::available()` applies the same rule and also
requires a concrete memory scope.
`reset_bytes` and `reset_commands` report the payload and physical operations
consumed by exact first-write resets; the graph's `MemoryPlan` remains the
compile-time reset-range authority.

Node-host `rund::compute::Poll` and `rund::compute::Completion` use the same failure vocabulary as
Standalone Compute. Each retains one typed `Reason`; `code()` and `error()` are
derived views. An admitted nonterminal poll has `Reason::Ok`, and an immediate
admission rejection is already terminal. `rund::compute::Completion::exit_code()` is the
direct application-boundary projection of that same Reason.
CPU leaves accelerator-only fields at zero. That value means unavailable or
not applicable; it is not a measured zero-duration device event.
`kernel_samples == 0` likewise makes `kernel_ns` unavailable. Timings are local
diagnostic evidence, not portable performance claims. Use the stateless
`kernel_timing_available()` helper rather than treating zero nanoseconds as a
measurement. Explicit input updates
retain their exact event owner in `WriteStats::uploads`; `Stats` does not infer
an upload event from a nonzero byte count.

`Job::profile()` returns one `telemetry::Profile` built from the live job's
owning `DeviceInfo`, `Stats`, and Job-scoped `MemoryStats`. The raw snapshots
remain available through read-only `device()`, `execution()`, and `memory()`
accessors. Capture fails for an invalid or moved-from Job rather than
substituting another backend or an empty snapshot. Capture also fails with
`compute_profile_busy` while the Job is queued, running, or writing input, so
an in-flight execution is never represented by zero counters. Derived
accessors are fixed-work, allocation-free reads; an unknown memory category
fails instead of selecting a default category.

`kernel_time()`, `dispatches_per_submit()`, `dispatch_reduction()`, and
`memory_usage(category)` return exact `Rate {numerator, denominator}` values.
`pipeline_cache()`, `buffer_reuse()`, `descriptor_reuse()`, and
`internal_traffic()` return `Share {selected, other}` values, keeping the two
64-bit cumulative terms separate so denominator addition cannot overflow. A
zero denominator—or two zero Share terms—is explicitly unavailable, and the
optional presentation value is absent. A `UINT64_MAX` component is explicitly
`saturated()` and also has no derived value because exact maximum and prior
overflow are indistinguishable. `largest_time()` returns every stage
tied for the largest cumulative compile, setup, submit, kernel, or readback
counter. It applies no thresholds, does not add potentially overlapping timing
scopes, and is not a portable cross-device speedup claim. The complete formulas
and failure reasons are owned by the checked-in
[Compute reference](https://github.com/sigee-min/runD/blob/main/docs/reference/compute.md#job-profile).

`profile->findings()` is the direct remediation path. It returns the same
fixed-capacity `rund::telemetry::Findings` used by Session events for buffer
allocation, upload/download copy, canonical graph scan, command-queue pressure,
and critical path. Each nonzero cost carries a typed cause and action; queue
pressure is emitted only when the observed in-flight peak reaches the exact
command capacity. The Profile keeps no mirrored Event or derived counters.

## Graph Observation, Cache, And Async Compile

`Program::graph()` returns immutable, domain-neutral `graph::Info`: canonical
resources, ordered accesses, lifetimes, dependencies, barriers, and a 128-bit
fingerprint. `Info::memory` is the canonical `MemoryPlan` observation with
logical, peak-live, retained physical bytes, retained Buffer count, and the
exact logical ranges reset immediately before their first writer. Disjoint
closed lifetimes may reuse the same aligned arena offset; the later lifetime is
cleared only after the earlier one ends. If `R` is the ordered reset route set,
its public totals are

```text
reset_bytes = sum(r in R, bytes(r))
reset_count = |R|
```

Every route retains its logical Buffer offset and extent even when several
routes share one arena owner. Padding and unrelated arena intervals are not
reset. The compiler seals accelerator binding ordinals once in
`O(B + L + R log R)` work for `B` bindings and `L` logical values; native
preparation freezes one contiguous reset span per step. Warm encoding visits
exactly those spans without a reset-route search or allocation.

Resources expose `active` count identity, `parent` count lineage, `source`
pointwise destructive-alias proof, arena group/offset, and `reset_node` when a
partial-write lifetime needs initialization. Indexed `ReadAt` Maps do
not receive a `source` transition: equal shape and coincident lifetimes do not
prove identity addressing, and parallel lanes could otherwise overwrite a
source element before another lane reads it. When several pointwise,
arena-backed, same-shape inputs qualify at one dense full-write Map, `source`
is the smallest canonical resource ID. It is
observation of the Flow-derived graph, not a second graph builder. Resource
metadata contains types, ranges, access modes, and alias groups—not game,
physics, thermal, electrical, ECS, or renderer concepts.
`resource::AccessMode` is the single read/write type shared with the generic
resource planner, and `resource::NoNode` marks a resource with no use.

Include `<rund/compute/async.hpp>` when this asynchronous terminal is used.
Create a positive-capacity `ProgramCache` from one opened `Device`, then build
with `on(device, cache)`. The Device remains the sole target authority and the
cache must belong to it. Equivalent canonical graphs on that Device share one compiled
Program even when diagnostic Map names differ. Concurrent identical misses
coalesce, and `flow.compile_async()` returns a future of the same compile
`Result`. Different inputs always execute again: the cache stores compiled
artifacts and never memoizes outputs, snapshots, ticks, or domain state.
Within that Device-bound, process-local cache the complete lookup key is the
`graph::Fingerprint`; graph and numeric policy are already part of the
fingerprint. Device and compiler/runtime ABI scope the cache itself, and no
cache entry is persisted. `stats().misses` is the sole count of admitted
compilations; `hits` counts ready reuse and `waits` counts in-flight
coalescing. The returned snapshot is `ProgramCache::Stats`; it has no
independent cache or mutation authority.

On Vulkan, a cache miss owns its SPIR-V shader module only during synchronous
pipeline construction. Cache publication retains the executable pipeline,
layouts, descriptors, exact artifact identity, and source; warm execution has
no live shader-module handle. This lifetime rule changes no graph identity,
numeric policy, dispatch order, or output bits.

Async admission belongs to the explicit `Compile {workers, capacity}` envelope.
Standalone Devices own the envelope passed to `open`; Session-bound Devices
share the service owned by `SessionConfig::compile`. Full admission returns
`Reason::AsyncCompileCapacity` before packaged-task or future construction; the
rejection is allocation-free and does not invoke the recipe factory. Reserved
slots fix FIFO order even when factories complete in another order, and a
failed factory cancels its slot without losing capacity. Admission does not
block, grow, spill, or compile synchronously. Shutdown cancels uncommitted
reservations and drains committed work, while independent workers may finish
in either order. Session close stops admission, blocks while it cancels/drains
resident jobs and the shared service, and leaves surviving Devices unable to
replace it.

A zero-capacity Flow exposes no executable nodes or barriers, but it does not
use a second empty-graph identity. The normal Flow-to-Graph builder
records the canonical Map expression, Scan mode, primitive descriptor options,
ordered interface, and fixed numeric policy in the fingerprint. Renames, dead
branches, and temporary creation order remain identity-neutral; changing an
operation, mode, option, output, or numeric policy produces a different cache
key even though execution creates no backend dispatch.

```cpp fragment
auto device = rund::compute::open(
    rund::compute::Target::cpu(4),
    rund::compute::Compile{.workers = 2, .capacity = 64});
if (!device) return device.exit_code();
auto cache = rund::compute::program_cache(*device, 64);
if (!cache) return cache.exit_code();

auto pending = rund::compute::on(*device, *cache)
                   .map<std::int32_t>("integrate", 1024,
                                      [](auto value) { return value + 1; })
                   .compile_async();
if (!pending) return pending.exit_code();
auto program = pending->get();
if (!program) return program.exit_code();
if (!program->fingerprint()) return 2;
```

## Backend And Error Rules

Choose exactly one target with `Target::cpu(n)`, `Target::metal()`, or
`Target::vulkan()`. The same `Target` value is passed directly to `on` and
`open`; an already-open `Device` is the other execution path. `Backend` is an
observation value for Device info, statistics, and telemetry, not a selector.
Backend identity, graph shape, ordered inputs and outputs, bounds, numeric law,
and stable failure reasons participate in the execution contract.
`Backend::Unavailable` is an observation-only sentinel for unavailable
snapshots. It cannot be passed to `on` or `open`; no invalid selection reaches
runtime and no target falls back to CPU.

After `open`, call `device.info()` for an owning `DeviceInfo` snapshot with
`backend`, `name`, `driver`, `driver_details`, `storage_alignment`, and
`storage_bytes`. The numeric fields expose the selected backend's immutable
storage-binding boundary. Accelerator identity values are exact copies of the
selected backend fields; an empty field remains empty. CPU values are derived
from the selected SIMD capability and worker owner at the call boundary, so
Device state keeps no duplicate identity strings. The snapshot can outlive the
Device. Allocation failure is reported as `Code::Capacity` rather than escaping
through the API.

On Apple Silicon, a Vulkan snapshot whose `driver` is `MoltenVK` identifies a
Vulkan-to-Metal translated execution path. Treat its timing as evidence for
that exact path: Vulkan lowering, SPIR-V, descriptors, commands, barriers,
output parity, and same-path regression. It is not native Vulkan throughput.
The direct Metal row is native Metal evidence on that Apple device; comparing
the two rows compares complete software stacks rather than isolating API cost.
Portable native Vulkan performance requires measurement on a native Vulkan
driver.

Fixed policy in `graph::Info` uses the public `Rounding`, `Overflow`, and
`Approximation` enum types directly, so introspection never requires casts from
an internal byte encoding.

`Reason` is the one stored failure authority. Branch on `reason()` when an
exact failure matters, or on its derived small `Code` category; retain the
derived `error()` text when diagnostics or evidence need it.
`Status::fail(Reason)` and `Result<T>::fail(Reason)` are the only failure
factories. No public factory accepts a borrowed string or a separately chosen
Code. `Result<T>::transform`, `and_then`, and
`value_or` preserve failures without converting expected errors to exceptions.
One checked-in schema generates the enum, diagnostic projection, and internal
foreign-boundary parser, so those views cannot drift into separate authorities.
The current codes are `Ok`, `Invalid`, `Unsupported`, `Unavailable`,
`Capacity`, `Compile`, `Binding`, `Transfer`, and `Execution`.
Within `Execution`, `DeviceBusy` means another Job owns the selected device,
`JobBusy` means the same resident Job is already active, and
`RuntimeBusy` means Session/runtime control is owned elsewhere. These reasons
are not aliases; retry and diagnostics can therefore follow the actual owner.
Pipeline preparation distinguishes empty/capacity/Device/shape/Fixed/alias
errors; execution distinguishes `PipelineBusy`, `PipelinePoisoned`,
`BufferBusy`, and `BufferPoisoned`. A poisoned output requires application
checkpoint restore or replacement, not retry. Factor, Solve, and Spectrum keep
their public per-batch status Buffer while projecting its exact semantic
failure to a canonical terminal Reason; they never disguise that status as
generic `BackendFailed`.
`Ok` is the only success value; `ok()`, truth conversion, and the empty success
error all derive from it. Category meanings are defined by the checked-in
[Compute reference](https://github.com/sigee-min/runD/blob/main/docs/reference/compute.md#selection-and-errors).

This page is the Wiki integration contract. The repository's checked-in
`docs/reference/compute.md` and `docs/reference/compute/services.md`
remain the engineering source of truth for the complete primitive, graph,
cache, typestate, memory, Node-native, and backend parity rules.
