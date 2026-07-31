# Telemetry

This page owns the Session telemetry boundary shared by Compute and Replay.
Compute counter meaning remains in
[Compute](../../../docs/reference/compute.md), and replay evidence meaning
remains in [Replay](./replay.md). This page owns selection cost, event shape,
callback ordering, and the release parity gate.

## User Contract

One `rund::telemetry::Sink` is attached when a `Session` opens. The sink has
two levels:

- `Basic` emits stable identities, counters, byte totals, hashes, and the
  typed terminal outcome;
- `Detail` adds diagnostic phase durations.

`telemetry::bind(observer, level)` creates that same non-owning Sink from an
lvalue callable accepting `const Event&`. It allocates nothing and retains no
callable copy; the observer must outlive terminal work and Session close.
Rvalue binding is ill-formed, so the convenience cannot silently retain a
destroyed temporary.

`Sink` is not a public callback-pointer aggregate. Its callback, context, and
level storage are private; a default Sink is the sole disabled value, and
`telemetry::bind` is the sole public authority that can create a configured
Sink. Consumers can test whether it is configured and observe `level()`, but
cannot assemble mismatched callback/context storage or mutate the level behind
an open Session.

The SDK spelling is `rund::telemetry::Level::Detail`. The operator and harness
selector is exactly `telemetry:detail`. One configured Sink owns observation
level, callback, and timing selection together.
The Sink is immutable after a successful `open()`. Scope timing selection may
therefore read that configuration without taking the mutable runtime-state
mutex: Session open completes before scope admission, and no lifecycle
operation rewrites the Sink.
Session identity follows the same publication rule. The active diagnostic
scope is the only changing common field: scope admission publishes its
64-bit value with a release store and event completion reads it with an
acquire load. The validated `darwin-arm64` and `linux-x64` product targets
require that atomic to be always lock-free; no mutex-backed fallback exists.

Basic observation performs no telemetry-owned wall-clock reads. Detail may
read a monotonic clock, but that work cannot change admission, scheduling,
canonical order, hashes, counters, result status, or backend selection. The
release contract runs the same workload at both levels and requires those
semantic fields to match exactly.

An unconfigured Sink is a zero-dispatch path. Session open installs no Compute
emit callback or context, and `Runtime::emit` returns before callback dispatch,
telemetry mutex work, or trace mutation. `TelemetrySkipped` means a configured
sink failed or threw; its record carries the exact typed
`ReasonCode::TelemetrySinkFailed`. Disabled telemetry never emits that trace
event.

## Event Shape

`rund::telemetry::Event` is one tagged, trivially copyable value. It contains
common `source`, `level`, `session`, and `scope` fields and four nested
projections:

- `compute` owns its typed code, backend, graph, worker, tile, dispatch,
  command-submit, buffer allocation/reuse, SDK boundary-copy bytes, canonical
  graph-read bytes, and, at Detail, kernel sample/time and submit/wait time;
- `replay` owns its typed `replay::Code`, mode, prepared-plan state, canonical
  input rows and bytes, producer-emitted rows, choice count, evidence rows and
  bytes, retained bytes, copied bytes, prepared-storage growth, and the
  published result hash;
- `queue` owns the source-selected queue peak and its exact capacity: Compute
  uses accelerator command in-flight slots, while Replay uses the scope's
  maximum ready depth and configured ready-queue capacity;
- `detail` owns prepare, work, and finish nanoseconds.

The source tag selects the applicable projection. An inactive projection is
zero-initialized and has no semantic meaning. The nested shape prevents one
flat bag of similarly named counters from becoming a second Compute or Replay
authority.

`Event::error()` selects the active projection's typed code and returns its
allocation-free stable text. Event stores no common reason string: that would
be a second outcome authority beside `compute::Code` or `replay::Code`. This
text is the stable Compute category for a Compute event and the exact typed
Replay code text for a Replay event. A Compute `Status` remains the owner of
its operation-specific diagnostic; copying that borrowed text into Event would
make a trivially copied event retain an unsafe lifetime. Unknown source tags
and Replay codes fail closed to stable invalid diagnostics.

Replay count and byte fields have distinct meanings. `input_rows` is canonical
input consumed by the simulation. `produced_rows` is the subset emitted by an
authoritative producer, so it equals `input_rows` for Live and Record and is
zero for Replay and Scenario. `input_bytes` is canonical input
content consumed by the scope, `evidence_bytes` is canonical persisted
evidence, `retained_bytes` is immutable result storage charged after
publication, and `copied_bytes` counts bytes physically copied by runD at the
result boundary. For Spill records, `physical_bytes` is the exact runD segment
file length, `allocated_bytes` is the conservative filesystem-allocation charge
held by the shared storage Budget, and `reserved_bytes` is in-flight admission
that has not yet committed. Published records therefore report zero reserved
bytes. Memory records report all three as zero. Prepared scratch capacity is not
retained content and is not reported as such. A retained result containing `B`
independent bytes requires `Omega(B)` storage; telemetry distinguishes that
information-theoretic cost from avoidable scratch growth or duplicate copies.

## Actionable Findings

`Event::findings()` derives a `Findings` value directly from the raw Event. It
does not store another counter set, allocate, invoke the observer, or mutate the
run result. `Findings::Capacity` is five: at most one finding is published for
each `Cost` (`Allocation`, `Copy`, `Scan`, `Queue`, and `CriticalPath`). Zero
allocation, copy, or graph-read cost is absence of that finding, not a second
zero-valued counter authority. The critical-path finding is always last.
Construction is private to `Event`: callers can inspect and copy the bounded
value but cannot append a derived claim or create a second diagnostic authority.
The derivation has one compiled Node owner. Public consumers parse the Event and
Finding value shapes, not the complete decision tree, and every executable calls
the same implementation. Derivation is constant bounded work, and translation
units that include Session or Replay telemetry do not instantiate the decision
tree.
Stable diagnostic text uses the same compiled telemetry boundary. The public
headers declare the typed projection while one switch owner maps each enum or
active Event code to text, so consumers cannot instantiate divergent copies.

`telemetry::describe(finding, writer)` and
`telemetry::describe(event, writer)` provide the first presentation path. The
writer is a borrowed callable accepting `std::string_view`; the helpers retain
no callable or text, allocate no storage, and visit the fixed-capacity findings
in their canonical order. A Finding emits its cost followed by every cause and
action in stable enum order. An Event emits one such diagnostic per line. The
raw typed fields remain the evidence authority; `describe` is a streaming text
projection, not another finding or formatting buffer.

Every finding carries a unit, observed amount and `Accuracy`, a typed `Cause`,
and a typed concrete `Action`. A reference is present only when raw authorities
are meaningfully comparable in the same unit:

- Compute allocation events reference buffer-reuse events;
- Replay copied bytes reference immutable retained bytes;
- queue depth references the capacity of that same queue;
- a Detail critical-path duration references the saturating sum of its three
  non-overlapping phase durations.

Compute boundary-copy bytes deliberately have no reference. Canonical graph
read bytes and transfer bytes are both byte counts but do not describe the same
work, so `has_reference()` is false instead of publishing a misleading ratio.
Replay storage-growth events likewise have no byte reference. A source counter
at `UINT64_MAX` is marked `Accuracy::Saturated`; saturation never rejects or
changes graph execution.

The Compute scan amount is the saturating sum of every canonical Read access in
the compiled `graph::Info`:

```text
graph_read_bytes = sat_sum(read access size_bytes)
```

It is derived once with graph description and propagated unchanged through
CPU, Metal, and Vulkan `Stats`. It is exact for the canonical graph read
plan, not a claim about post-fusion device transactions, cache-line fills, or
hardware memory traffic. `Cause::GraphRead` and
`Action::ReduceGraphBound` make that boundary explicit.

Standalone `Job::profile().findings()` uses this same `Findings` type and the
same compiled `Event::findings()` decision owner. One compiled Stats projection
supplies both Profile and Session Compute events: boundary copies are
`sat(uploaded_bytes + downloaded_bytes)`, scan cost is `graph_read_bytes`, and
queue pressure is `command_inflight_peak / command_capacity`. Profile stores no
Event or derived counters. It materializes one bounded temporary projection,
derives the findings, and returns that trivially copyable value. A backend with
no timing source projects its critical path as unavailable rather than treating
zero timing counters as measured zero.

A queue finding is emitted only for the exact predicate
`capacity != 0 && depth >= capacity`. Its cause is `QueueAtBound`; the action
is `ReduceFanout`. `Reference::QueueCapacity` names the matching bound for both
queue kinds. The finding does not claim
that increasing capacity is safe or sufficient because independent admission
and resource bounds may own the pressure.

Basic has no timing evidence. Its critical-path finding is therefore
`Accuracy::Unavailable`, `Cause::TimingUnavailable`, and
`Action::EnableDetail`. Detail computes the exact maximum of prepare, work, and
finish. Every phase equal to that maximum is retained in the Cause mask, and
the corresponding Actions are retained in the Action mask; there is no hidden
tie breaker. `members(mask)` returns all known members in stable enum order and
`name(member)` returns stable allocation-free text. Both helpers have capacity
for the complete public enum, so a valid multi-member mask is never silently
truncated. `Members<T>::Capacity` is derived from that enum's authoritative
member array and construction is private to `members(mask)`; callers cannot
append an eleventh value or create a second truncation policy.
`name(Accuracy)` and `name(Reference)` are the same single allocation-free text
authority for evidence quality and comparison kind. Product diagnostics use
those projections rather than enum integers or caller-owned string switches.

For a Compute work-phase maximum, the finding distinguishes graph work from
submission overhead without a guessed threshold. When at least one command
submission and one kernel timestamp sample were observed and measured
submit/wait duration `S` is greater than cumulative measured kernel duration
`K`, the measured non-kernel interval is positive:

```text
submit_overhead = S - K > 0
```

That finding retains `Cause::Work`, adds `Cause::SubmitOverhead`, and selects
`Action::BatchJobs`. Otherwise it selects `Action::ReduceGraphBound` as before.
The comparison is made only from the raw `Stats` timing authorities projected
into the Event; no ratio, device-specific cutoff, or sampled recommendation is
stored. `BatchJobs` promises only that one submission can amortize per-submit
work across already-independent resident Jobs. It does not claim that queue
contention, kernel work, or dependency barriers disappear.

## Emission

Each terminal Compute submission emits one Compute event. Each Live, Record,
Replay, or Scenario scope emits one Replay event after its result and hashes
are final. Compute events caused inside a replay scope precede that scope's
Replay event because scope publication follows task drain.

The completion owner reads immutable Sink and Session identity directly,
acquires the active scope atomically when the producer did not supply it, and
invokes the callback outside runtime and scheduler locks. Event dispatch and
scope publication therefore take no runtime-state mutex; only the bounded
trace update after the callback serializes with runtime trace state. This
removes one mutex acquisition from every emitted event and one from every
successful scope publication as a structural operation count, not a wall-time
speedup claim. The same-Session reentry guard remains active during the
callback. A callback exception is contained, records `TelemetrySkipped` in
bounded trace, and cannot replace or mutate the completed operation result.
Event references are borrowed for the callback only.

`Runtime::emit` is the single internal callback, level-selection, Session-id,
reentry, exception, and trace owner for both projections. Compute and Replay
only populate their typed projection and submit the completed value to that
owner; neither invokes the configured sink directly.
The producer transfers its stack Event into that boundary by rvalue and the
boundary fills the common fields in place. No intermediate Event snapshot is
copied before the sink borrows the final `const Event&`; a consumer that needs
longer retention explicitly copies the trivially copyable value.

The event `scope` is a Session-local diagnostic correlation identity. It is
not persisted replay identity and never enters deterministic hashes. Replay
records use the canonical scope projection owned by the Replay contract.

## Detail Phases

Detail timing uses one monotonic clock and three non-overlapping intervals:

```text
prepare = enter the public verb, validate, build or load, and install the plan
work    = invoke the user callback and drain admitted work
finish  = canonicalize evidence and publish the immutable result
```

For Replay, one clock begins at the public Live, Record, Replay, Scenario, or
checkpoint-resume verb. The runtime marks the callback and drain boundaries;
the facade stops the clock only after comparison, canonical record storage,
hashes, and Event fields are final. Expected validation, cold projection build,
Scenario choice sorting, and restore therefore cannot fall outside the three
durations. Sink callback time is excluded because it is consumer-owned. The
three durations are non-overlapping and may be summed by a consumer; no stored
total repeats that authority. Early rejection reports its elapsed interval as
prepare and leaves work and finish zero. Compute does not add a second set of
clock reads or a second projection: the shared compiled Stats projection sets
`prepare_ns` to the saturating sum of its compile and setup source counters,
`work_ns` to submit/wait evidence when available and otherwise kernel evidence,
and `finish_ns` to readback evidence.
The complete Compute timing coordinates remain owned by `Job::profile()`.
A duration of zero at Basic level means unmeasured. `level` distinguishes that
state from a measured interval that rounded to zero.

The normal Detail scope reads the clock exactly four times: public entry,
work entry, finish entry, and final publication. The final timestamp is
captured before the read count is observed. This explicit sequence is the
single authority and cannot depend on a compiler's function-argument
evaluation order. Basic performs zero reads through the same owner.

## Verification

Release acceptance requires, on one installed artifact and source manifest:

1. Basic and Detail produce equal result status, deterministic hashes,
   ordering, and stable counters for Live, Record, Replay, and Scenario;
2. Basic performs zero telemetry-owned monotonic-clock reads;
3. one terminal operation emits one event, callback exceptions are contained,
   and same-Session callback reentry fails without deadlock; an unconfigured
   Sink performs no dispatch, telemetry lock, or trace write;
4. Record reports producer count one, Replay and Scenario report zero, and
   telemetry agrees with retained evidence sizes;
5. capacity much larger than payload reports retained `Theta(B)`, not scratch
   capacity, and 100 warm scopes report zero prepared-storage growth;
6. the exact `telemetry:detail` harness selector runs the parity contract and
   is the sole detail selector in the public command registry;
7. Event stores no reason string, and `error()` projects exactly from the
   source-selected typed code without allocation;
8. actionable findings are allocation-free, never exceed five, retain every
   critical-path tie, preserve all non-time findings between Basic and Detail,
   map only raw counters plus declared bounds to causes and actions, and
   `Job::profile().findings()` equals the same canonical Event projection;
   a Compute work maximum with `command_submits != 0`, `kernel_samples != 0`,
   and `submit_wait_ns > kernel_ns` reports `submit-overhead` and `batch-jobs`,
   while equality or a missing sample/submission does not invent overhead;
9. the installed focused Session/Replay entries expose the trivially copyable
   fixed-capacity finding values and allocation-free `describe` projection
   transitively; the support leaf is not a direct SDK entry, while caller
   `append` and unbounded `push_back` construction remain ill-formed; a
   configured Sink can be created only by binding an lvalue observer.

The installed [`telemetry.cpp`](../../../package/tests/consumer/example/telemetry.cpp)
executes one Basic Live observation and consumes its actionable findings.
Exact Basic/Detail parity across Live, Record, Replay, Scenario, and Compute is
owned by `runtime.task.replay.telemetry` and the `telemetry:detail` route.

Performance evidence follows the single
[telemetry overhead method](../../../docs/reference/performance/method.md#telemetry-overhead-method)
and reports the measured Detail delta rather than hiding it. Passing semantic
parity alone is not a performance claim.
