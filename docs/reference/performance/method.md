# Performance Method

This page and [`baseline.tsv`](./baseline.tsv) are the product performance
baseline authority. The five installed-Release measurement routes compare
their output before a successful evidence packet can be published:

- `tools/measure/scheduler/run`
- `tools/measure/compute/run`
- `tools/measure/flow/run`
- `tools/measure/graph/services/run`
- `tools/measure/telemetry/run`

`tools/internal/measure/compare` is the sole measurement-log parser and
comparison owner. `tools/internal/measure/schema.pm` is the sole baseline
schema, host/profile selection, route cardinality, canonical-order, and
immutable-packet seal owner. The shared `tools/internal/measure/finish`
boundary invokes the comparator once; individual routes do not mirror either
policy.

For an explicit baseline review, appending the read-only `observe` operand
prints the selected profile followed by the parser's canonical
`metric/value/unit` projection. Every unit comes from the route parser's
measured schema; the command never copies a label from the baseline. Normal
comparison rejects a baseline unit that differs from that observed unit. It
accepts only the already-declared metric set and never edits the baseline,
installs an artifact, or admits a host. Baseline cutting can therefore reuse
the exact release parser without adding a second route parser or a hidden
update mode.

Every measurement number, including raw samples, diagnostic spans,
throughput, ratios, counts, and memory facts, uses this grammar:

```text
(0|[1-9][0-9]*)(\.[0-9]+)?
```

It is a finite, nonnegative decimal with no sign or exponent. `NaN`,
infinities, negative values, empty required values, duplicate columns or keys,
malformed pairs, unknown row kinds, and injected `baseline` rows are rejected
before semantic projection. A value excluded from the timing budget is still
validated; diagnostic status never makes corrupted telemetry admissible.

## Frozen Point

The selected profile's `baseline` and `environment` rows in `baseline.tsv`
are the sole exact source-manifest and host-fact authority. This page does not
mirror those changing values. An unmatched kernel or machine fails closed
instead of silently borrowing another environment's timing budget. A new host
requires a reviewed profile in the same table.

The `baseline` rows retain the pre-edit product manifest and three raw-log
SHA-256 identities per route. Packets 1 and 2 are the adjacent same-machine
calibration observations `x_1` and `x_2`; packet 3 owns the frozen observation
`B`. Adjacency is structural: the three inputs must occupy consecutive
positions in that route's timestamp-named evidence-packet directory. A cut
cannot skip an intervening evidence packet and cherry-pick a quieter sample.
The `value` of each `upper` row is packet 3's exact canonical value, and
`D` is the observed calibration spread

```text
D = max(|x_1 - B|, |x_2 - B|).
```

All three packets must expose the same canonical metric and unit set. The
projector stores the final admission allowance `A`, not raw spread, in the
`envelope` column. The comparator has no hidden multiplier:

```text
L = B + A.
```

The projector derives `A` by cost class:

| Cost class | Admission allowance |
|---|---:|
| Scheduler bytes, Telemetry allocations | `max(0.10 B, 1.5 D)` |
| Flow frontend time in milliseconds | `max(0.25 B, 5 D)` |
| Scheduler, Compute, Flow graph construction, graph-service time | `max(2.5 B, 5 D)` |
| Telemetry cold time | `max(0.25 B, 5 D, 100000 ns - B)` |
| Telemetry warm time | `max(0.25 B, 5 D, 5000 ns - B)` |

Negative budget terms are clamped to zero. The count rule preserves the exact
zero-allocation gate. Telemetry uses explicit product budgets: at most 100
microseconds of cold instrumentation overhead and 5 microseconds after
preparation. Flow's millisecond frontend retains a 25-percent regression
effect size; its sub-microsecond graph-construction timers join the short-span
class. Short cross-thread, graph-construction, and driver timings use a
3.5-times limit because the sustained M4 Pro verification sequence exhibited
a 3.04-times scheduler phase shift while all structural counters and semantic
hashes remained identical. The five-spread term dominates when the three
calibration packets were noisier.

These are deterministic engineering guardrails, not claimed confidence
intervals: three adjacent packets cannot establish a distributional tail
probability. Improvements always pass because timing and memory rules have
only an upper bound. The comparator evaluates the decimal formula with
arbitrary precision, so binary floating-point rounding or overflow cannot
change admission at the boundary.

Every profile has exactly sixteen provenance identities: `manifest`, then
three raw logs each for scheduler, Compute, Flow, graph services, and
telemetry. The numbered identity is the SHA-256 of the canonical raw-log file,
not its source manifest and not an arbitrary 64-hex label. It also has the
exact host facts required by its operating system and one semantic SHA-256 for
each of the five routes. Missing or extra
identity and environment rows fail the whole baseline table, including when
the malformed row belongs to a profile other than the current host. This
prevents a partial profile from becoming an accidental fallback authority.

Rows are canonical, not merely set-equivalent. Profiles are lexicographic;
within one profile the order is `baseline`, `environment`, scheduler, Compute,
Flow, graph services, telemetry. Identities use the order above, environment
uses `system`, `release`, `machine`, `workers`, then the Darwin-only `model`
and `cpu`, and every route has `semantic` followed by lexicographically sorted
upper metrics. `workers` is a positive canonical integer and every text host
fact is nonempty. A reordered row, zero worker count, empty text fact, missing
metric, or extra metric invalidates the whole table.

The Darwin product schema has exactly 325 data rows: sixteen identities, six
environment facts, and route cardinalities 95 scheduler, 145 Compute, 5 Flow,
9 graph services, and 49 telemetry. The header is additional. Each route count
includes its one semantic identity. `tools/internal/measure/schema.pm` is the
executable cardinality authority; this paragraph records its reviewed Darwin
projection rather than defining a second count.

## Exact Meaning

Every route also has one `semantic` SHA-256. The comparator constructs it from
the ordered measurement schema and every non-performance field. It therefore
fixes row admission, workload identity, status, counts, hashes, worker shape,
warm-cost counters, dispatch shape, memory ownership, and units exactly.
Wall time, CPU time, raw allocation observations, Detail phase durations,
derived throughput and ratios, and driver timing spans are not part of that
digest. They have one authority: their declared one-sided performance rules.
Including them in the exact digest would reject an improvement before the
upper rule was evaluated. Copied bytes remain an exact semantic invariant
through the independently reconstructed public and storage projections.

The graph-service asynchronous same-key row has one schedule-dependent
diagnostic split. For `R=32` requests, the parser requires exactly one cache
miss and proves `waits + hits = R - 1 = 31`. Whether a reuse observes the
in-flight compile as a wait or the completed entry as a hit depends on host
scheduling, so the two individual values remain in the raw evidence but are
excluded from semantic identity. Their field names, request count, miss count,
units, and exact sum remain semantic. This prevents an OS scheduling choice
from impersonating a different graph workload without weakening the
one-compile contract.

Run-specific source-manifest and executable hashes are provenance, not
workload semantics. The comparator validates their exact SHA-256 shape and the
evidence packet seals their values, but excludes those two rows from the
Telemetry semantic digest. Otherwise editing this baseline would change the
source manifest, which would change the semantic digest and require another
baseline edit indefinitely. A different valid run identity must preserve the
same semantic digest; a malformed identity still invalidates the measurement.

End-to-end medians are the timing authority. Compute driver spans such as
kernel and submit-wait duration remain diagnostic telemetry; independently
gating both a span and its enclosing median would create two performance
authorities for one execution. Scheduler process-memory rules use total RSS
and heap growth; their component readings remain diagnostic while frame and
capacity facts stay exact.

Diagnostic spans indicate optimization direction, while runD's checked
counters and end-to-end medians remain the sole product evidence.

Physical Scan width is fixed at 128 lanes on Metal and Vulkan. The lowering
does not choose a width or alternate algorithm from the input count. For a
logical block of `B` elements, lane-local work is `ceil(B/128)` contiguous
elements followed by one fixed lane-prefix tree and a contiguous materializing
sweep. Current-source collective diagnostics must compare graph/output hashes,
dispatches, warm-zero counters, end-to-end median, and kernel span together;
the kernel span alone is not a product speedup claim.

## Compute Backend Method

Compute measures CPU, Metal, and Vulkan independently. Semantic graph
identity, numeric policy, stable ordering, output hash, and bounded logical
count are common; shader structure, workgroup geometry, prefix hierarchy,
scratch shape, and dispatch count are backend-owned. No backend is slowed to
match another backend's portability constraints.

The Compute raw log's `environment` record is the physical-path identity for
interpreting a backend row. In particular, a Vulkan record whose `driver` is
`MoltenVK` executes the Vulkan API, generated SPIR-V, descriptors, command
buffers, barriers, and synchronization through MoltenVK's Vulkan-to-Metal
translation on the selected Apple GPU. Its timing includes that translation
layer and the underlying Metal driver. Such a row is valid evidence for Vulkan
lowering and command-structure correctness, descriptor/command behavior on
that path, output parity, and same-driver-path regression. It is not native
Vulkan-driver throughput evidence and cannot support a native Vulkan ranking,
speedup, or portability claim.

On the same Apple host, a Metal record whose `driver` is `Metal` is native
Metal throughput evidence for that exact device, OS, driver path, source
manifest, and workload. A Metal-versus-MoltenVK timing delta compares those
two complete software paths on one GPU; it does not isolate API overhead and
must not be relabeled as Metal versus native Vulkan. Native Vulkan throughput
requires a separately admitted native-Vulkan host profile and measurement
packet. Same-path regression additionally requires the sealed backend name,
driver, and driver-details identity to remain the same; a driver-path change is
a new environment, not a performance regression sample.

The optional current-source `--resident`, `--collective`, `--sort`, `--bulk`,
`--batch`, `--pipeline`, `--pipeline-profile`, `--recurrence`, and
`--window-repeat` diagnostics are intentionally outside the installed-Release
baseline route. `--resident`
isolates resident creation at 1,024 and 1,048,576 elements, validates the first
execution and output, and reports CPU plus one selected backend without
changing the canonical algorithm or memory placement. `--bulk` compiles one canonical
512-by-512 fixed matrix multiplication and one canonical `N = 2^20` fixed
Fourier transform, runs one CPU oracle sample, then reports the median of
fifteen selected-backend samples. Every accelerator row must match the CPU
graph and output hashes and must observe zero warm pipeline compiles,
allocations, uploads, and downloads. It never selects an algorithm from the
workload size: the same transform schedule owns every admitted power-of-two
count.

The installed-Release measurement executable includes only the installed
public SDK. Internal matrix-tile and transform-stage cost contracts are visible
only to the monorepo `RUND_COMPUTE_FOCUS` target that owns focused modes; the
installed executable does not accept focused modes. This keeps private Kernel
headers out of the SDK consumer boundary and makes a stale internal include a
compile failure instead of an undeclared package dependency.

`tools/measure/compute/run --sort <backend>` uses the same dense and bounded
sparse Sort workloads, result-hash parity, resident preparation, and zero-warm
work checks as the installed route, but reports fifteen selected-backend
samples and only one CPU oracle sample. It exists so a Sort source edit rebuilds
and measures the focused Compute closure instead of rerunning unrelated
families. It is diagnostic evidence rather than a baseline update path; the
installed Release route remains the publication authority.

`tools/measure/compute/run --pipeline <metal|vulkan>` executes a bounded
current-source comparison for one selected accelerator. It does not invoke the
evidence finisher and cannot replace the installed route's frozen-manifest
evidence. Pipeline enters that route only after a matching host profile has
three complete calibration packets and an explicit reviewed baseline cut.

`tools/measure/compute/run --pipeline-profile <metal|vulkan>` isolates the
opt-in step-profile cost on that same current-source boundary. It prepares two
otherwise identical three-Program Pipelines, with separate intermediate and
output buffers, under `PipelineProfile::None` and `PipelineProfile::Steps`.
After one warm-up of each Pipeline, six `A, B, B, A` cycles produce twelve
`run()` wall samples per mode. The caller-provided three-row `profile()` copy
runs only after an enabled timed run; its `observation` duration is reported
separately and is never included in either run-wall sample.

The focused row is admissible only when output, Pipeline fingerprint, backend,
terminal identity, one-submit/three-dispatch topology, and all three declared
profile rows agree. Every warm run must report zero compile, allocation,
descriptor, upload, download, and roundtrip activity. Metal retains zero
additional commands and at least `64D` allocated step-control bytes. Vulkan
reports either `3` commands and `64D` bytes without timestamp support, or
`4 + 2A` commands and `64D + 16A` bytes with timestamps, for `D` declared and
`A` active Programs. The absolute wall delta is paired with an explicit
direction so the measurement grammar remains nonnegative.

Bulk byte accounting separates complex value traffic from logical twiddle
reads. For element width `E`, dispatch count `P`, and `S = log2(N)` stages,

```text
value_bytes       = 4 N E P
coefficient_bytes = S (N / 2) (2 E) = S N E
logical_bytes     = value_bytes + coefficient_bytes.
```

The first term counts split real/imaginary reads and writes once per dispatch;
the second counts one cosine and one sine read per logical butterfly. These are
algorithmic byte counts, not a hardware-cache traffic claim. Cache-line reuse,
compression, and driver behavior require hardware counters and therefore must
not be inferred from them. End-to-end wall median remains the throughput
authority; kernel and submit-wait medians remain diagnostics.

Every environment, resident, warm, workload, and orchestration row names one
of those three backends, and the mixed row names exactly their complete
set. Their canonical metric keys retain that backend segment. A backend-free
row in one of those families has no measurement owner and is rejected rather
than inheriting another backend's limit.

The dense Sort row uses 262,144 descending U32 values. The bounded sparse row
uses the same physical capacity, a stable one-of-sixteen filter, and 16,384
active values. Each Program is compiled and made resident before one untimed
warm-up. Five measured runs time only `Job::run()`, and validation plus explicit
readback occur outside the interval. Each validation point reads the output
exactly once; the same final value view owns semantic validation and output-hash
publication. A row is admissible only when its output is
sorted, its output hash matches the CPU reference, it submits one prepared
command stream, and every measured warm run has zero pipeline compiles, buffer
allocations, uploads, and downloads.

For radix pass `d`, native GPU stability is checked by

```text
destination = bucket_base + prior_block_count + prior_equal_lane_count.
```

The cross-block contract also uses repeated equal keys and original ordinal
values, so a fast but unstable scatter or reuse of an original-block histogram
cannot pass by producing only the right key multiset. End-to-end median is the
performance authority; kernel and submit spans remain diagnostic.

The Vulkan in-flight row measures one bounded command envelope twice over the
same prepared Jobs and canonical input. Serial execution submits and waits for
each Job before the next submission. Concurrent execution submits the complete
envelope before waiting in submission order. The completion service may retire
an earlier slot while the caller is still submitting later Jobs. If claim and
retirement times are `c_i` and `r_i`, observed occupancy is
`P = max_t sum_i [c_i <= t < r_i]`; submission order alone does not prove
`c_k < r_1`. The admissible asynchronous measurement is therefore
`0 < P <= C` for command capacity `C`, with zero admission rejections. Exact
capacity saturation and ninth-submission rejection remain owned by the gated
`runtime.compute-accel` contract, which prevents retirement until all eight
slots are claimed.

The two end-to-end medians are independent upper metrics. Jobs per second,
items per second, speedup, and observed in-flight peak are schedule-dependent
diagnostics and cannot gate or alter semantic identity. Envelope size, input
count, sample count, command capacity, rejection count, command submits,
dispatches, warm-cost status, and graph/output hash parity remain in the exact
semantic projection. Changing this metric set requires a complete five-route
calibration cut; every timing row must be measured by that cut.

Every concurrent Job contributes completion-local command evidence. The
measurement folds independent Jobs with the product monoid

```text
(submits, dispatches, rejections, capacity, peak)
  = (saturating sum, saturating sum, saturating sum, max, max).
```

The identity is `(0, 0, 0, 0, 0)`. Addition is associative under unsigned
saturation and `max` is associative, commutative, and idempotent, so the
projection is independent of host completion order. No Job submission starts a
global telemetry epoch; doing so would first wait for older fences and collapse
the bounded envelope to one effective slot.

The Metal and Vulkan Batch rows measure the same 64 prepared Jobs and
canonical 64-element input twice. Serial execution runs and waits for every Job
in order; Batch execution admits that exact retained set to one
`compute::Batch` and requires one native command submission. Eight paired samples alternate
serial-then-Batch and Batch-then-serial, four times each, after both paths are
pre-warmed. They produce independent serial and Batch end-to-end wall medians;
the per-pair speedup median is diagnostic. Those two wall values are the only
new baseline timing authorities. Jobs per second and speedup are derived
diagnostics.

Every measured run has zero warm compilation, allocation, download, and upload
counters. The serial submit total is exactly 64, the Batch submit total is
exactly one, and the pre-read Job snapshots contain zero copies of that shared
submit. Every Job's graph and output hash must match its own serial result, all
64 outputs must match the canonical value, and Metal and Vulkan hashes
must agree.

The general Compute workload's `warm_zero` is the conjunction of zero pipeline
compiles, buffer allocations, descriptor-pool creations, descriptor-set
allocations, uploads, download events, and downloaded bytes. Internal and
external producer/consumer roundtrip bytes are algorithmic traffic executed by
the prepared graph, not warm setup or host-transfer churn; they remain explicit
per-run diagnostics and do not alter `warm_zero`. Focused Pipeline rows retain
their stronger, separately named zero-roundtrip contract because their three
unary steps are required to remain resident without materialization.
Explicit serial and Batch oracle runs and reads happen only after every timed
pair, so readback and validation cannot bias either measured order.

For each sample, the route also sums driver submit-wait and kernel spans and
records `host_residual = max(wall - submit_wait, 0)`. Their medians are
diagnostic and excluded from both semantic identity and baseline metrics. A
dominant host residual after one-submit batching points to pre-recording work,
such as Metal indirect command buffers. Vulkan prepared Jobs already retain one
secondary command buffer, so their warm command construction is one primary
timestamp/execute wrapper independent of prepared dispatch count. A dominant
Vulkan host residual therefore points above native step encoding. A dominant
submit-wait span instead confirms native submission amortization as the active
structural lever.

The Metal and Vulkan Pipeline rows reuse the exact same three compiled unary
Programs and four caller-owned 4,096-element `I32` Buffers for both paths. The
Programs implement, in declaration order, `3x + 1`, `5x - 7`, and `2x + 11`;
the expected final value is therefore `30x + 7`. Serial execution invokes each
Program against those Buffers and waits three times. Pipeline execution invokes
the prepared three-step owner once. Metal identifies the reusable indirect
command-buffer path as `reusable_icb`; Vulkan identifies its retained immutable
primary path as `immutable_primary`.

Both paths are pre-warmed. Twelve paired execution samples alternate
serial-then-Pipeline and Pipeline-then-serial, yielding repeated ABBA order with
six first positions per path. The serial and Pipeline wall medians are
independent focused diagnostics. Submit-wait and kernel spans are diagnostic.
Pipeline claim and status-control spans are also diagnostic because they may
overlap the enclosing wall, kernel, or submit-wait interval and must never be
added into a fabricated decomposition.

After all execution pairs, a second twelve-pair ABBA schedule reruns each path
and times only its explicit typed output read. Those two read wall medians are
separate focused diagnostics; driver readback spans remain diagnostic. Every
read validates `30x + 7`, and an endian-independent content hash must match
between serial and Pipeline for every sample.

The parser admits only `count=4096`, `samples=12`, three serial submits, one
Pipeline submit, three dispatches on both paths, three steps, four resources,
two hazard barriers, no claim conflict, a fully verified prefix, no failed
step, zero Map status entries, and a 128-byte terminal control observation.
This all-Map workload has no imported or private replacement status words and
no bounded-control telemetry sources, so Metal and Vulkan each report exactly
two control commands: one prepared open and one terminal close. Before each
read, warm compilation, Buffer allocation, descriptor-pool creation,
descriptor-set allocation, upload, download, and internal/external round-trip
counters are all zero. These fields, the complete Pipeline fingerprint,
command-path name, and content identity are exact semantic evidence. The
current-source build
manifest binds both rows and the separate-Program comparator to one source
state; the raw CSV is not an installed baseline packet.

`tools/measure/compute/run --recurrence <metal|vulkan>` isolates the prepared
Map recurrence lowering on the same current-source boundary. It compiles one
4,096-element `I32` Program implementing `x + 1`, executes exactly 256
iterations, and compares two routes: 256 independently submitted Program runs
and one `Pipeline::repeat<256>` run. Twelve paired samples alternate order six
times each after both routes are warm. Typed readback occurs outside the timed
pairs.

The row is admissible only with identical nonzero content hashes, 256 serial
submits and dispatches, one recurrence submit, one recurrence dispatch, one
logical Pipeline step, a fully verified prefix, and zero warm compile,
allocation, descriptor, upload, download, and round-trip counters. The logical
Pipeline hazard count remains 255 because it records the authored recurrence
proof. The element-local lowering removes those physical inter-iteration
barriers; it does not rewrite the public logical plan.

For `N` iterations, `E` elements, carried payload `S`, invariant payload `C`,
output payload `O`, and `W` device-capacity windows, the serial prepared
command shape has `N * W` Map dispatches and exposes
`Theta(N * E * (S + C + O))` payload loads and stores. The proved
single-dispatch element-local recurrence has `W` dispatches and exposes
`Theta(E * (S + C + O))` payload loads and stores while retaining the same
`Theta(N * E)` arithmetic in the same per-element iteration order. Dispatch
and generated-program payload operations improve by the exact factor `N`.
Physical memory traffic and the reported wall ratio remain measurements
because compiler register allocation, spills, arithmetic, occupancy, cache,
driver, and submission costs are device facts.

`tools/measure/compute/run --window-repeat <metal|vulkan>` fixes
`Max = 516096`, `Tile = 1024`, `K = 504`, and `N = 64`. Its serial comparator
precomputes the same 504 tile seeds outside the timed region, then times
32,256 independently submitted one-Action Pipeline runs. The nested route
times one complete Seed/Action/Fold Pipeline execution. Twelve samples use
balanced ABBA order after both routes are warm; observation and the serial
outer Fold occur after timing.

The nested row is admissible only with 571 retained route templates, 33,264
authored Seed/Action/Fold occurrences, 504 executed outer windows, 32,256 executed inner
iterations, one nested submission, no failed coordinate, exact serial/nested
result parity, and zero warm compile, allocation, upload, download,
binding-mutation, and fallback evidence. An eligible status-free element-local
Action must additionally use the common tile-transducer proof and the `K * 3`
physical Seed/transducer/Fold Program-occurrence shape; the authored count does
not expand the native stream. The CSV
`*_warm_binding_mutation_count` columns are the public
`PipelineStats::rebinding_count`: they count post-prepare retained-binding
mutations, not cold native capture or emission of frozen descriptors. Their
zero is interpreted with the `compute.window` owner/View identity snapshot and
is not standalone no-rebinding proof. Program-internal normalization traffic
remains visible through the round-trip counters. The measured dispatch count
is reported independently because Seed, Action, and Fold Programs may each
lower to more than one native dispatch. The wall ratio therefore measures
submission, control, and backend command-path effects for this declared
workload; it is not an algebraic claim about every Action body.

Metal rows on Apple identify the reusable ICB path. Vulkan rows whose
environment reports MoltenVK prove the Vulkan API, SPIR-V, descriptor,
push-constant, command-buffer, and barrier path over that translation layer;
they are not native Vulkan throughput evidence.
The Metal wall time includes the hard-cut executor's remaining host envelope:
one outer command-buffer/encoder lifecycle, one bulk resource-residency call,
one full-ICB range call, commit/completion, and fixed control observation. It
contains no runD command/range/binding/state traversal, but it also includes
the device cost of issuing frozen commands whose inactive payload threads
return through uniform guards. One nested submit and zero binding mutation
must therefore be interpreted with both the structural hard-cut contract and
the measured wall result, not as literally zero host or inactive-device cost.

## Telemetry Overhead Method

`tools/measure/telemetry/run` measures the complete public Replay path, not one
proxy operation. Its eight independent workload groups are the Cartesian
product

```text
{Live, Record, Replay, Scenario} x {cold, warm}.
```

Live measures the canonical input boundary without persistence. Record adds
evidence production. Replay consumes a prepared Record without invoking the
producer. Scenario consumes that same Record with one canonical Choice. The
producer count is therefore exactly one for Live and Record and zero for
Replay and Scenario. All four operations use the same 32-byte input, source,
sequence, seed, worker width, capacity, retention, and memory storage policy.

Each workload is measured at Disabled, Basic, and Detail. Disabled installs no
sink or callback, Basic emits one Replay event, and Detail emits the same event
plus its non-overlapping phase durations. The log seals both the source
manifest and measured executable SHA-256, so values from different artifacts
cannot form one comparison.

A cold interval includes Session construction, open, exactly one operation,
the single drain-and-stop `close()`, and destruction. For Replay and Scenario, decoding the expected
Record is deliberately outside the interval: the sample measures product
execution rather than benchmark fixture decoding. A warm interval contains
only one operation on an already-open Session. Session open, expected Record
decode, two warm-ups, the single `close()`, and destruction are outside the warm
interval. For each independently loaded Replay and Scenario lane, the first
warm-up builds the expected Record projection, the second warm-up proves reuse,
and every enabled measured warm event must report `Reused`. Enabled Live events
report `None` and enabled Record events report `Built` in every lifecycle
because neither consumes a prepared expected Record. Disabled installs no sink
and therefore retains the all-zero event projection. Warm execution must report
zero storage growth.

Every lifecycle has two untimed warm-up rounds and twelve measured blocks. The
four operations use this Williams order, repeated three times:

```text
Live     Record   Scenario Replay
Record   Replay   Live     Scenario
Replay   Scenario Record   Live
Scenario Live     Replay   Record
```

Thus every operation occupies each ordinal position three times. Within each
operation block, the twelve level triplets repeat all six permutations of
Disabled, Basic, and Detail twice. Every level occupies each ordinal position
four times; Basic precedes Detail in six pairs and follows it in six. Pair IDs
are retained through samples and deltas; summaries stay scoped to one operation
and lifecycle. No aggregate across those groups may hide a slow path.

Every sample records steady-clock wall nanoseconds, `RUSAGE_SELF` user-plus-
system CPU time converted exactly from its microsecond fields to nanoseconds,
process-wide C++ allocation calls across all threads, and
`StorageReport::copied_bytes`. Detail also publishes its non-overlapping
prepare, work, and finish nanoseconds; Disabled and Basic must publish zero for
all three. For twelve sorted values, the median is the
exact arithmetic mean of positions six and seven and nearest-rank p95 is
position twelve because `ceil(0.95 * 12) = 12`. Half-integral medians use a
`.5` suffix; integer arithmetic prevents binary floating-point rounding.

The log publishes absolute observations, paired Detail-minus-Basic deltas, and
paired Basic-minus-Disabled reference deltas for each operation and lifecycle.
Because the numeric grammar has no signed values, a delta is a direction plus
a nonnegative magnitude. Only the positive part of the Detail-minus-Basic
median and p95 for wall time, process CPU time, and allocation count enters the
release gate. This yields `4 * 2 * 3 * 2 = 48` timing rules. Disabled-to-Basic
deltas and absolute values remain diagnostic; copied bytes are an exact
semantic invariant rather than a second performance authority.

The measurement independently reconstructs the public Replay projection. All
additions below saturate at `UINT64_MAX`. For Record, Replay, and Scenario:

```text
input_rows     = Record::input_count()
input_bytes    = StorageReport::logical_bytes
produced_rows  = input_rows for Record, otherwise 0
choices        = 1 for Scenario, otherwise 0
evidence_rows  = observations + host_events + input_rows + traces + captures
evidence_bytes = StorageReport::encoded_bytes + Record::capture_report().retained_bytes
retained_bytes = StorageReport::retained_bytes + Record::capture_report().retained_bytes
copied_bytes   = StorageReport::copied_bytes + Record::capture_report().retained_bytes
physical_bytes = StorageReport::physical_bytes
allocated_bytes = StorageReport::allocated_bytes
reserved_bytes = StorageReport::reserved_bytes
storage_growths = StorageReport::growths
result_hash    = Record::hash()
```

Here `observations`, `host_events`, `traces`, and `captures` come from their
public Record counts, including `Record::captures().size()` for the final term.
The capture archive is disjoint from `StorageReport`, so its retained bytes are
added exactly once rather than hidden in or double-counted with storage.

Live has one 32-byte input and one produced row. Its evidence row count is
`observations + host_events + traces`; persisted-byte, growth, and result-hash
fields are zero. Every enabled event must equal these independently derived
fields exactly. It must also carry Replay source, the requested operation,
the expected preparation state, typed result code, and nonzero Session and
scope identities. Disabled must preserve the same public result while emitting
an all-zero event projection. This makes the measurement an executable oracle
for user-facing telemetry, not a self-comparison of event fields.

The sample schema exposes decision counters directly; it does not compress
them behind a benchmark-local counter hash:

| Signal | Bottleneck question | User decision |
| --- | --- | --- |
| `operation`, `lifecycle` | Which user path and ownership boundary is expensive? | Compare the matching group; never average cold setup into warm throughput. |
| `result:*`, `public:input:*`, `public:produced:rows` | Is canonical input or generated evidence growing? | Revisit the capture boundary, schema, or batching. |
| `storage:logical:bytes`, `storage:encoded:bytes`, `storage:retained:bytes`, `storage:cached:bytes` | Is amplification in content, encoding, retention, or prepared cache? | Tune the bound at the owning storage layer. |
| `storage:physical:bytes`, `storage:allocated:bytes`, `storage:reserved:bytes` | Is owned file content, charged filesystem allocation, or in-flight admission consuming the hierarchy? | Inspect the matching Session, tenant, and root Budget reports before changing retention. |
| `capture:*`, `public:evidence:*` | Are captures or evidence dominating retained data? | Narrow capture policy or shorten its retention window. |
| `public:copied:bytes`, `event:copied:bytes` | Are physical boundary copies growing faster than canonical evidence? | Inspect materialization and chunk boundaries. |
| `storage:growths`, `storage:chunks`, `storage:segments` | Is warm execution repeatedly expanding prepared storage? | Resize the bounded storage plan; warm growth should be zero. |
| `storage:hits`, `storage:misses`, `storage:evictions`, `event:plan` | Is the prepared plan or storage cache being reused? | Keep the owning Session warm or adjust the bounded cache. |
| `telemetry:detail:prepare:ns`, `telemetry:detail:work:ns`, `telemetry:detail:finish:ns` | Which non-overlapping Detail phase dominates? | Optimize that phase without creating a second release gate. |

Semantic parity is checked before any statistics are printed. Basic and Detail
must match typed status, input ordering and transcript hashes, result hash,
public storage and capture reports, source, Session and scope identities, and
the complete Replay projection. Their requested levels differ; Basic must
report zero for all `telemetry:detail:*` fields. Disabled must match the same
public result while producing zero callbacks. Any mismatch invalidates the
whole measurement before it can be averaged into an overhead result.

## Evidence Publication

A measurement packet contains the unmodified route log and a separate
`baseline.log`. `run.tsv` seals the route, current host, both file names, and
both SHA-256 digests. It also seals the workload result independently as
`workload:status` and `workload:exit`; `passed` is valid only with exit `0`,
and `failed` only with a canonical process exit in `[1, 255]`. The selected
profile, compared metric count, and comparator result live under the
`proof:*` hierarchy. A comparator failure can therefore never rewrite a
successful workload into a failed workload or erase its exit status.

The packet also seals the selected environment profile, measured executable
name and SHA-256, configured generator, compiler path and compiler SHA-256,
source identity, revision, and dirty state. Its `run.tsv` field order and set
are exact. The retained source manifest, source identity, raw log, and
comparison result are regular non-symlink files whose recorded SHA-256 values
must match; unknown packet entries are rejected. A successful proof is exactly
one canonical comparator row. A failed calibration proof retains its complete
diagnostic bytes and uses profile `-` and metric count `0`, while the
independently selected packet profile remains exact.

The configured CMake cache is the single compiler authority. Flow frontend
measurement and packet publication both resolve that same executable through
`tools/internal/measure/compiler`; changing `${CXX}` after configuration
cannot measure one compiler while sealing another.

Evidence status accepts the packet only when all fields are unique, the host
is the current host, both files and hashes match, and a fresh comparator run
over the recorded raw log is byte-identical to the recorded result. A copied
packet, deleted measurement log, edited result, stale profile, or changed
baseline therefore cannot report `passed`.

Each route publishes an atomic attempt marker after acquiring its build locks
and before preparation. `running` reports `in-progress`. Preparation and
pre-workload frontend failures occur before `finish`, publish no packet, and
leave the marker as the sole `failed` result. Once execution reaches `finish`,
the packet preserves the workload result even when comparison fails. Any
execution, comparison, or publication failure still leaves the attempt marker
failed. A completed packet is visible only after its attempt marker is
atomically cleared, so an interrupted process cannot publish a successful
result for an unfinished attempt.

## Update Contract

There is no update flag, placeholder table, discovery fallback, or implicit
host admission. Changing a workload, exact semantic projection, baseline
value, calibration envelope, or environment profile requires an explicit
reviewed edit to `baseline.tsv`. A baseline cut must name its source manifest,
retain the raw measurement packets, and pass the positive and negative
`tools.measure` contract. Deleting a metric is also an explicit cut:
unbaselined output and missing baseline rows both fail.

A cut uses three sequential executions of every route at one stable product
point `M0`. Packets 1 and 2 supply `x_1` and `x_2`; packet 3 supplies `B`.
All fifteen packets must seal `workload:status=passed` and
`workload:exit=0`. Their comparator proof may be `failed`: before the table is
cut, an M0 observation is evaluated as calibration input rather than Release
admission. This exception applies only to calibration admission;
ordinary release evidence still requires a passing comparator proof.
After the explicit table edit, the product manifest becomes `M1`. The table's
`manifest` identity remains `M0`, because it identifies the measured product
point; rewriting it to include its own edit would be a circular hash demand.
Release is then rebuilt at `M1`, and every route runs once more against the
new table to publish ordinary same-`M1` admission evidence.

The stdout-only candidate projector is:

```text
tools/internal/measure/project ROOT PROFILE \
  scheduler1 scheduler2 scheduler3 \
  compute1 compute2 compute3 \
  flow1 flow2 flow3 \
  graph1 graph2 graph3 \
  telemetry1 telemetry2 telemetry3 > candidate.tsv
```

It accepts only the current host's checked-in environment profile and fifteen
fully sealed canonical packet directories. It does not edit or admit anything.
The existing log parser projects each packet directly; no second parser,
placeholder baseline, copied unit label, or unchecked raw-log discovery path
exists. All fifteen raw-log hashes must be globally distinct. Every route's
three packets must be adjacent, chronological, and equal in executable
artifact identity; all fifteen packets must share source manifest, source
identity, revision, dirty state, generator, compiler path, and compiler
SHA-256. The three metric and unit sets must be identical and have the fixed
route cardinality. Semantic identities must match exactly. The projector
preserves packet 3's numeric spelling for `B`, computes `E` with
arbitrary-precision decimal arithmetic, and writes exactly the canonical
header plus the row count computed by the shared schema (`325` for the admitted
Darwin profile) to standard output.

After review, apply that complete output as the explicit `baseline.tsv` edit.
The cut validator is:

```text
tools/internal/measure/cut ROOT PROFILE \
  scheduler1 scheduler2 scheduler3 \
  compute1 compute2 compute3 \
  flow1 flow2 flow3 \
  graph1 graph2 graph3 \
  telemetry1 telemetry2 telemetry3
```

Arguments are the same retained packet directories. The validator reruns the
projector and requires its complete byte stream to equal the checked-in table;
it then loads that table through the complete canonical schema. Consequently
the projector and cut cannot drift on packet sealing, profile selection,
metric projection, row order, `B`, or `E`. A modified proof result, source
identity, compiler, artifact, packet field, unit, row order, or 64-hex-shaped
value without its retained packet fails the cut.

The retained calibration packets are required only while accepting a baseline
cut. Ordinary comparison, Release admission, and clean-checkout use consume
only the checked-in table and the current candidate log. Raw packets remain
review evidence, not a runtime dependency or fallback authority.

The focused edit-build-test loop is intentionally separate. Its correctness
and no-op ownership are structural harness contracts, while focused wall time
depends on the selected source closure and compiler-cache state. No synthetic
focused timing is inferred from the five installed-product routes. Measure a
specific focused command independently when changing that loop.

`tools/measure/build/run [build [target]]` owns that independent observation. It
uses one canonical compile-database command for the five public domain
umbrellas plus the opt-in Compute async, math, and Session entries and reports preprocessed bytes,
repository and total transitive headers, and the
median of five syntax compiles. It intersects Ninja's live target graph with
its depfile database before reporting materialized umbrella and selected
private-leaf fan-out, so only live graph outputs contribute to the count. A
dirty output may retain discovered dependencies that precede its current
compile; the Socket leaf therefore also reports the current compile-database
direct-edge ceiling. A target dry run reports production, test, and other dirty objects
separately; only a zero-dirty target makes its materialized depfile rows a warm
observation. Complete source manifests before and after the observation must be
byte-identical, and the output seals both that manifest and the exact
compiler-flag identity. This development observation is not a sixth
installed-Release baseline route.
