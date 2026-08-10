# Performance Method

This page and [`baseline.tsv`](./baseline.tsv) are the product performance
baseline authority. The five installed-Release measurement routes produce one
sealed workload observation each:

- `tools/measure/scheduler/run`
- `tools/measure/compute/run`
- `tools/measure/flow/run`
- `tools/measure/graph/services/run`
- `tools/measure/telemetry/run`

`tools/measure/admit/run` is the sole Release admission command. It records
three consecutive observations for every route, projects their canonical
per-metric medians, and compares that one candidate to the checked baseline.
`tools/internal/measure/compare` is the sole raw measurement-log parser.
`tools/internal/measure/project` is the sole median projection owner, and
`tools/internal/measure/admit` is the sole candidate comparison owner.
`tools/internal/measure/schema.pm` owns the baseline schema, host/profile
selection, route cardinality, canonical ordering, and immutable-packet seal.
The raw route boundary records its own comparison result for diagnosis; the
admission boundary is the only Release decision.

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
SHA-256 identities per route. The adjacent same-machine observations are
`x_1`, `x_2`, and `x_3`; every metric freezes the exact decimal spelling of

```text
B = median(x_1, x_2, x_3).
```

When equivalent decimal values occupy the median, the lowest packet ordinal
owns its spelling. Adjacency is structural: the three inputs must occupy
consecutive positions in that route's timestamp-named evidence-packet
directory. A baseline cut and a Release admission both use this exact rule;
neither can skip an intervening observation or select a quieter sample. The
median has a one-observation contamination bound: one arbitrary scheduler
interruption cannot move `B`, while two degraded observations remain visible
in `B` rather than being hidden by an allowance.

Each Compute Product row measures one public `Flow` compiled into one prepared
`Pipeline`. A cold row starts before authoring and ends after the first terminal
result read. The read-completion timestamp precedes the terminal Profile
projection, so evidence validation cannot inflate first-result latency. A warm
row records sixty consecutive `Pipeline::run()` samples
after one fixed sixty-run conditioning block and performs one terminal read
after the samples. The conditioning block is untimed and occurs once per warm
shape or bounded-count phase; it is never inserted between recorded samples.
No sample is retried or filtered. Bounded rows repeat the same prepared
capacity at the declared active counts; each phase observes its terminal once
while atomically publishing the next count and poisoned-tail input.

The implementation owner is `tools/measure/compute/suite/product.cpp`, with
single-purpose `product/model.hpp`, `oracle.hpp`, `report.hpp`, `exact.hpp`,
and `bounded.hpp` seams. The entry owns scenario order and CSV column order;
the seams own only their named model, oracle, reporting, exact-shape, and
bounded-shape work. Together these six owners form the complete installed
Product measurement implementation.

For a new Release source, `tools/measure/admit/run` creates an independent
three-packet set for each route. Every input must have a passed workload, the
same source/toolchain identity, the same executable identity within its route,
the same canonical metric set and units, and the same semantic identity. It
then evaluates the median candidate against the existing `B + A` limits. The
command records all fifteen packet paths, the candidate TSV SHA-256, and the
admission proof in one immutable `measure-admit` evidence packet. A raw route
whose one-observation comparison fails remains valid diagnostic input when its
workload and sealed semantics pass; the fixed median is the only admission
result. A failed median is retained as failed evidence and never widens `A` or
permits a replacement subset.

All three packets must expose the same canonical metric and unit set. The
projector stores the final admission allowance `A` in the `envelope` column.
The comparator has no hidden multiplier:

```text
L = B + A.
```

The projector derives `A` by cost class:

| Cost class | Admission allowance |
|---|---:|
| Scheduler bytes, Telemetry allocations | `0.10 B` |
| Flow frontend time in milliseconds | `0.25 B` |
| Scheduler, Compute, Flow graph construction, graph-service time | `2.5 B` |
| Telemetry cold time | `max(0.25 B, 100000 ns - B)` |
| Telemetry warm time | `max(0.25 B, 5000 ns - B)` |

Negative budget terms are clamped to zero. The count rule preserves the exact
zero-allocation gate. Telemetry uses explicit product budgets: at most 100
microseconds of cold instrumentation overhead and 5 microseconds after
preparation. No observed separation among the three calibration packets
changes `A`: a transient remains in the retained raw evidence, not in the
future release tolerance. Flow's millisecond frontend retains a 25-percent
regression effect size; its sub-microsecond graph-construction timers join the
short-span class.

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

The Darwin product schema has exactly 262 data rows: sixteen identities, six
environment facts, and route cardinalities 95 scheduler, 82 Compute, 5 Flow,
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
`--batch`, `--pipeline`, `--checkpoint`, `--pipeline-profile`, `--recurrence`,
`--window-repeat`, `--plan-memory`, `--prepare-memory`, and
`--virtual-residency` diagnostics are
intentionally outside the installed-Release baseline route. `--resident`
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

`tools/measure/compute/run --virtual-residency <cpu|metal|vulkan>` is a separate
current-source product diagnostic over the public opt-in
`<rund/compute/virtual.hpp>` surface. Each of three sequential process packets
opens an executable selected backend and measures one real fixed-slot workload with
`L = 65,537` logical `I32` elements, a 4,096-element page, and three physical
slot pairs, so `C = 12,288` elements and `L > C`. The tail makes 17 logical
pages and the fixed capacity makes six waves per run. Packet markers delimit
the three complete CSV streams; they are diagnostic packet boundaries, not
Release evidence packets and are never inputs to baseline admission.
The backing transfers cover 17 active pages, or 278,528 bytes per direction;
the fixed physical batch covers all three slots in every wave, including the
padded tail slot, or 294,912 bytes per direction. The CSV keeps those facts in
separate Profile-owned columns.

The cold interval starts at public graph authoring, then includes compile,
VirtualBacking and VirtualBuffer construction, VirtualPipeline preparation,
input-backing seed, one run, and one full terminal output-backing read. The
tool owns only these steady-clock wall boundaries. The machine-readable cold
row contains their author, compile, prepare, seed, run and read partition; it
does not require cold compilation to be zero. It validates every output
element and independently checks the content hash, but serializes the plan and
result identities from the terminal Profile's Stats authority.

VirtualPipeline seals preparation evidence before its first run starts a new
terminal execution epoch. The route therefore takes one public `Profile`
immediately after preparation and serializes its compile, cache, buffer,
descriptor and compile-time fields under `prepare_*`; the same snapshot binds
the memory owner for that epoch. It separately labels the terminal Profile
execution fields `terminal_*`. Both are direct snapshots; the benchmark
computes no counter delta and never treats a zeroed terminal epoch as proof
that cold preparation did no work.

`prepare_evidence_status` is a direct serialization of
`PipelineStats::preparation_evidence`, not a backend switch. Metal reports
`owner_local_compile_cache`: the physical status or nested-aggregate
preparation operation writes the same compile/cache/time fact into the
prepared owner's existing `AccelRunFacts` handoff. CPU reports
`unavailable_no_native_producer` because it has no native Pipeline compiler or
cache producer. Vulkan reports `unavailable_backend_global` until its
descriptor and pipeline producers have an owner-local preparation handoff.
Their structural zero fields are not interpreted as measured zero. The wall
columns `author_us`, `compile_us`, and `prepare_us` continue to measure the
public phase boundaries and are never converted into backend counters.
Consumers must gate every `prepare_*` numeric field on
`prepare_evidence_status=owner_local_compile_cache`; either unavailable status
makes those numeric placeholders inadmissible as compile/cache evidence. The
fixture enforces that unavailable rows contain only zero placeholders, so a
partial adapter-global snapshot cannot accidentally acquire authority through
the CSV.

The same prepared VirtualPipeline then opens one sample epoch and runs exactly
60 times. There is no Profile projection or caller observation between warm
runs. After the sample epoch closes, the route performs one full backing read
and one Profile projection. It reports nearest-rank warm p95, the ordinary
even-sample median p50, logical elements per second from p50, per-run waves,
compute/H2D/D2H submissions, backing and physical transfer bytes, `L/C`, plan
identity, graph/result hashes, raw compile/buffer/descriptor allocation
counters, and Profile MemoryStats. `ResidencyStats` must report 60 sampled,
allocation-free terminals, and the public per-run compile, buffer-allocation,
descriptor-pool creation, and descriptor-set allocation facts must remain
zero. The tool does not add a second heap or memory counter beside Profile.
Any failed run, missing terminal, counter mismatch, output mismatch, nonzero
runD-owned/Profile warm-allocation evidence, or hash mismatch fails the
packet; no retry or synthetic row is emitted.

CPU and Metal are executable measurement targets. On the Apple release host,
Vulkan exposes `VK_KHR_portability_subset`; public preparation returns exact
`BackendUnsupported` before backing callbacks or physical materialization, and
the diagnostic reports that target as blocked without emitting a timing row.
Native Vulkan retains the same diagnostic implementation, but its product
measurement remains unverified until a native device supplies evidence.

Every row is labeled `current_source_diagnostic`. It copies cache hit/eviction,
download-event and readback, kernel and backing-I/O time, plan peak/committed
and scratch bytes, and the Host, Device, Resident, Staging and Transfer
current/peak/cumulative counters directly from the terminal `Profile` and
frozen `PipelinePlan`. Page count, selected slot capacity, logical/page/working
set bytes and capacity wave count are the plan's values rather than fixture
constants. `terminal_reads=1` and `profile_projections=1` describe
the harness observation boundary; they are not backend counters. The warm
allocation columns are explicitly named `rund_allocation_free_runs` and
`rund_allocation_scope=rund_owned`. The route does not measure process-global
allocation; the backend product contract owns that assertion. Executable rows
record `process_global_allocation_status=not_measured`. A blocked portability
adapter has no row and therefore cannot be relabeled as performance evidence.
These rows remain outside the installed Release baseline and cannot enter
admission.

The installed-Release measurement executable includes only the installed
public SDK. Internal matrix-tile and transform-stage cost contracts are visible
only to the monorepo `RUND_COMPUTE_FOCUS` target that owns focused modes; the
installed executable does not accept focused modes. This keeps private Kernel
headers out of the SDK consumer boundary and makes a stale internal include a
compile failure instead of an undeclared package dependency.

`tools/measure/compute/run --plan-memory <backend>` fixes
`Max = 516096`, `Tile = 8192`, and `K = 63`, then compiles a Seed graph with
500 alternating Scan/Map pairs and freezes the issue-shaped sequence: an
`N = 64` nested window group, a later consumer of its published result, an
ordinary 64-iteration recurrence, an `N = 1` nested window group, and a final
publish/commit. The public reproducer has exactly 140 compact route templates
and 4,413 authored commands and exercises that complete frozen sequence. Its
only `prepare()` call uses a budget one byte
below `peak_bytes` and must fail with `PipelineMemoryBudget` without a device
allocation; the mode never materializes a Pipeline. Its CSV reports all
four disjoint preparation byte components, logical `peak_bytes`, exact CPU
`arena_extent_bytes`, page-rounded Device `committed_peak_bytes`,
logical/live/physical reports, structural counts, planning status, and both
process current RSS and process maximum RSS before/after. Current residency is
sampled at each boundary; maximum RSS remains diagnostic process high-water
evidence rather than an ownership attribution.

`tools/measure/compute/run --prepare-memory <backend>` is the sole focused
route allowed to materialize that same product-scale builder. It applies
`MemoryBudget{plan.peak_bytes}`, requires exact prepared-plan identity, and
reports Pipeline Host, Tile, Resident, Staging, and Device current/peak rows,
preparation wall time, process current/maximum RSS, and the complete public
failure location if the backend rejects. A successful row proves the product
owner stayed inside the frozen structural gates by comparing the consumed
logical host/native bytes, source-transient bound, descriptor/command/native
object counts, and semantic fingerprint with the frozen accelerator limit.
Separate backend Host/Device/Staging rows reconcile into the complete public
counters. A fixed, allocation-free `memory_snapshot()` walk also reports the
largest nonzero retained group as category/use/index/current bytes with the
explicit `pipeline` lifetime. Plan-only and failed rows instead say
`not_materialized`; their existing `largest_*` columns remain plan workspace
coordinates and are not mislabeled as observed retained owners. The acceptance
row also records the number and cumulative requested bytes of C++
`operator new` calls during the preparation interval. Those two fields expose
allocation topology and requested payload; they are diagnostic totals, not a
simultaneous high-water, and do not claim visibility into C/Objective-C or
driver-private allocation.
The acceptance gate sets a ceiling at preparation-start current RSS plus
`committed_peak_bytes`. The caller buffers and compiled Programs in
`persistent_bytes` already exist at that baseline and cannot be added again.
Preparation-end current RSS must fit that ceiling. A newly established process
maximum must also fit it; an unchanged earlier maximum is not attributed to
preparation. This is the sole RSS acceptance comparison. `peak_bytes` remains
the exact logical `MemoryBudget` authority, while `arena_extent_bytes` exposes
the unrounded CPU mapping span after internal alignment. Process RSS also includes allocator
metadata, runtime stacks, and backend-private storage that neither field may
relabel as exact runD payload; if the whole observation crosses the committed
envelope after every owned preparation contract passes, the row is
`process_rss_contract_failed`. It remains a failed product contract instead of
hiding the gap in a persistent-byte allowance.
This measurement prevents an earlier compiler/setup peak from hiding a later
multi-gigabyte preparation spike without relabeling opaque allocator or driver
bytes as an exact `PipelinePlan` component; physical Device and process RSS
rows remain separately named telemetry. An unavailable
backend is reported as unavailable, while any other rejection is admissible
only with the public Capacity code or a known stable native preparation
location.

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

The installed Compute route is the Product matrix. Every row names exactly one
of CPU, Metal, or Vulkan and one public chain:

- `map -> window -> filter -> reduce` at `N = 4096` and `262144`, with radii
  `4` and `1024`;
- `map -> pool -> filter -> reduce` with Keep/Clip windows
  `(N,K,S) = (4096,129,2)` and `(262144,2049,2)`;
- bounded `map -> rolling Min -> filter -> reduce` at capacity `M = 262144`,
  radius `1024`, and active counts `0`, `257`, `131072`, and `262144`.

The input, count, result, graph hash, output hash, and selected Range candidate
are validated against an independent CPU oracle. Inactive bounded elements use
alternating zero and `UINT32_MAX` poison. A stale capacity tail therefore cannot
pass by matching the active prefix accidentally.

For each cold row, `first_result_us` encloses public Flow authoring, Program
compilation, input upload, Pipeline preparation, one execution, and the final
typed read. Phase durations remain diagnostics. For each warm row,
`warm_p50_us` and `warm_p95_us` are computed from all sixty consecutive
prepared-Pipeline executions. Nearest-rank p95 is position fifty-seven because
`ceil(0.95 * 60) = 57`, so it is the fourth-highest observation rather than a
single scheduling maximum. A preceding sixty-run conditioning block separates
steady resident evidence from the cold first-result boundary without timing
or filtering that block. `active_elements_per_s` is derived from p50 and never
acts as a second timing authority.

After conditioning, the route calls `Pipeline::begin_samples()`, times sixty
`run()` calls without a Profile capture or read, and calls `end_samples()`.
The single terminal read is followed by the cohort's only Profile capture.
The existing `PipelineStats` owns two saturating `uint32_t` witnesses:

```text
R = sat32(sum accepted terminals)
C = sat32(sum clean accepted terminals)
```

A terminal contributes to `C` exactly when it succeeds and that sample epoch
has observed no Pipeline compile, Buffer allocation or reuse, descriptor-pool
or set allocation or reuse, upload, host write, download, cache lookup or
eviction, command-capacity rejection, shader or pipeline construction, or
transfer submission. A read or write while the sample epoch is active makes
the cohort non-clean. The row is admissible only for exact, non-saturated
`R = C = 60`; therefore a dirty middle execution cannot be hidden by the last
execution's Stats snapshot. CPU records zero
execution submissions and each GPU records one per run. Metal's terminal
unified-memory read needs no separate transfer submission, while Vulkan's
terminal read owns one. That physical difference is exact semantic evidence,
not a timing adjustment.

Dispatches, submissions, readbacks, logical and physical transfer bytes, peak
retained and resident bytes, logical and backing scratch bytes, cache evidence,
selected candidate/width/stage/shared-capacity, source and execution identity,
and graph/output hashes are exact fields. They distinguish multiple dispatches
inside one submission from multiple submissions and prevent a faster but
different graph, candidate, or observation boundary from entering the timing
comparison.

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
iterations, and compares three routes: 256 independently submitted Program
runs, one terminal-only `Pipeline::repeat<256>(..., write_final(...))` run,
and one lossless `Pipeline::repeat<256>(..., write_each(...))` run. Twelve
serial/terminal pairs and twelve terminal/history pairs independently alternate
order six times each after all three routes are warm. Typed readback occurs
outside the timed pairs.

The row is admissible only with identical nonzero serial and terminal content
hashes, exact iteration-major history contents, 256 serial submits and
dispatches, one terminal submit and dispatch, one history submit and dispatch,
one logical Pipeline step on each fused route, fully verified prefixes, and
zero warm compile, allocation, descriptor, upload, download, and round-trip
counters. The logical Pipeline hazard count remains 255 because it records the
authored recurrence proof. The element-local lowering removes those physical
inter-iteration barriers; it does not rewrite the public logical plan.

For `N` iterations, `E` elements, carried payload `S`, invariant payload `C`,
output payload `O`, and `W` device-capacity windows, the serial prepared command
shape has `N * W` Map dispatches and exposes
`Theta(N * E * (S + C + O))` payload loads and stores. The proved terminal-only
element-local recurrence has `W` dispatches and exposes
`Theta(E * (S + C + O))` payload loads and stores. The history route also has
`W` dispatches and exposes `Theta(E * (S + C + N * O))` payload traffic because
every authored output is mandatory. Both fused routes retain the same
`Theta(N * E)` arithmetic in the same per-element iteration order. Dispatch
count improves by the exact factor `N`; only the terminal route can remove the
intermediate output stores. Physical memory traffic and both reported wall
ratios remain measurements because compiler register allocation, spills,
arithmetic, occupancy, cache, driver, and submission costs are device facts.

`tools/measure/compute/run --window-repeat <metal|vulkan>` fixes
`Max = 516096`, `Tile = 1024`, `K = 504`, and `N = 64`. Its serial comparator
precomputes the same 504 tile seeds outside the timed region, then times
32,256 independently submitted one-Action Pipeline runs. The nested route
times one complete Seed/Action/Fold Pipeline execution. The repeated route
times 256 actual `run()` calls on a separate ordinary Pipeline inside one timed
interval. The sealed route uses the identical nested declaration with
`sealed_repetitions<256>()`. After all four routes are warm, the harness runs
three independent twelve-sample AB/BA comparisons: serial versus single
ordinary nested, single ordinary nested versus sealed, and 256 actual ordinary
runs versus sealed. Each comparison alternates which route runs first, so both
directed cross-route carryovers occur six times. Before recording each
comparison, one untimed AB followed by BA establishes the same phase-local
precondition and ends on the route that begins the first measured pair.
Boundaries between successive measured pairs are therefore same-route
carryovers, and a prior phase cannot become the immediate predecessor of the
next phase's first sample. Observation and the serial outer Fold occur after
timing.
The queue and resident count are already materialized inputs to both routes;
this workload measures the nested consumer, not GYEOL's active-queue producer
or a Compact-plus-consumer fusion.

The nested row is admissible only with 571 retained route templates, 33,264
authored Seed/Action/Fold occurrences, 504 executed outer windows, 32,256 executed inner
iterations, one nested submission, no failed coordinate, exact serial/nested
result parity, and zero warm compile, allocation, upload, download, and
fallback evidence. The common tile-transducer proof is
mandatory. Vulkan and a Metal stream that is ineligible for the narrower
complete-aggregate proof use the `K * 3` physical
Seed/transducer/Fold Program-occurrence shape. This exact Metal workload must
instead admit `WindowIndexedReduceSumU32` and report two physical dispatches,
one control command, one submission, `K + 1` workgroups, exact result/failure
parity, and no canonical occurrence-stream fallback. The matching contract
test must also prove that its two `K`-word partial ranges are non-overlapping
plan-owned Seed workspace and that no native aggregate scratch Buffer is
allocated or double-counted. In either shape the authored count does not
expand the native stream. The `compute.window` contract proves warm retained
identity by comparing the complete frozen owner/View snapshot across
executions. Program-internal normalization traffic
remains visible through the round-trip counters. The measured dispatch count
is reported independently because Seed, Action, and Fold Programs may each
lower to more than one native dispatch. The wall ratio therefore measures
submission, control, and backend command-path effects for this declared
workload; it is not an algebraic claim about every Action body.

Metal rows on Apple identify the calibrated size-class ICB path. Vulkan rows
whose environment reports MoltenVK prove the Vulkan API, SPIR-V, descriptor,
push-constant, command-buffer, and barrier path over that translation layer;
they are not native Vulkan throughput evidence.
The Metal wall time includes the hard-cut executor's remaining host envelope:
one outer command-buffer/encoder lifecycle, one bulk resource-residency call,
`C = ceil(D / 65,536)` retained ICB range calls, commit/completion, and fixed
control observation. It contains no runD command/binding/indirect-grid/state
traversal; its only host stream walk is the `C` compact 16-byte chunk records.
The preparation-memory row must report the frozen/consumed chunk counts and the
sum of device-calibrated `allocatedSize` bytes, not a coefficient or guessed
driver allocation. The canonical stream still includes the device cost of
frozen commands whose inactive payload threads return through uniform guards.
The exact aggregate stream instead contains one `K`-threadgroup tile dispatch,
one ICB Buffer barrier, and one ordered finalize dispatch in a two-command
size-class chunk; it has no inactive guard commands but still pays
fixed-capacity tile work and the same host envelope with `C = 1`. One nested
submit and the unchanged frozen owner/View snapshot must therefore be
interpreted with both the selected structural path and the measured wall
result, not as literally zero host or inactive-device cost.

The explicit native chunk-boundary contract is
`tools/measure/compute/run --metal-icb-boundary`. It is Apple-only,
`EXCLUDE_FROM_ALL`, and absent from CTest, so default verification retains only
the allocation-free size-class cases. The probe must encode and execute 65,537
real ICB dispatches as capacities `65,536 + 1`, place a fixed-value write at
the last full-chunk command and its read at the first tail command, call the
same chunk-loop helper as production, and complete through exactly one outer
command-buffer submit. Its CSV row is admissible only when `chunks=2`,
`boundary=65536`, `boundary_barriers=1`, `command_submits=1`, and `result`
equals unsigned `0x13579bdf`; `full_bytes` and `tail_bytes` are the device's
actual `allocatedSize` values, while `elapsed_us` is diagnostic rather than a
baseline speed claim.

### Input-Sealed Repetition Throughput

The sealed row is admissible only when the single ordinary, actually repeated
ordinary, and sealed Pipelines publish the same result bits, their
`PipelinePlan` values are equal, one ordinary execution and the sealed execution
use the same physical submit and dispatch counts, and the sealed Pipeline
reports `sealed_repetition_count = R` with
`coalesced_repetition_count = R - 1`. The repeated interval must complete
exactly `R` successful ordinary `run()` calls and advance that Pipeline's
generation by exactly `R`; the immutable per-run stats shape then makes its
reported submit and dispatch totals exactly `R` times the single-run counts.
Diagnostic stats reads remain outside the timed interval. All four routes must
report zero warm compilation, allocation, upload, download, and fallback
evidence, while the frozen owner/View snapshot must remain identical. The
untimed pre/post validation sweeps inspect all `R`
ordinary repeated-run stats individually; timed intervals inspect the final
immutable per-run shape after the clock stops. Each ratio uses only the medians
from its own AB/BA pair. The benchmark's caller-owned inputs are frozen for the
complete sample; the product contract independently rejects transactional state
and any exact external write-to-next-read overlap.

Let `T_single` be the ordinary nested median, `T_R_ordinary` the median of the
timed interval containing `R` actual ordinary runs, and `T_sealed` the sealed
median. The report derives:

```text
sealed_equivalent_time        = T_sealed / R
measured_throughput_speedup    = T_R_ordinary / T_sealed
single_execution_ratio        = T_single / T_sealed
```

`measured_throughput_speedup` is an actual paired wall-time ratio; it never
substitutes `R * T_single` for the measured repeated interval.
`single_execution_ratio` is the latency comparison, and it must be reported
beside the throughput figure. Neither value applies to changing active queues,
state feedback, per-tick checkpoints, host intervention, or intermediate
publication; those workloads are ineligible rather than silently coalesced. A
ratio derived from a historical row or another executable may not be spliced
into the paired sealed result.

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

Every lifecycle has two untimed warm-up rounds and sixty measured blocks. The
four operations use this Williams order, repeated fifteen times:

```text
Live     Record   Scenario Replay
Record   Replay   Live     Scenario
Replay   Scenario Record   Live
Scenario Live     Replay   Record
```

Thus every operation occupies each ordinal position fifteen times. Within each
operation block, the sixty level triplets repeat all six permutations of
Disabled, Basic, and Detail ten times. Every level occupies each ordinal
position twenty times; Basic precedes Detail in thirty pairs and follows it in
thirty. Pair IDs are retained through samples and deltas; summaries stay scoped
to one operation and lifecycle. No aggregate across those groups may hide a
slow path.

Every sample records steady-clock wall nanoseconds, `RUSAGE_SELF` user-plus-
system CPU time converted exactly from its microsecond fields to nanoseconds,
process-wide C++ allocation calls across all threads, and
`StorageReport::copied_bytes`. Detail also publishes its non-overlapping
prepare, work, and finish nanoseconds; Disabled and Basic must publish zero for
all three. For sixty sorted values, the median is the exact arithmetic mean of
positions thirty and thirty-one and nearest-rank p95 is position fifty-seven
because `ceil(0.95 * 60) = 57`. The p95 is therefore the fourth-highest
observation, so one isolated high interruption cannot itself be emitted as
the p95 order statistic. Half-integral medians use a `.5` suffix; integer
arithmetic prevents binary floating-point rounding.

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

A raw measurement packet contains the unmodified route log and a separate
`baseline.log`. An admission packet contains the projected `candidate.tsv` and
`admission.log`, which names all fifteen sealed raw inputs in canonical order.
The admission runner executes Product Compute first, the light Flow,
graph-service, and Telemetry routes next, and Scheduler last. This keeps GPU
cold/warm evidence independent of Scheduler's sustained CPU and memory load;
projection and publication still use the schema's canonical route order.
`run.tsv` seals the route, current host, both payload names, and both SHA-256
digests. It also seals the workload result independently as
`workload:status` and `workload:exit`; `passed` is valid only with exit `0`,
and `failed` only with a canonical process exit in `[1, 255]`. The selected
profile, compared metric count, and comparison result live under the
`proof:*` hierarchy. A raw comparator failure can therefore never rewrite a
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

Evidence status accepts a raw diagnostic packet only when all fields are
unique, the host is current, both files and hashes match, and a fresh parser
comparison over its retained log is byte-identical to its result. It accepts an
admission packet only when the canonical projection and checked-baseline
admission replay byte-identically from every named raw packet. A copied
packet, deleted log, edited result, stale profile, changed baseline, or
reordered raw input therefore cannot report `passed`.

Each raw route and aggregate admission opens an atomic attempt marker after
acquiring its build locks and before preparation. `running` reports
`in-progress`. Preparation and pre-workload frontend failures publish no
packet and leave the marker as the sole `failed` result. A standalone raw
route retains a failed marker when its one-packet diagnostic comparison fails.
During the fixed fifteen-packet admission, a raw comparison failure remains in
its immutable raw packet but is not a Release decision; the aggregate marker
stays live until projection and admission finish. Any aggregate workload,
projection, admission, source-stability, or publication failure leaves that
aggregate marker failed. A completed aggregate packet is visible only after
its own marker is atomically cleared, so an interrupted process cannot publish
a successful admission for an unfinished packet set.

## Update Contract

There is no update flag, placeholder table, discovery fallback, or implicit
host admission. Changing a workload, exact semantic projection, baseline
value, calibration envelope, or environment profile requires an explicit
reviewed edit to `baseline.tsv`. A baseline cut must name its source manifest,
retain the raw measurement packets, and pass the positive and negative
`tools.measure` contract. Deleting a metric is also an explicit cut:
unbaselined output and missing baseline rows both fail.

A cut uses three sequential executions of every route at one stable product
point `M0`. Packets supply `x_1`, `x_2`, and `x_3`; their per-metric median
supplies `B`. All fifteen packets must seal `workload:status=passed` and
`workload:exit=0`. Their comparator proof may be `failed`: before the table is
cut, an M0 observation is evaluated as calibration input rather than Release
admission. This exception applies only to calibration admission;
ordinary release evidence still requires a passing comparator proof.
After the explicit table edit, the product manifest becomes `M1`. The table's
`manifest` identity remains `M0`, because it identifies the measured product
point; rewriting it to include its own edit would be a circular hash demand.
Release is then rebuilt at `M1`, and `tools/measure/admit/run` records a new
independent three-packet set for every route against the new table. Its one
aggregate packet is the ordinary same-`M1` admission evidence.

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
selects the median observation with arbitrary-precision decimal arithmetic,
preserves its exact numeric spelling for `B`, and writes exactly the canonical
  header plus the row count computed by the shared schema (`262` for the admitted
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
