# Virtual Performance Method

This page owns `--virtual-residency`, `--virtual-crossover`, and
`--virtual-route-matrix`. These are
current-source diagnostics outside the installed Release baseline. Generic
manifest, semantic admission, comparison, and publication rules remain owned
by [Performance Method](../method.md).

The crossover producer is compiled by responsibility under
`tools/measure/compute/virtual/crossover/`: `prepare.cpp` owns the paired
Pipeline/backing construction and deterministic seed, `evidence.cpp` owns
output/hash, cold-terminal, Profile, and warm-sample proof validation,
`report.cpp` owns the frozen CSV row and field order, and `run.cpp` owns the
ordered ABBA coordinator and public measurement loop. `internal.hpp` contains
only the borrowed measurement records and declarations; no leaf owns a second
sample, Pipeline, Profile, or CSV schema authority.

The crossover aggregator follows the same ownership boundary under
`tools/measure/compute/virtual/crossover/aggregate/`: `parse.cpp` owns packet
and key validation, `report.cpp` owns the frozen column/consensus/bracket/slice
projections, and the parent `aggregate.cpp` owns only stdin packet sequencing
and failure classification. No parser or report helper is reimplemented in
the coordinator.

The natural route-matrix producer is likewise compiled by responsibility under
`tools/measure/compute/virtual/route_matrix/`: `run/prepare.cpp` owns bounded
case geometry, overflow-checked residency configuration, program construction,
backing seed/tail initialization, and Pipeline admission; `run/sampling.cpp`
owns timed Pipeline execution, output reads outside the timed wall, the fixed
CPU/backend/CPU cohort, and summary statistics; `run/evidence.cpp`
owns paired cold/conditioning checks, terminal/output/hash/Profile evidence,
route capture, and final row evidence publication. The substantive
`run.cpp` coordinator owns device ordering, case enumeration, failure
classification, and CLI parsing. The route oracle is physically divided under
`oracle/`: the parent owns deterministic expected values, `cohort.cpp` owns
DeviceVsm and WindowRing cohort proofs, `backing.cpp` owns memory/backing
receipts, `mismatch.cpp` owns ordered failure classification, `route.cpp` owns
route/measurement labels, and `support.cpp` owns checked arithmetic. Its
`local.hpp` is declarations-only, so these leaves share no mutable oracle.
`report/` remains the sole CSV schema and packet-consensus owner, and
`route_matrix/internal.hpp` contains only shared records and declarations for
these phases.

## Natural route matrix

`tools/measure/compute/run --virtual-route-matrix <metal|vulkan>
[--profile core|full]` is a standalone, current-source measurement executable.
It observes natural public admission and never forces a route or bypasses a
fallback. Core contains unary Pointwise Q=9, the explicit
`pointwise_staged` unsupported/not-comparable cell when no current natural
distinction exists, resident Pointwise Q=9, and spatial Window Q=9/W=2.
Full uses Pointwise Q=3/9/257/4096 and Window W=2 for those Q values plus W=4
at Q=9/257. Every row uses page elements 256 and Pointwise frame/capacity 2;
Window W remains a separate parameter. Q4096 remains bounded at 1,048,576
elements and emits failure rather than shrinking or silently selecting another
workload.

The wrapper builds only the EXCLUDE_FROM_ALL matrix target, holds the
accelerator lock, and runs three independent diagnostic packets. Each packet
does one fresh-owner cold run, one untimed conditioning run, then 30
`CPU,backend,backend,CPU` cycles (60 synchronous warm samples per side).
Its stdout is reserved for the 134-field CSV stream; configure/build and lock
diagnostics stay on stderr without changing failure status.
The aggregate retains raw rows and accepts a row only when packet schema,
exact core/full cardinality (4/18), unique join keys, order,
workload/backing/shape/hash identity, route proof identity/flags, owner and
counter evidence, one Final/publication, and all 60 allocation-free samples
agree. The join key includes profile, evidence scope, backend/API/driver,
case/backing/geometry, semantic/plan/proof identities, and observed route. A
natural route or proof mismatch is retained as `not_comparable`; it is never
relabeled as a passing fallback. Failure and unavailable rows remain raw
evidence. Only admitted rows expose timing, and a published timing field is
the median of the three packet values rather than packet 1. Every one of the
60 backend samples is status- and output-checked after its timed
`Pipeline::run()` wall; verification is outside that wall, and the first
failed sample invalidates the row. The untimed conditioning run is validated
with the same status/output/hash predicate before the warm cohort. Raw route
evidence carries the first proof identity, proof-valid count,
identity/flag-stability and capability-consistency fields, plus exact
prepared/executed/Final/publication counts; packet semantic agreement includes
these fields and the explicit `timing_authority`.
The observer is compiled by responsibility: `observer/evidence.cpp` owns the
single route-evidence accumulator and lifecycle/hash/capability checks,
`observer/hooks.cpp` owns the two delegating DeviceOps hooks, and
`observer/lifecycle.cpp` owns installation, reset, sealing, and restoration;
`observer/local.hpp` contains only their shared state and declarations.
The observer obtains route kind and endpoint from the selected typed proof's
queries. It preserves the existing CSV flag encoding without reconstructing
parallel route fields or owning a second physical topology.
The CSV also carries lifecycle observations/mismatches and per-run authority,
pipeline-terminal, and capability-guarantee counts; a DeviceVsm cohort is
accepted only with 62 observations, zero mismatches, and every required
guarantee count equal to 62.

Metal is the native primary. Vulkan rows are marked
`MoltenVK_portability_only` and are not native-Vulkan rankings. The packet
records actual owner/route/proof IDs and physical command submissions,
logical batches, queue calls, dispatches, transfers, page/backing traffic,
memory, Final/publication, and host callback counters. GPU/kernel and submit
spans have independent scope/status fields; when no exact producer exists they
are `unavailable_no_exact_producer` and numeric zero, never inferred from wall
time. Observer overhead is included in wall time. Adjacent-Q labels are
published only within one family/backing/Window key after three matching packet
labels; cross-family and cross-backing speedups are forbidden.

The separate Direct pointwise Q2--Q4 diagnostic is owned by
[Window](./window.md). It does not change the rolling Graph crossover surface
on this page.

Natural admission for an ordinary all-staged unary nonresident Pointwise run
with Q>=2 first selects the StagedLoop when the exact mapped Host-visible /
coherent input and output views are available. A pre-lease capability miss
cleanly declines to the existing fallback; no fallback is permitted after
owner mutation or native acceptance. The explicit DeviceVsm/resident/required
packet below remains a separate diagnostic and must not be used to relabel a
different route.

## Residency Packet

`tools/measure/compute/run --virtual-residency <cpu|metal|vulkan>` runs the
explicit DeviceVsm callback-backing diagnostic packet. The separate
`tools/measure/compute/run --virtual-residency-resident
<cpu|metal|vulkan>` packet keeps the same graph, shape, sampling, and receipt
rules but allocates both logical backings through the public
`resident_virtual_backing<T>` API. Each command runs three sequential process
packets over the public VirtualPipeline surface. Packet markers delimit
complete CSV streams; they are not Release evidence packets.
Each packet opens the selected Device exactly once and retains that owner
through environment projection and the full cold/warm cohort. Accelerator
packets open the selected native Device before opening the CPU metadata owner,
so environment reporting cannot consume, replace, or precede the measured
native owner. If native open fails, a failure-only catalog probe records the
exact backend reason; it never supplies a fallback Device or a measurement
row.

The fixed product workload uses `L=65,537` I32 elements, 4,096-element pages,
and per-bank `K=3`. It therefore has 17 pages, six epochs, and six total
input/output frame pairs across two banks. Logical backing bytes are 262,148
per direction. Physical active-frame traffic is 278,528 bytes per direction
on a path that actually transfers output; inactive terminal frames are never
counted.

The focused executable installs a measurement-only observer over the exact
production `DeviceOps` DeviceVsm slots for an explicit route. It copies the live operation table,
delegates preparation and execution to the original non-null slots, observes
the retained aggregate Final after the delegated call returns, and restores
the original table unchanged. The execution observer forwards the complete
ordered input-backing span unchanged; it does not collapse a multi-input run
to the first input. It supplies no alternate lowering, result, capability, or
fallback. The older service-aware Sliding slots are not wrapped and cannot
satisfy this packet. CPU has no DeviceVsm slots and is reported as
`cpu_reference`. Empty and one-page cells are reported separately; a Q>=2
accelerator row is accepted only as
`gpu_driven_device_vsm_whole_run_staging_product` when every measured run
produces the complete aggregate receipt and explicitly reports the current
whole-run callback-backing staging boundary.

For a Q>=2 accelerator row, the resident packet additionally requires the
status `gpu_driven_device_vsm_resident_product`, exact resident input and
output binding, zero whole-run staging, zero uploaded/downloaded bytes, and
zero `ResidencyStats::backing_{read,write}_bytes`. The DeviceVsm aggregate
receipt must still report exact logical GPU backing bytes in each direction;
that producer describes device access, not Host callback traffic. Seed and
terminal observation use the same public resident backing outside the sampled
run wall. A callback-backing row cannot satisfy the resident packet by merely
reporting one submit.

The resident packet also authenticates the output-hash hard cut. The cold run
must perform exactly one stable-Host-view hash observation and no reuse. After
the untimed conditioning run, all 60 warm samples must keep the observation
counter unchanged while advancing the committed hash-reuse counter exactly
once per sample. The public terminal read still recomputes the expected content
hash outside the sampled wall. This proves that a timed warm row did not hide
an O(N) Host output scan behind a DeviceVsm label.

The observer's post-delegate Final lock and value copy occur before the public
run returns and are therefore included in the accelerator run wall. They
allocate nothing in the sample epoch, and the harness neither estimates nor
subtracts their cost.

Here Q is the number of active page coordinates, not the K-frame capacity
epoch count. For one accepted Q-coordinate run the receipt requires one
successful preparation and execution; Q generated, Forecasted, Promoted,
completed, Drained, and Persisted pages; exact logical GPU backing bytes; one
native submit and payload dispatch; zero epoch-native submits, Host service
turns, and epoch callbacks; one aggregate Final; one Authority accept; two
Pipeline terminals; and one backing publication. The warm row sums those
exact facts over all 60 samples. The public Profile must independently report
one handoff, one native batch, one queue call, one compute submission, one
dispatch, and no H2D/D2H submission. A missing receipt is
`device_vsm_not_observed` and fails the row rather than being renamed as
product execution.

The route separately authenticates three backing facts. Backend
`gpu_addressable_backing` means the immutable native proof binds GPU buffers;
`public_gpu_addressable_backing` means those exact buffers are the public
Virtual backings rather than private staging; and `whole_run_staging` records
the current callback-backing pre/post copy path. `bounded_page_io` describes
resident-to-fixed-W-ring GPU page access. It is not a bounded external
Forecast/Drain service. The distinct `bounded_external_page_service` counter
must remain zero until a running-GPU/Host or device-storage authority is
implemented and proved. The route also authenticates device-generated
recurrence, fixed native and common storage, one native submit, zero Host
service turns and epoch callbacks, and one aggregate terminal once per
accepted run. Whole-run preencoding has no producer in this observer and is not
used as proof. Submit and callback counters alone can therefore never relabel
the older O(Q) authored intermediate as DeviceVsm.

Preparation lifetime is part of the receipt. The cold row must observe one
cold owner and no rearm. After the conditioning run, every sampled Q>=2 row
must observe the same cached owner through a warm rearm, no cold owner, and a
monotonic rearm count reaching at least the sample count. The rearm clears only
the aggregate backend result cell; Pipeline snapshots and Authority
credentials remain per-run. This prevents a repeatedly allocated one-submit
path from being reported as a warm product route.

The cold interval includes public graph authoring, compile, backing and view
construction, VirtualPipeline preparation, seed, one run, and one terminal
backing observation. Phase walls remain separately labeled. One Profile after
preparation owns `prepare_*` facts; the terminal Profile owns execution. The
harness computes no counter delta and never interprets an unavailable native
producer as zero.

After one conditioning run, the same owner opens one sample epoch and runs 60
times without Profile or output observation. It then performs one terminal
backing read and one Profile projection. Acceptance requires all runs,
output/hash/identity, exact traffic and counters, 60/60 allocation-free
terminals, and fixed memory. No retry, sample filter, or synthetic row is
allowed.

Directional overlap is admitted only from actual H2D/D2H wall intervals
intersected with an exact native-submission-in-flight receipt. Their saturating
sum must equal `overlap_ns`. Backing duration is never relabeled as transfer.
A coherent output view performs no D2H and must report zero downloaded bytes,
readback time, and D2H overlap while preserving backing bytes and output hash.
This receipt is not an exact GPU-kernel timestamp.

The phase columns retain separate producers but are not assumed additive.
`submit_wait_ns` is the native driver submit span; `route_native_submits` is
the DeviceVsm Final's exact submit count; `route_completed_ns` is its native
aggregate completion duration; `route_host_service_turns` must be zero;
`backing_io_ns` is measured whole-run callback I/O duration; and
`controller_stall_ns` is the public submit-to-product-completion envelope,
which may contain native and backing spans. Consequently subtracting or
summing these columns without interval evidence is forbidden.
`fusions` and `fusion_rejections` are the public graph counters. Logical,
physical-transfer, page, and backing bytes retain their existing owners. The
current product Final has no exact ready-edge stall-duration producer, so the
CSV explicitly emits
`ready_edge_stall_status=unavailable_no_exact_producer` and
`ready_edge_stall_ns=0`. The zero is unavailable data, not a measured zero and
cannot support a ready-edge latency or GPU-idle claim.

CPU, Metal, and Vulkan are executable residency-packet targets. Vulkan uses
the retained selected-local Pipeline owner; a full prerecorded command buffer
is not admitted. On Apple, the Vulkan packet is MoltenVK portability evidence,
not native Vulkan-driver throughput or sparse-device evidence. MoltenVK may
allocate opaque queue work, so warm acceptance uses runD's exact sample and
fixed-memory producers rather than claiming process-wide driver allocation
freedom.

## Crossover Surface

`tools/measure/compute/run --virtual-crossover` creates one paired CPU/Metal
owner in the same Apple-only packet. Three complete process packets are
required. Missing cells, semantic mismatch, unavailable native Metal, failed
output, or warm-allocation failure invalidate the surface.

The grid is frozen as:

```text
G = 4096 logical payload elements per page
K = 3 Device frames per bank
H = 12 accelerator Host input frames
N in {G-1, 3G-1, 4G-1, 16G-1, 64G-1}
r in {1, 8, 32, 128}
active ratio in {1/16, 1/4, 1/2, 1}
n = floor(N * numerator / denominator)
F = G + 2r authored physical frame elements
```

CPU and Metal share `N,n`, seed, Clamp Window Sum graph, poisoned tail, G, and
K. Graph identity, logical/page geometry, frame capacity, output bytes,
contents, and output hash must match. Backend plan identity may differ.

Radius is a semantic arithmetic-intensity proxy. One output has `2r+1`
logical terms, so the exact declared ratio is:

```text
semantic_additions = 2rn
semantic_additions_per_logical_io_byte = 2rn / (2*n*sizeof(I32)) = r/4
```

This does not infer hardware instructions, physical halo traffic, cache-line
reuse, or FLOP/byte.

Each cell creates fresh paired owners, records one first run, and performs 60
untimed conditioning runs. The warm cohort uses thirty
`CPU,Metal,Metal,CPU` cycles, exactly 60 unobserved runs per backend. One
terminal backing read and Profile follow. The row publishes p25, even-sample
p50, p75, nearest-rank p95, and MAD. Retry, filtering, mid-cohort observation,
and replacement owner are forbidden.

The terminal Profile is the only VSM counter authority. Execution-tier hit
ratio is:

```text
cache_hit_count / (cache_hit_count + page_in_count)
```

CPU executes in Host frames, so this is not called a Device hit. Metal alone
may derive Host-supply hits as `page_in-(late+prefetch)` when the operands are
valid. Software VSM hits are not hardware-cache hits.

A packet label is `metal_observed` only when Metal p75 is below CPU p25,
`cpu_observed` under the symmetric condition, otherwise `indeterminate`.
Publication requires the same label in all three packets. A bracket requires
opposite published winners at adjacent measured N for identical radius and
active ratio. Indeterminate neighbors are uncertainty, not brackets. No
interpolation, extrapolation, or monotonicity assumption is allowed.

Rows serialize exact command submits, submit wait, readback, native kernel
timestamp when available, uploaded/downloaded bytes, directional overlap,
stall, cache/backing counters, and the three public bounded-window receipt
coordinates. Clamp Window Sum is structurally eligible for the recurrent
owner, but this crossover surface authors a graph-shaped workload outside the
ordinary unary default; its non-admitted path therefore remains the rolling
fallback. Both backends must publish zero handoffs, batches, and queue calls in
every crossover row; the packet aggregator rejects a nonzero receipt. A
separate same-workload diagnostic forced the recurrent path to measure its
byte/terminal tradeoff; it is recorded in [Results](./result.md), not admitted
into this frozen surface. A zero kernel timestamp means unavailable; it is
never reconstructed from a host receipt.

## GraphResident Measurement Surface

The runner is physically divided under `virtual/graph_residency/run/`:
`fixture.cpp` owns backing construction, seeding, output oracle, and pipeline
preparation; `evidence.cpp` owns proof, alias, and native-fact projection;
`timing.cpp` owns monotonic phase checks; and the parent `run.cpp` owns only the
cold, conditioning, and warm sequence plus final publication. `local.hpp`
declares one shared case view and helpers without duplicating pipeline or
backing state.

The bounded GraphResident measurement is invoked with
`tools/measure/compute/run --virtual-graph-residency <cpu|metal|vulkan>`. Its
private workload specification is shared with the GraphResident contract and
freezes Q=5, C=2, 73 elements (five pages and a seven-element tail), 128
literal leaves, and the five-stage branch/fan-in topology
`A→X, B→Y, X,Y→Z, C→W, Z,W→O`. The workload digest, topology digest, seeds,
expected output, and FNV hash are emitted with every row; they are identity
and correctness evidence, not performance claims. The semantic product
fixture covers this exact shape with both sealed unsigned roots, U32 and U64;
the comparable timing workload remains U64.

The CPU row uses ordinary Host/Memory backings. Metal and Vulkan rows require
the actual GraphResident owner and proof: three internal banked owners, four
external endpoint classes, the authenticated wavefront and alias/liveness
facts, fresh generation/nonce continuity, and the native lifecycle counters.
The accelerator row is invalid if it falls through GraphPointwise, stages
through Host, uses an epoch service, or reports a quarantined terminal.

For each typed product run, the public branch order and non-page-multiple tail
are checked after backend compilation. One cold run and three warm rearms must
retain the same proof digest, sealed scalar/domain/element width, and physical
bindings. Every Metal/Vulkan run must produce one native submit, dispatch,
Final, Authority accept, and backing publication, with one terminal per graph
stage, zero epoch submits, Host callbacks/service, and backing I/O. The
backend execution must compile and execute the generated typed shader; source
text inspection alone is not evidence. This fixture is semantic evidence only:
reduction remains U64-only, and no dynamic paging or nonresident GPU page
service is claimed.

Cold wall time encloses authoring, compilation, preparation, seeding, the first
run, and one terminal output read. After one untimed conditioning run, the
harness retains 60 consecutive synchronous run samples without Profile, Stats,
backing, or output observation. It then performs one terminal read/profile.
All samples are retained: the report uses nearest-rank p25, p50, p75, p95,
and median absolute deviation in steady-clock microseconds. There are no
retries, filters, or outlier removal. CSV output has separate `semantic` and
`timing` rows and includes device/driver identity plus the exact native,
wavefront, Host, backing, output, and version fields. A timing row is a
measurement of this harness boundary and is not a throughput or speedup claim.
Accelerator warm samples also carry diagnostic-only, non-additive phase
durations: `pre` is run-enter to native submit, `queue` is submit to callback,
which ends at the native completion callback entry; `finish` is classification
and DeviceVSM product Final, and `post` is product Final to public run-return.
A phase sample is valid only when the run succeeds with exactly one
submit, callback, and Final and the nonzero steady-clock marks satisfy
`enter <= submit <= callback <= final <= return`; absolute marks are never
serialized. Phase percentiles and MAD are available only for 60/60 valid warm
samples; otherwise their valid count reports the incomplete phase and the
phase is unavailable. CPU has no phase sample, while its existing wall timing
and validation remain unchanged. The timing row appends p25, p50, p75, p95,
MAD, and valid-count columns for each of `pre`, `queue`, `finish`, and `post`.
The submit, callback, and Final clock probes run inside the measured synchronous
wall, are not subtracted, and provide attribution only; the 100x wall gate
therefore applies to that instrumented wall.
The timed loop intentionally has no per-sample scalar output receipt or Profile
observation: the 60 samples prove successful publication/version progression
and the exact terminal hash, not a broader 100x or per-sample output claim.

Cross-process GraphResident acceptance uses the CPU reference baseline
`B_cpu = median(p50_1, p50_2, p50_3)` and the strict target threshold
`B_cpu / 100`. The endpoint count is read from the sealed product proof as
`proof.residents.count` and must equal the declared input count plus one; it
is not inferred from resource or owner counts. Route evidence requires
`pipeline_terminal_count == stage_count * runs`, while aggregate Final and
publication each equal `runs`. A same-source, same-environment cohort is
accepted only once; reruns require a semantic, backend, or integration change.
The batch value `T_batch / K` is a throughput diagnostic and must not be used
as a single-run latency claim.

GraphResident warm reuse covers only the private prepared dispatch recording
and its immutable descriptor/buffer bindings. Vulkan cold preparation owns one
secondary recording and its descriptor lease; each run records a fresh primary,
executes that secondary, inserts the compute-to-Host barrier, submits, waits,
and performs its own Final/publication. Metal cold preparation owns one
one-dispatch ICB; each run creates and submits a one-shot command buffer that
executes that ICB and waits before the same terminal path. This is not full
command-buffer or submit/Final caching: a submitted, in-flight, device-loss, or
`UnknownMayWrite` owner is never reused.

The wrapper uses the repository's existing three-independent-process packet
protocol. Each process emits one raw CSV decision stream delimited by the
`diagnostic_packet` markers; the measurement leaf does not aggregate or invent
a second packet authority. Cross-backend conclusions, if later needed, are
computed from those raw packet rows by the owning integration surface. Raw
rows are explicitly `unsealed_current_source` and carry the packet ordinal and
selected device/driver identity only. `tools/measure/packets` supplies the
three process ordinals and packet markers; it does not seal a manifest, HEAD,
dirty-tree, source, or binary record. The raw CSV does not claim those facts;
root-level evidence capture owns any later local provenance record.

## GraphPointwise Measurement Surface

`tools/measure/compute/run --virtual-graph-pointwise <cpu|metal|vulkan>` is the
current-source GraphPointwise packet. It builds the public three-input U64
graph, authors the public PageMap overload, seeds the three public backings,
and records one cold run, one untimed conditioning run, then 60 consecutive
warm samples followed by one terminal read and Profile. The cold wall includes
authoring, compilation, preparation, seeding, the first run, and its terminal
read; warm samples contain only the synchronous public run call.

The fixed semantic shape is 73 logical elements in five 16-element pages with
a seven-element tail, three external inputs, three graph stages, two-frame
epochs, and three batches. Input 0 is an identity Begin map; input 1 is a
reverse End map within each two-frame batch; input 2 remains ordinal. The
fingerprint, input digest, expected output, output hash, versions, proof
identity, PageMap digest, stage/input/page/tail geometry, and owner/control
identity are checked independently by the packet oracle.

CPU is accepted only as the same public graph reference with no DeviceVSM
owner. Metal and Vulkan are accepted only when the retained GraphPointwise
owner and proof produce one native submit, one payload dispatch, one aggregate
Final/publication per run, zero epoch submits and Host service/callbacks,
authenticated PageMap/proof geometry, stable warm owner/rearm identity, and
allocation-free samples. A GPU fallback or ordinary Host/GraphPointwise route
is reported as a failed packet rather than relabeled as GPU evidence.

The leaf emits only `graph_pointwise,semantic` and `graph_pointwise,timing`
CSV rows. The latter carries steady-clock p25, p50, p75, p95, and MAD over all
60 samples; both rows carry packet ordinal, selected device/driver identity,
graph/proof/owner identities, lifecycle counters, residency bytes, hashes, and
publication/version fields. The wrapper retains the existing packet protocol
and accelerator lock; this surface does not alter `result.md` or baseline
performance claims.

Failure or incomplete rows remain serialized with the actual compute status
code, reason, and error from the observed Status. The timing row always keeps
the cold wall and reports the actual number of completed warm samples; p25,
p50, p75, p95, and MAD are blank unless `end_samples` and timing summarization
both completed successfully. Such a row is diagnostic failure evidence, not a
performance result.

The measurement implementation keeps the public cold/conditioning/warm run
sequence in `graph_pointwise/run.cpp`. Fixture construction, backing I/O, and
DeviceVSM evidence interpretation are separate compiled authorities under
`graph_pointwise/run/`; none of those helpers owns or duplicates the run
sequence or the packet oracle.

Reporting is likewise compiled by responsibility. The root
`graph_pointwise/report.cpp` remains the public environment/schema/Report
facade; `graph_pointwise/report/local.hpp` contains declarations only;
`report/csv.cpp` owns the exact CSV field order and row serialization; and
`report/diagnostic.cpp` owns failure-diagnostic projection. These leaves borrow
the existing `Result`, `Facts`, and oracle state and do not create another
acceptance, execution, or result authority.

Rows also carry the raw first-failure diagnostic when the graph failure log has
one: `failure_present`, numeric phase/check, stage, batch, and
`failure_page` copied from `ResidencyStats`. With no log, `failure_present` is
`0`, stage is `UINT32_MAX`, and page is
`ResidencyStats::no_failed_page`; the raw ordinal fields retain their zero or
maximum sentinels. These fields explain an observed failure only and do not
alter packet acceptance.

## Claim Boundary

These diagnostics do not prove a native three-tier scheduler, NVMe scheduling,
terminal-persistent dirty output, Graph throughput, Vulkan throughput, or a
device-independent CPU/GPU winner. Every row remains a packet-sidecar
diagnostic and cannot enter baseline admission.
