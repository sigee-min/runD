# Compute Virtual Residency Contract

This page owns the opt-in `<rund/compute/virtual.hpp>` product and the single
Graph-aware virtual-residency authority. It does not introduce a second
allocator, memory ledger, graph compiler, or public counter record.

## Global VSM authority

The global design has one immutable authority and one mutable authority:

```text
Program graph::Info + operation geometry
        |
        v
ResidencyPlan                         immutable demand/epoch/identity owner
        |
        v
DeviceState::ResidencyAuthority       sole page-to-frame/tier/state owner
        |
        +--> backing/host/device page stores
        +--> backend page mapper
        +--> transfer engine
        |
        v
Pipeline epoch executor               prefetch(e+1), compute(e), writeback(e-1)
        |
        v
Stats + MemoryStats + Profile         only public evidence owners
```

`ResidencyPlan` has two storage representations, not two policies. A monotonic
page stream uses an O(1) formula so a billion-page dataset does not allocate a
billion-row trace. A Graph schedule stores canonical `PageKey(resource,page)`
demands and transitions per execution epoch. Both representations consume the
same page width, frame budget, identity domain and checked arithmetic. No
executor may reconstruct page demand from a Buffer size, a frame ordinal or a
backend command.

The Graph planner canonicalizes every epoch's demand, merging duplicate read
and write access into read-write. Admission fails when one simultaneous demand
set exceeds the frame budget. For a missing page it chooses an unpinned victim
by deterministic farthest-next-use; equal next-use distance chooses the
greatest `(resource,page)` key. A dirty victim always emits `Writeback` before
`Unmap`; a replacement always emits `Fetch` before `Map`. These transition
orders are executable invariants, not telemetry labels.

Window demand includes the exact clamped input halo for each output tile. The
Graph planner can also describe a three-phase hierarchical Scan and a
fixed-order Reduce partial tree. That immutable Graph schedule is a planning
contract today; the public executor does not claim to consume its multi-phase
Scan schedule. The public Scan route instead uses a sequential carry with one
frame, as stated below. No rejected route is counted as virtual execution.

Implementation closure is intentionally fail-closed by layer. The immutable
stream/Graph planner, exact Window and Scan footprint projectors, deterministic
victim choice, Device-global physical frame pool, transactional page mapping,
cross-Pipeline cache reuse, prefetch, dirty writeback-before-reuse, and terminal
dirty drain are implemented and directly tested. The executable public product
supports the exact routes listed below; a planner-only Graph route is not
product completion.

The current authority path is:

```text
VirtualBacking              logical dataset and callback serialization
    -> VirtualBuffer<T>      typed logical extent
    -> ResidencyPlan         immutable page/frame/epoch geometry
    -> Device residency pool sole mutable page-to-frame cache
    -> Device Pipeline budget one admission ledger for retained owners
    -> prepared Pipeline     shared frame arenas + range transfer seam
    -> Stats                 execution/residency evidence
    -> MemoryStats           retained logical/committed memory evidence
```

VirtualPipeline publication also snapshots the prepared Pipeline's initial
`Profile` epoch before any residency epoch executes, binding execution and
memory to one observation. Metal's prepared-stream producer hands
its exact owner-local compile count, cache-hit count, shader-compile duration,
and pipeline-create duration through the existing `AccelRunFacts` handoff;
Compute projects that handoff once. CPU has no native compile/cache producer,
so those coordinates remain unavailable structural zeros rather than a claim
that its nonzero preparation wall time was free. Program authoring and
`Program::compile()` remain earlier public wall-clock phases and are not folded
into prepared-Pipeline facts.

The same `Profile` carries
`PipelineStats::preparation_evidence`. `OwnerLocal`, `NoNativeProducer`, and
`BackendGlobalOnly` describe the physical producer boundary directly; tools do
not derive that boundary from the selected backend. The Metal residency submit
path publishes per-epoch wait duration only through its owner-local
`KernelResult`, which is folded into the terminal `Profile`; it does not mutate
an adapter-global counter or acquire a device-global telemetry lock.

The cold residency integration also proves one private claim authority from
the complete canonical resource set: every resource is Pipeline-owned, its
claim names that exact owner, no alternate or transactional owner exists, and
the input/output ordinals name the disposable execution arenas. Every epoch
rechecks that positive authority before selecting
`PipelineSubmitMode::Residency`. Start,
terminal generation/poison publication, upload, and download then use the
Pipeline gate rather than the Device claim gate. A Pipeline with any external
resource cannot satisfy the proof and retains the ordinary Device claim path.
On Metal, the private transfer route validates the cold-retained
`MetalResidentOwner`'s adapter, id, exact resident ref, and retained MTLBuffer
directly. The immutable Context/Buffer capability projects that owner before
backend lookup, and the Metal owner itself is recovered through its typed
control-block capability rather than a `void*` downcast. Consequently the warm
private route does not acquire either the adapter mutex or resident-registry
mutex and does not mutate adapter-global transfer or dispatch counters; the
returned typed transfer result remains the `Stats` input. Shared Buffer
transfers retain their registry validation, locks, and adapter diagnostics.

One `VirtualPipeline` admits at most one active `run` call. Its owner-local
gate is nonblocking: a second call returns typed `PipelineBusy` before a
backing callback, transfer, or dispatch. Distinct VirtualPipelines retain
independent owner gates. When their input/output views share one or more
`VirtualBacking` objects, execution acquires the complete backing-authority set
without lock-order dependence and serializes every cross-Pipeline callback
critical section.
Thus cross-connected pipelines `A -> B` and `B -> A` cannot deadlock, and their
backing callbacks never overlap across those Pipelines; pipelines over
disjoint backings retain their independent concurrency.

One backing may classify itself as `Persistent` and positively admit two
parallel read callbacks. The Pool then schedules epochs `e+1` and `e+2` into
its two cold-created prefetch lanes while compute executes `e`; a Host backing
or a backing that retains the default one-reader contract uses one lane. This
is a bounded physical-I/O capability, not a change to logical ordering:
writes remain serialized, terminal publication remains ordered, and no runD
worker exceeds the declared read concurrency.

`VirtualBacking` owns one source-private synchronization/poison state through
its single private pointer. The public abstract interface exposes no mutex or
state layout. The authority belongs to the backing object rather than a
`VirtualBuffer` wrapper, so multiple views over the same dataset cannot run
callbacks concurrently or disagree about a partial write. Source-private
`recovery_bytes` is zero when clean. Before the first output callback of
`run(n)`, the runtime publishes `n*b_o` because that callback may fail after
changing a prefix. A fresh view remains unreadable, and `run(0)` or any smaller
positive retry fails with `BufferPoisoned` before any callback. Only a
same-or-larger successful retry overwrites the complete requirement and
returns it to zero. `size_bytes()` is immutable for the lifetime of the
backing. Callback implementations own their storage medium and must not
re-enter a Pipeline holding that same backing.

## Portable product scope

`virtual_pipeline(program,input,output,config)` accepts one logical input and
one logical output. The implemented execution routes are:

- independent Map, including different legal input/output element widths;
- symmetric cross-page Window Sum/Min/Max with exact Clamp or Clip halo fill;
- page-partial Reduce Min/Max/CountNonzero and unsigned U32/U64 Sum, folded in
  deterministic page order to one scalar output;
- pure inclusive or exclusive Scan using a checked sequential carry and one
  resident frame.

Signed/fixed Sum reduction is rejected because independently overflowed page
partials are not equivalent to one globally checked sum. Multi-frame
hierarchical Scan, Sort, Gather, Scatter, indirect-count, recurrence, and other
cross-page graphs remain unimplemented product routes and return
`PrimitiveUnsupported`; no dense fallback is reported as virtual execution.
`ResidencyConfig::{device_resident_bytes,host_staging_bytes}` are complete
retained byte budgets. Zero selects a bounded two-frame default.

Let a Program page contain `E` elements, input/output element widths be
`b_i,b_o`, and the logical element count be `N`:

```text
G_i = checked E_i * b_i
G_o = checked E_o * b_o
P   = ceil(N / payload_elements)
D_f = 2 * (G_i + G_o)              canonical cache + execution pair
H_f = 3 * G_i + G_o                input/output mirror + two prefetch lanes
K   = min(P, floor(device_resident_bytes/D_f),
             floor(host_staging_bytes/H_f))
Q   = ceil(P / K)
L   = checked N * (b_i + b_o)
C   = checked K * D_f
```

The Device-global residency Pool owns four backend Buffers for one compatible
layout. `cache_input/cache_output` are canonical device-tier frames;
`input/output` are disposable execution frames bound into the prepared
Pipeline. Each arena has `K` disjoint ranges. Host retained storage contains a
canonical input mirror, an output/writeback mirror, and two prefetch images.
The Pool holds one lifetime reservation from the existing Device Pipeline
budget; every compatible VirtualPipeline shares that owner and does not reserve
the bytes again. Vulkan may additionally retain a Pipeline-private native
transfer command arena; it does not own a second page cache.

The prepared count `M=N` is capacity. `run()` is exactly `run(M)`; `run(n)`
accepts the active prefix `0<=n<=M` without rebuilding or changing identity.
One source-private projection owns the invocation geometry:

```text
P_n = ceil(n / E), with P_0 = 0
Q_n = ceil(P_n / K), with Q_0 = 0
A_n = min(P_n, K)
```

`PipelinePlan::residency`, retained memory and backing identities remain the
full-capacity coordinates. `ResidencyStats::active_count`,
`resident_frames_peak=A_n`, epoch/page-in/page-out counts and traffic are latest-run
coordinates. `n>M` is `ShapeMismatch` and changes no evidence. Clean `n=0` is
an accepted terminal with the empty FNV output identity and zero backing
callbacks, dispatches, submissions and transfers. A backing recovery
requirement instead makes it a zero-callback `BufferPoisoned` failure. If a
terminal dirty writeback fails, the cache authority discards every affected
device frame before releasing its lease. The poisoned backing retains the
recovery obligation, but no later Pipeline or different backing can inherit
partially published bytes as a cache hit or eviction source.

For active epoch `q<Q_n`, the runtime performs this ordered state transition
while holding the Pipeline and both backing authorities:

```text
probe the Device-global cache and select deterministic victims
write back every dirty victim before its frame can be reused
read/prefetch complete missing logical pages into the canonical host mirror
upload misses into cache_input, then project demanded frames to execution input
execute the prepared K-frame Pipeline
download demanded execution output and project it to cache_output
hash exact logical bytes; defer backing writes until eviction or terminal drain
```

Thus output backing traffic for one successful `run(n)` is exactly `n*b_o`.
Input backing traffic is the complete byte extent of cache misses and may be
lower than `n*b_i`; `page_in_count + cache_hit_count = P_n`. Physical input
traffic is `page_in_count*G_i`, and physical output traffic is `P_n*G_o`.
Fixed prepared commands may consume zero-padded frames, but those frames never
reach backing callbacks, output hashing, or publication. Active size affects
epoch count and traffic, not retained frame capacity. The compact linear plan is
`O(1)` in `P`, so even a billion-page logical dataset does not allocate a
billion-entry demand vector.

The executable CPU, Metal, and native-Vulkan paths use ordinary backend
Buffers. On unified-memory Metal the owner-local cache/execution projection
validates retained Buffer capabilities and copies disjoint Shared ranges
directly; it creates no command buffer, takes no adapter/resident registry lock,
and allocates nothing on the warm path. Shared public copies keep their ordinary
backend command and lock semantics.
`PipelinePlan::residency` reports the logical geometry and fixed resident-set
extent. The Device-global Pool's actual backend allocation granularity is
reported by its existing Buffer/Device memory owners, while Pipeline-private
native owners remain in `PipelinePlan::committed_peak_bytes`. Neither charge is
inferred from frame geometry.

On a Metal 4 SDK and runtime, an allocation-free capability query first gates
VirtualPipeline preparation. A supported path reserves the Device Pipeline
budget's complete currently available capacity before creating any native
owner; zero available capacity therefore fails without a Metal allocation.
While that reservation excludes concurrent preparation, the backend
transactionally materializes one Pipeline-owned submission queue, command
allocator, reusable command buffer, shared event, and residency set. Its exact
exposed retained charge,

```text
M_submit = command_allocator.allocatedSize + residency_set.allocatedSize
```

must fit inside the reservation. Preparation checked-adds `M_submit` to
`prepared_native_bytes`, `prepared_bytes`, `peak_bytes`,
`committed_peak_bytes`, `total_bytes`, `logical_bytes`, `live_bytes`, and
`physical_bytes` in one unpublished `PipelinePlan` candidate. It then
partitions and commits the exact charge, refunds the surplus capacity, and
publishes the plan and ticket to the same Pipeline state. Query, reservation,
materialization, plan arithmetic, alternate preparation, or commit failure
publishes none of those owners.
Staged Metal owners remain candidate-local under the capacity reservation;
they do not mutate either prepared stream or its memory meter. Every failure
destroys both candidates before refunding the reservation. Only after the exact
Device ticket commits does the no-fail publication step request residency,
attach each candidate to its prepared stream, and add the same exact bytes to
the prepared-memory meter.
Queue, command, event, and residency objects expose no separate byte-size
coordinate and create neither a second memory ledger nor a hidden first-run
owner. The prepared-memory meter includes `M_submit` in backend device memory.
Transactional Pipelines apply the same rule to both prepared streams and admit
their checked sum.

The first successful VirtualPipeline epoch uses the classic Metal queue. That
completion is the required ICB prime and is the only transition from `Cold` to
`Ready`. Later VirtualPipeline epochs reuse the same Metal 4 command allocator,
command buffer, residency set, and classic-prepared ICB ranges. Each epoch is
`begin, useResidencySet, execute ICB ranges, end, commit, signal event, wait,
reset`; the shared-event wait has a literal 5,000 ms upper bound and allocator
reset occurs only after the event reaches the submitted value. A failed classic
completion never promotes `Cold`; a failed Metal 4 completion makes that owner
unusable. In either case the failing epoch is not downloaded or written to the
output backing. The ordinary Pipeline terminal law remains authoritative: a
poisoned or device-lost Pipeline rejects later attempts, while a non-poisoning
failure leaves the VirtualPipeline eligible for a later standard submission;
no failed allocator is reused. The Pipeline submission gate and ordinary
resource claims cover both queues, so a classic and Metal 4 submission cannot
overlap on the same Pipeline.

This is a source-private submission policy, independent of timing. Ordinary
Pipeline, dispatch tracing, and devices built or running without Metal 4 keep
the classic path and its existing submission cardinality. The actual Apple M4
product contract proves the cold prime plus sixty warm public runs with exact
output/hash/traffic/submission evidence and zero process-global warm
allocations; it does not generalize that driver result to other devices.

## Immutable demand schedule

For a monotonic stream, `ResidencyPlan` stores only the page width, logical page
count `P`, selected frame capacity `K`, and their identity. The planner receives
`page_bytes,P,requested_frames,max_frames`, validates the request, and selects
`K` exactly once:

```text
K = min(P, requested_frames)
W = P == 0 ? 0 : ceil(P / K)
```

`P > 0` with zero requested frames is infeasible, and a request greater than
`max_frames` is rejected. `W` is derived rather than stored. Epoch `w < W` is
obtained by formula, not by an O(`P`) trace:

```text
first(w) = checked_mul(w, K)
count(w) = min(K, P - first(w))
```

For a Graph schedule the plan instead owns canonical epoch use spans and the
minimal ordered transition stream. The victim calculation consumes future
epoch demand directly, so it is Belady-optimal for the frozen schedule rather
than an LRU approximation. Thus a 500 GB monotonic stream retains constant-size
plan state while a non-monotonic Graph retains only its checked execution
demand. Execution counters remain runtime evidence rather than planner state.

The four frame arenas are ordinary backend Buffers. The canonical cache pair
and disposable execution pair have the same `K * page_bytes` geometry. The
Device-global Pool queries the exact backend allocation requirement for each
arena, reserves their checked sum plus its fixed host images once from the
existing Device Pipeline budget, and owns that reservation until the Pool is
destroyed. On Metal the Buffer requirement is the resulting resource's
`allocatedSize`; on Vulkan it is `VkMemoryRequirements::size`. A compatible
Pipeline retains the Pool by shared ownership but does not copy the Pool charge
into its private `PipelinePlan::committed_peak_bytes`, so two Pipelines cannot
double-admit or double-report the same physical owner. The Pool's live Buffers
and host images remain visible through the Device/VirtualPipeline memory view.
Vulkan pool reuse is eligible only for an exact committed-size match, so an
oversized best-fit candidate cannot change the admitted Pool coordinate.

Before any Pipeline-private owner is
materialized, Vulkan creates an unbound transfer `VkBuffer`, reads its
`VkMemoryRequirements::size`, and destroys it. That exact charge is checked
into `prepared_native_bytes`, every aggregate plan coordinate, and the existing
Device Pipeline reservation. After admission commits, transfer preparation
stages an unpublished exact-size native candidate, records both commands,
verifies its `allocated_bytes` against the sealed requirement, and only then
replaces the prepared command owner. Candidate failure destroys its commands
and allocation. Host backing and host staging remain separately owned
coordinates; no residency-specific ledger or allocator is introduced.

## Capacity, immutability, and identity

Checked arithmetic rejects counter and epoch overflow. The compact completed
object has no mutation API or problem-size-dependent storage.

Identity uses domain `rund.compute.pipeline.residency`, policy version 2. A
stream identity covers `stream,page_bytes,P,K`. A Graph identity covers
`graph,page_bytes,K` and the canonical `(node,tile,PageKey,access)` demand
sequence. Derived transitions and execution counters do not add identity
inputs. Changing demand, page width or frame capacity changes identity.

## Execution capability boundary

The global-cache software-residency product is implemented and
product-verified on CPU and Metal. The native Vulkan frame implementation is
compiled and retains its exact
transfer-admission contract, but no native Vulkan product device is available
in the Apple release environment, so its execution status is unverified.

Vulkan adapter creation freezes the presence of
`VK_KHR_portability_subset`. The selected BackendOps projects that immutable
fact through the existing Compute DeviceOps capability interface. A
portability-subset adapter returns typed `BackendUnsupported` before planner,
Pipeline, Buffer, transfer, residency, or backing-callback materialization.
That typed result is terminal for public preparation. CPU is positively
admitted by the Compute capability owner, and Metal positively publishes
support through the accelerator interface.

After that gate, Device Pipeline admission remains the physical capability
authority: successful preparation proves the selected backend can materialize
the exact committed working set before any backing transfer or run is
attempted.

## Verification and integration boundary

The implementation ownership map is:

```text
node/include/rund/compute/virtual.hpp          public typed facade
node/src/compute/virtual/state.hpp             prepared owner state
node/src/compute/virtual/prepare.cpp           CPU/DeviceOps capability gate
node/src/compute/virtual/active.{hpp,cpp}      active-prefix projection
node/src/compute/virtual/run.cpp               lock and phase orchestration
node/src/compute/virtual/run/projection.*      arena and epoch coordinates
node/src/compute/virtual/run/backing.*         backing callbacks and recovery
node/src/compute/virtual/run/cache.*           cache/execution projection and writeback
node/src/compute/virtual/run/epoch.*            transfer and dispatch sequence
node/src/compute/virtual/run/evidence.*        Stats terminal publication
node/src/compute/virtual/run/sample.cpp        warm sample epoch lifecycle
node/src/compute/virtual/stats.{hpp,cpp}        per-epoch Stats accumulation
node/src/compute/virtual/observe.cpp            Profile and plan observation

node/src/compute/pipeline/residency/authority.* private-resource proof
node/src/compute/device/residency.*             mutable frame-state authority
node/src/compute/device/residency_pool.*        Device-global tier owners
node/src/compute/device/residency_prefetch.*    fixed prefetch worker/storage

node/src/accel/kernel/prepared/model.hpp         owner-local cold facts
node/src/compute/stats.cpp                       sole cold-fact projection

node/tests/contract/compute/virtual/            planner and backing oracle
node/tests/contract/compute/virtual/product/    public product contract leaves
node/tests/contract/compute/virtual/product/concurrency/pipeline.cpp
                                                same-owner nonblocking gate
node/tests/contract/compute/virtual/product/concurrency/backing.cpp
                                                backing-set serialization
tools/measure/compute/virtual/                  public cold and warm evidence
```

Each production leaf is compiled directly by the Node source registry. The
product contract mirrors user-visible responsibilities across preparation,
execution, active prefixes, backing failure/recovery, owner and backing
concurrency, evidence, width, and surface leaves.

`compute.pipeline-transfer` separates its reusable fixture model, exact
success/Stats/MemoryStats accounting, claim-and-failure law, and canonical
download/hash order into independently compiled leaves. The Vulkan fused
completion owns physical H2D/D2H byte publication; the mapped host copy owns
readback duration and returns that same duration through the typed download
result consumed by Pipeline Stats.

`compute.pipeline-residency` owns compact billion-page epochs, zero and bounded
capacity, exact first/tail epoch projection, and canonical geometry identity.

The Accel bridge keeps its public `DeviceOps` table in
`node/src/compute/backend/accel.cpp`. Its virtual-residency leaves have one
source-private interface and one owner per physical responsibility:

```text
node/src/compute/backend/accel/local.hpp       DeviceOps leaf contracts
node/src/compute/backend/accel/buffer.cpp      committed-size query/allocation
node/src/compute/backend/accel/transfer/prepare.cpp   transfer preparation
node/src/compute/backend/accel/transfer/upload.cpp    fixed-arena upload
node/src/compute/backend/accel/transfer/download.cpp  fixed-arena download
node/src/compute/backend/accel/residency.cpp   candidate admission transaction

node/src/accel/backend/ops/table.hpp           backend capability interface
node/src/accel/vulkan/adapter/state.hpp        frozen portability-subset fact
node/src/accel/vulkan/ops.cpp                  Vulkan capability projection

node/src/accel/vulkan/kernel/pipeline/transfer/materialize.cpp  candidate and composite command
node/src/accel/vulkan/kernel/pipeline/transfer/upload.cpp       mapped input publication
node/src/accel/vulkan/kernel/pipeline/transfer/download.cpp     terminal copy, hash, and timing
```

The residency leaf alone stages the Metal submission candidates, checks their
retained sum against the Device Pipeline budget, projects that exact sum into
the unpublished plan, commits the admission ticket, and publishes the owners.
For Vulkan, the buffer leaf supplies the allocation-free transfer commitment
query to the common cold plan. The transfer preparation leaf verifies and
publishes the candidate only after that common admission is committed; its
materialize, upload, and download files each own one physical responsibility.
None of these leaves owns a second memory counter.

`compute.pipeline-vulkan-transfer` checks the sealed Device commitment against
`PipelinePlan`, the exact native transfer charge against
`MemoryStats::staging`, and a Device budget one byte below the sealed total on
native Vulkan. The latter rejects with zero backing callbacks, zero Device
Pipeline commitment, and no change to any Device `MemoryStats` allocation
coordinate. A portability-subset adapter stops at the earlier product
capability boundary, leaving this native-only physical proof unexecuted.

`compute.virtual-residency-product` uses only the public facade. Its capability
leaf first classifies each selected backend. An executable backend proceeds to
cold preparation, one cold execution, sixty warm
executions, one terminal observation, output/hash parity, poisoned tail,
backing read/write failure and retry, backing-wide poison across a fresh view,
fixed plan/memory capacity, exact page/epoch/backing/frame-transfer/submission
facts, zero process-global warm C++ allocations, and zero claim duration and
conflicts for every counted warm sample. `allocation_free_runs` advances only
when both the allocation predicate and those two owner-local claim facts are
zero, so the terminal `Profile` certifies the complete unobserved warm cohort
without adding another counter owner. On a portability-subset
Vulkan adapter the leaf instead proves exact typed `BackendUnsupported`, zero
backing callbacks, and unchanged Device `MemoryStats` and Pipeline-admission
coordinates, then reports the backend as blocked rather than product success.
A second product case
executes a legal 64-bit-input to 32-bit-output Map so the input and output frame
widths cannot be collapsed into a hidden equal-width assumption.

The cache leaf prepares two independent VirtualPipelines with the same Device,
Program and input backing but different output backings. The first run
materializes one page. The second must issue zero input backing reads, publish
one cache hit and one dirty terminal writeback, and produce the same output.
This is the positive proof that mutable page residency is Device-global rather
than a Pipeline-local residency mirror.

The same-capacity leaf first runs and terminally verifies one full-capacity
cold execution, resets the output to tail poison, then runs `{0,7,35,M}`
through that same prepared owner. Every active count has one unobserved
conditioning execution, sixty sampled warm executions and one terminal
observation. It proves the active projection, zero-work law, inactive logical
tail poison, exact prefix hash and callback deltas, and invariant plan/memory
identity. The failure leaf copies a prefix before returning failure, rejects
zero and insufficient recovery runs without callbacks, then proves a complete
retry clears the backing requirement.

The concurrency leaves use only the public facade. The owner case blocks the
first backing callback and proves that a second call on that exact prepared
VirtualPipeline terminates with `PipelineBusy` and adds zero callbacks before
the first call is released. The backing case starts two distinct pipelines
over cross-connected backing views, proves callback `max_active == 1`, and
requires both runs to reach successful terminals. Barriers and condition
variables establish the phases; every wait has a fixed liveness bound and no
sleep interval contributes to acceptance.

`begin_samples()` and `end_samples()` delimit the complete logical-run cohort.
As with an ordinary Pipeline, `profile()` returns `ProfileBusy` while that
cohort is active; only the terminal Profile captured after `end_samples()` can
certify every sampled run.

The native Vulkan fixed-frame owner cold-records one immutable compute secondary
and one reusable primary containing upload copy, compute execution, and
download copy. An epoch therefore publishes one compute submission and zero
separate transfer submissions. This path is reachable only after the adapter
capability gate positively identifies native Vulkan. The Apple release Vulkan
adapter exposes the portability subset, so public product preparation is
blocked before this owner is created and produces no Vulkan performance row.

`compute.virtual-residency-oracle` remains the backend-independent one-page
reference transform and poisoned-tail golden. It proves backing range order,
terminal-only observation, fixed working storage, output identity, and warm
allocation freedom without calling production planner or execution helpers;
`compute.pipeline-residency` independently owns the `P,K,W` mathematics. The
current-source measurement route under
`tools/measure/compute/virtual` has directly compiled orchestration, oracle,
report and wall-sample leaves. It first records one full-capacity cold row from
authoring through compile, prepare, seed, `run(M)`, and terminal backing read,
with output parity, hash and Profile checks. It then resets the output fixture
to tail poison and runs active counts `{0,17,32769,M}` through that same
prepared owner. Each active row has one unobserved conditioning run, a
sixty-run sample cohort, and exactly one terminal backing read and Profile,
plus warm p50/p95, active-prefix throughput, backing and physical
frame traffic, submissions, fixed memory, sample cohort, plan/graph/result
identity, and terminal-only observation. It is diagnostic evidence, not an
installed baseline row. The report leaf copies execution, cache, download,
timing and memory fields from the terminal `Profile` and frozen plan without a
counter delta. One public post-prepare `Profile` snapshot owns separately
labeled `prepare_*` compile/cache/allocation/timing evidence from its execution
projection and memory evidence from the same epoch because first execution
starts a distinct terminal epoch; `terminal_*` remains the terminal `Profile`.
Its warm-allocation evidence is explicitly runD-owned. The
route measures CPU and Metal. A portability-subset Vulkan request stops at the
typed capability boundary and reports blocked without a timing row; native
Vulkan measurement awaits a product device.

## Explicit remaining performance boundary

The current executor overlaps backing prefetch for epoch `e+1` with Pipeline
compute for epoch `e`. At the next epoch it also performs dirty backing
writeback before waiting for that prefetch, so independent input/output backing
I/O overlaps in the product contract. It does **not** yet overlap a device copy
into a second execution bank with the preceding GPU dispatch: there is one
execution bank, and cache-to-execution projection completes before submission.
The public Scan route is checked and cross-page correct but sequential (`K=1`),
not the planner's hierarchical multi-frame Scan. NVMe scheduling, cross-layout
victim selection, native sparse mapping, Window/Reduce/Scan graph composition,
and a persistent dirty cache across public run terminals are not implemented.
A typed rejection or this documented boundary is not counted as those features.
