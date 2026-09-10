# Virtual Residency Product

This page owns the public `VirtualBacking`, `VirtualBuffer<T>`,
`VirtualPipeline<R(A)>`, `ResidencyConfig`, active-prefix, and route contract.
Mutable cache state, physical memory, graph demand, and backend terminals are
owned by the sibling pages linked below.

## Surface

`virtual_pipeline(program,inputs...,output,config)` accepts one through seven
logical inputs and one logical output within the selected route's narrower
admission. `VirtualBacking::size_bytes()` is immutable; checked
`read` and `write` callbacks own their storage medium. `VirtualBuffer<T>` is a
typed logical view and never materializes the full logical extent as resident
Buffers.

`ResidencyConfig::{device_resident_bytes,host_resident_bytes}` select the
eligible per-Pool working set. Zero uses the bounded two-frame-per-bank
default. The Device Registry budget, not a second VirtualPipeline budget, caps
new physical arenas. Exact capacity and retention formulas are owned by
[Pool](./pool.md).

`Session::compute(VirtualPipeline&)` is the same public Request/Submission/
Poll/Completion bridge as the resident Job and Pipeline surfaces. An eligible
strict DeviceVSM request retains its `VirtualPipelineState`, prepares and
accepts the native command, then exposes a `Pending`/`BackendSubmitted`
boundary until the backend callback validates terminal evidence and wakes the
retained continuation. That continuation alone performs the common Final,
Stats, and publication transition. Ordinary routes remain queued and
worker-blocking except for the mapped all-staged unary Pointwise `StagedLoop`
contract below; general nonresident Forecast/Promote/Drain/Persist recurrence,
Host epoch-zero service outside the documented fallbacks, GPU-driven execution,
and the 100x latency target are not implemented here. Cancellation is accepted
only before the worker claims execution. A request already `Running` reports
`AlreadyCompleted` for a late cancel, and a concurrent virtual run reports the
existing `PipelineBusy` reason.

### Ordinary Pointwise route priority

For unary nonresident Pointwise with `Q>=2`, all-staged input and output first
probe `StagedLoop`. Admission requires exact mapped Host-visible/coherent views
for every input and the output; the route enum is the sole decision carried
into preparation, so shape, tier, and max-read policy are not recomputed. The
admitted loop owns one native recurrence submit, Q GPU epochs, zero transfer
submits/bytes, zero Host epoch submit/service/callbacks, and one
Final/Authority/output publication/version. A mapped structural
`BackendUnsupported` before owner, rearm, stage, lease, or native acceptance is
a clean decline to the legacy route. Any owner mutation, lease/native
acceptance, or non-capability failure is terminal and cannot fall back.

The legacy nonresident Persistent Pointwise R2 route is `BackendChunked` with
two-coordinate chunks and `ceil(Q/2)` physical submissions; old `OneSubmit`
evidence is lower-level fallback evidence only. Unsupported Persistent spatial
Window declines before lease to dedicated bounded Window (`Q<=4`) or Stream
(`Q>4`) with one logical handoff, one Final, and one output version; current
Metal fallback queue calls may be Q. Explicit/resident DeviceVsm Window is
separate from that fallback: a finite centered I32/U32 Window may select the
typed `WindowRingFused` proof for either Resident or fully pre-staged
input/output. The proof accepts bare forms and exact authenticated,
parameter-free `CanonicalTotalU32` Map chains of symmetric depth 1–3 before
and after the Window. That route has one physical dispatch/submit, `Q` seed
and `Q` compute epochs, `2Q` internal phases, no Host epoch service/callback
or transfer submission, and one Final/publication/version. It does not admit
I32 Map fusion. Dynamic
true fixed-R PagedLoop remains blocked by the absence of a portable same-submit
GPU↔Host system-scope rendezvous/forward-progress primitive.

`virtual_pipeline(..., GraphPageMap, config)` is the separate public overload
for the nonresident GraphPointwise page-binding view. Its entries use dense
public input ordinals and are copied during preparation. A mapped external
input must be a read-only Backing resource; the full `K <= 32` template and
the actual tail `q` are checked for bounds and bijection before the remapped
plan is published or Authority is mutated. A bounded shape probe may run
earlier to derive the physical footprint. Mixed, internal/output, transient,
ReadWrite, and resident DeviceVSM remaps remain rejected or partial under
[Graph](./graph.md).

## Preparation ownership

Virtual preparation keeps one orchestration boundary in
`node/src/compute/virtual/prepare.cpp`. Its private immutable admission
owners are split as follows:

```text
virtual/prepare/geometry.cpp
    VirtualGeometry route classification and physical-shape identity
virtual/prepare/window.cpp
    Window endpoint, ring mode, and budget preflight
virtual/prepare/validation.cpp
    Program/buffer contract and backend capability admission
virtual/prepare.cpp
    public validation, fusion probe, and graph/multi/single route dispatch
virtual/prepare/residency.cpp
    single-route budget, ResidencyPlan, Pool, and immutable state publication
virtual/prepare/residency/bank.cpp
    one physical Pipeline bank preparation and backend selection admission
virtual/prepare/residency/local.hpp
    declarations-only seam; no retained plan or pool authority
```

`VirtualGeometry` and `VirtualWindowPreflight` are copied into the prepared
state once. Runtime route predicates consume those records; they do not
recompute preparation-time geometry, endpoint, or budget decisions. Backend
capability decline remains a clean pre-mutation decline, while failures after
Pool/Pipeline mutation retain their existing terminal ownership.

Nonresident GraphPointwise with a nonempty map uses a private proof-owned
PageMap binding for the
nonempty resource-wide `S=1` full-page permutation. It is the exact 824-word /
3296-byte map image with `K <= 32`, full target/source bijection, and tail
validation described by [Graph](./graph.md); the GPU source lookup is the only
reordering authority and active preparation requires zero parameter bytes.
Resident GraphResident uses the same private image independently. Other
resident DeviceVSM remaps remain unsupported. Partial-byte, multi-source, and
internal/output/transient forms remain outside this binding.

## Implemented Routes

The listed routes execute through the public product on CPU, Metal, and
Vulkan. The Vulkan evidence on Apple is the MoltenVK portability path, not a
native sparse-capable Vulkan-device result.

- pointwise Map graphs, including legal unequal input/output widths;
- symmetric cross-page Window Sum, Min, and Max with Clamp or Clip fill;
- centered Clamp/Clip Window composed with either the exact U32
  wrapping-immediate Maps or up to three independent parameter-free
  canonical-total single-read/value-write U32 Map DAG stages before and after
  it;
- page-partial Reduce Min, Max, CountNonzero, and unsigned U32/U64 Sum, folded
  in deterministic page order;
- pure inclusive or exclusive hierarchical Scan over all admitted frames,
  plus an exact canonical total element-local one-through-seven-read/one-value-write
  U64 Map expression DAG followed by U64 Scan;
- the recurrent canonical total U64 pointwise Map-chain-to-Reduce route owned
  by [Graph](./graph.md), including the exact two-public-input
  Map-to-{Sum, CountNonzero, Min, Max} service-free boundary.

Reduce currently requires one collective node. Scan accepts either that pure
shape or exactly one canonical total element-local U64 Map expression DAG with
one through seven public reads and one terminal value write before the collective. An eligible pure
unsigned U32 or U64 Sum, CountNonzero, Min, or Max Reduce, or
Inclusive/Exclusive Scan, or the admitted composed U64 Scan, selects the
DeviceVsm one-submit path. Reduce uses
one GPU workgroup to accumulate all pages into one scalar; Scan uses one GPU
workgroup to own the carry across all pages. Both return one aggregate Final.
Sum and Scan prove overflow before an output write; CountNonzero is bounded by
the admitted U32 logical-element limit, and Min/Max consume a nonempty extent.
Signed, fixed, segmented, and other unsupported Reduce/Scan types retain their
established Host routes. Clip Window with a preceding Map is DeviceVsm-only:
the service-aware identity-fill fallback is rejected because transforming its
padded identity could count inactive tail elements. Signed or fixed Sum is rejected because independently
overflowed page partials are not equivalent to one globally checked sum.
Multiple collectives, multiple logical backing intermediates, Sort, Gather,
Scatter, indirect-count, recurrence, and other cross-page routes remain
unimplemented and return typed rejection. A rejection is not product
completion and no dense fallback is reported as virtual execution.

## Active Prefix

The prepared count `M` is capacity. `run()` means `run(M)` and `run(n)` accepts
`0 <= n <= M` without rebuilding or changing plan identity. For page payload
`E`, page count `P_n`, and per-bank capacity `K`:

```text
P_n = ceil(n / E), with P_0 = 0
Q_n = ceil(P_n / K), with Q_0 = 0
A_n = min(P_n, 2K)
```

The plan and retained-memory coordinates stay at full capacity. Latest-run
statistics use `n`, `P_n`, `Q_n`, and `A_n`. `n>M` is `ShapeMismatch` and
changes no evidence. A clean zero run is an accepted terminal with empty FNV
identity and no callbacks, dispatches, submissions, or transfers. A backing
recovery obligation instead makes it a zero-callback `BufferPoisoned` failure.

Only exact logical output bytes reach backing. Fixed prepared commands may
consume zero-padded physical frames, but inactive bytes cannot reach hashing,
callbacks, or publication. A complete cache frame is active-extent
independent; only a boundary frame whose fill depends on the visible prefix
retains that extent in its key.

## Evidence

`PipelinePlan::residency` publishes immutable logical geometry and identity.
`Stats::pipeline.residency` publishes the latest execution/cache/backing facts
and the explicit warm sample cohort. `Stats::{uploaded_bytes,
downloaded_bytes}` report physical frame transfers only. Profile preparation
facts and terminal execution facts are distinct epochs; an unavailable native
producer remains unavailable rather than becoming a measured zero.

The implemented and rejected behavior is proved through the public facade by
the surfaces in [Verification](./verify/README.md). Performance interpretation is not
owned here; see [Virtual Performance](../../../../../docs/reference/performance/virtual/README.md).
