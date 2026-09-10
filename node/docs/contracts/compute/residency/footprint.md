# VSM Footprint Alignment

This page owns the page/epoch alignment law for Window and typed Graph
execution. It separates canonical backing pages from expanded kernel tiles so
an overlap is represented by one page identity and pin, not by duplicate
backing reads into two materialized frames.

Status: the geometry projection and the resident Stencil shared-halo kernel
are implemented. Direct Plan now seals the exact backing-to-frame read range,
including halo boundary offsets. Authority now mints a short-tail pointwise
descriptor with a sealed `ZeroInactiveTail` fill: the Host service clears the
inactive suffix before its exact final backing slice is read, and inactive
native lanes cannot observe or publish that padding. Centered stride-one
Window now seals the exact scalar-aligned `Clamp` repeat or `Clip` identity
recipe plus the canonical multi-page `FootprintEpoch`. Authority fetches each
canonical page into one `K+2` Host ring, reuses the exact opposite-bank
resident page under a callback pin, and mints one Promote/Assemble capability
for all canonical-to-expanded Device slices. Finite-retention live sets that
cannot fit H still fail cold.

This closes duplicate backing reads for the ordinal Direct Window path. For
`P` pages, payload width `T`, radius `r`, and scalar bytes `b`, the
service-aware path reads exactly

```text
logical_elements * b
```

and assembles `P*(T+2r)*b` Device bytes from those canonical Host pages. Actual
Metal and Vulkan contracts exercise `P=9`, `Q=5`, W=2, U32 `Clamp(Sum)` and
`Clip(Sum)` with a short final page, exact logical backing bytes, one native
submit, zero epoch submits/callbacks, and one aggregate Final/publication.

Direct Plan also seals one distinct canonical backing materialization plus a
fixed-size `WindowFootprintProjection`. For an epoch of at most `K` target
frames it returns at most `K+2` unique canonical source pages and at most
`3K` exact `(source page, target frame, source offset, target offset, bytes)`
slices. The projection carries exact next-use and closed current-epoch pins;
the short canonical tail is an authenticated zero-inactive-tail page. Seal
rejects an aliased canonical/expanded identity or a Host ring too small for
the maximum source set. The service-aware Window product consumes this
projection through its Authority-minted Fetch and Promote/Assemble
capabilities. Canonical pages occupy the bounded `K+2` Host ring; an
opposite-bank reuse pins and copies an already Resident canonical page instead
of rereading backing. Promote binds those exact sources, the `K` expanded
Device targets, every slice, and the boundary-fill recipe in one move-only
callback-return capability. Actual Q=5 contracts therefore read 420 logical
input bytes once, assemble 576 expanded Device bytes, submit the whole native
recurrence once, and publish one Final. General Graph multi-page consumers and
asymmetric/multi-pass Window footprints remain partial.

The separate DeviceVsm product now implements centered, stride-one Clamp/Clip
U32 Window Sum/Min/Max overlap elimination without materializing halo-expanded
Host frames. It reads
the logical backing once into one whole-run GPU resident buffer, executes the Q
page coordinates in one device loop, and publishes one logical output buffer.
For adjacent pages it authenticates exactly
`(Q-1)*(read_prefix+read_suffix)` avoided physical page-in bytes. Its
fixed-size proof also seals the canonical page width, the exact `Q-1` adjacent
page-boundary transitions, and a modulo-2^32 checksum over every epoch's
`(epoch, first canonical page, last canonical page, active core elements)`.
The Metal and Vulkan shaders accumulate those two footprint result words while
advancing the page loop; aggregate Final rejects a missing, duplicated, or
misprojected traversal before publication. Clamp uses
the frozen Direct or SharedHalo path. Clip Sum uses the frozen
PrefixDifference path; Clip Min/Max use BlockPrefixSuffix. DeviceVsm executes
those Clip passes inside one whole-run payload dispatch and reuses its private
input/output staging as prefix/forward/backward rows, so it adds no Host epoch
turn and no separate Q-sized scratch allocation. This closes page/epoch
projection authority for the whole-run-resident centered Window product
subset; it does not implement the
canonical multi-page `FootprintEpoch` as physical Forecast bindings for the
service-aware fallback, general Graph consumers, or asymmetric/multi-pass
Window shapes beyond those exact Clip algorithms.

## Canonical page and expanded tile

Let `b` be element bytes, `T` the core output elements of one VSM page, `W`
the backend workgroup width, and `r` a symmetric Window radius. Admission
requires:

```text
T mod W = 0
G = T * b
core(e) = [e*T, min(N, (e+1)*T))
footprint(e) = clamp([e*T-r, (e+1)*T+r), [0,N))
```

`G` is the canonical backing/cache page geometry. `footprint(e)` is a set of
canonical page identities plus exact boundary byte slices; it is not another
cache key and is never persisted as a halo-expanded backing page. Adjacent
Windows reuse an overlap because their footprints name the same canonical
page under overlapping pin intervals.

Canonical and expanded tail extents are sealed independently. If s is the
scalar width, N=logical_bytes/s is the active element count, and P is the
canonical payload width, the raw canonical extent is zero when N mod P = 0
and is N otherwise. The expanded/frame extent records the clipped or fill
tail of the halo-expanded frame and can remain N even when the canonical
extent is zero. Thus N=48, F=16, R=2, and P=12 has canonical extent zero but
a nonzero expanded tail; the two values remain part of distinct
materialization identities. This is seal geometry only: it does not claim a
GPU-owned fixed-native recurrence or a Graph halo materialization.

The current Persistent boundary is still an O(Q) preencoded, HostCoherent
service boundary. It bounds the fixed-W transaction and its callbacks, but
does not turn the recurrence or its halo ownership into a GPU-generated
fixed-storage operation.

The Device tile may be expanded to `T+2r` elements or loaded a workgroup at a
time. A shared-halo workgroup of width `W` and admitted halo capacity `C`
requires:

```text
shared_bytes = checked (W + 2*C) * b
groups_per_page = T / W
```

The loader assigns contiguous vector-width segments to adjacent lanes and
executes one uniform barrier before inactive tail lanes may return. Clamp or
Clip fill occurs after the exact canonical slices are resident. The backend
may select Direct, SharedHalo, prefix, or block-prefix/suffix lowering only
through the frozen Range planner; VSM does not rank a second kernel family.

For the implemented I32 Window shape, `T=4096` and `W=256`, so one canonical
page is sixteen workgroups. The admitted SharedHalo capacity `C=256` consumes
`(256+512)*4=3072` bytes per workgroup and no global intermediate. These are
geometry and resource facts, not a speedup claim.

## Page-use projection

Window projection emits one canonical `PageUse` for every page intersecting
`footprint(e)`. Duplicate page identities within a node are canonicalized by
unioning access and dirty extent, taking the earliest ready/prefetch epoch,
the nearest next use, and the closed union of pin intervals. Authority mints
the physical bindings from that row.

Graph projection consumes the existing `TiledGraphInvocation::project`
facts. For every resource/port use it preserves:

```text
(PageKey, access, dirty, next_use, pin, prefetch_epoch, ready_epoch)
```

Transient resources pin from producer through last consumer and never enter
backing Forecast/Persist. Backing resources may be fetched before their exact
consumer pin, but Device promotion and physical alias reuse still wait the
projected resource predecessor. Interval coloring remains the sole physical
alias authority.

The runtime may keep a canonical Host page across multiple Window or Graph
consumers when its next use wins the deterministic Authority victim scan. It
may not keep a raw pointer after the authenticated Host-slot generation is
retired. The pointwise Graph product now represents that Host ownership with a
typed Forecast capability and gates same-bank reuse on explicit release after
the callback and H2D consumer return. This closes raw-token lifetime for the
already-canonical one-page Graph input. Direct Window now lowers every sealed
`FootprintEpoch` into canonical Host page bindings. Its Authority-minted reuse
capability authenticates an opposite-bank source key/frame/full visible slice
and callback pin; Promote/Assemble then authenticates every
canonical-to-expanded Device slice. Ordinal Direct consumers therefore share
canonical bytes without duplicate backing reads. Arbitrary non-ordinal Graph
consumers still cannot name this Direct-only Assemble transaction. DeviceVsm
separately registers its complete resident input and output set with Authority,
seals the fixed-size Window footprint authority, and requires the GPU aggregate
transition count and projection checksum at Final. It bypasses the
service-aware ring for its admitted whole-run resident Window subset and exact
U64 Map-to-Reduce Graph subset (Sum, CountNonzero, Min, and Max). The Direct
physical Forecast/Assemble path is independently implemented above; neither
path is evidence that arbitrary non-ordinal Graph footprint binding or the
general Graph wavefront is complete.

## Fusion and materialization elimination

Fusion is legal only when all of the following are frozen before Pipeline
identity is minted:

1. the producer is pure and has one consumer in the fused region;
2. the intermediate does not escape to backing, publication, profile, or a
   separately ordered collective;
3. producer and consumer share a compatible tile/domain and deterministic
   arithmetic order;
4. the combined register, shared-memory, descriptor, and source/template
   bounds pass backend admission;
5. removing the intermediate leaves the same external PageUse and dirty
   projection.

An admitted fused region has one Pipeline/source identity, one native node,
and no VSM resource or physical class for the removed intermediate. A rejected
fusion retains the ordinary typed Graph edge; it is not emulated by allocating
then hiding an intermediate counter.

The implemented bounded pure Map prefix follows this law before one
collective and removes its internal VSM values. Actual Metal and Vulkan
DeviceVsm Q=5/9/257 contracts prove two authored canonical total U64 Maps plus
Sum retain three authored nodes but lower to two physical stages, one native
submit, and no Host epoch service/callback. They cover both wrapping Add and a
non-summary `xor -> multiply -> add` chain. A separate exact Window fusion
admits up to three independent parameter-free canonical-total
single-read/value-write U32 Map DAG stages before centered Clamp or Clip
Sum/Min/Max and up to three after it. The immediate `+3`/`*2` pair stays
specialized; the generic proof seals every exact artifact identity and retains
every ParsedIR authority. Q=5/9/257 actuals prove both the single-stage branch
expressions `(x+1)+(2*x)` / `(x+5)+(3*x)`, a physical five-stage chain, and
the maximum seven-stage
`Map -> Map -> Map -> Window -> Map -> Map -> Map` chain. The latter two
deliberately use individually valid expressions whose adjacent composition
exceeds the static expression envelope, so the one payload dispatch cannot be
explained by ordinary Map folding. No Map intermediate is materialized. A
prefix Map around Clip is DeviceVsm-only because the service-aware
identity-fill fallback would not preserve the mapped boundary semantics.
Parameterized/non-U32 Window Maps, more than three Map stages on either side,
fusion across another collective, and stateful, profile, or publication
boundaries remain partial.

## Acceptance

- Adjacent Window epochs must read each canonical backing byte at most once
  while its Host page remains resident; delayed consumers reuse the same page
  identity and exact slot generation.
- A whole-run-resident DeviceVsm Window must return the proof-sealed `Q-1`
  canonical boundary transitions and exact aggregate footprint checksum from
  the GPU result cell. Host-derived success counters are not evidence.
- A halo page pinned by a nearer Graph consumer cannot be evicted for a farther
  forecast. Once the closed pin ends, next-use and physical-frame tie break
  choose the victim deterministically.
- Tail pages authenticate their exact visible byte slice; fill bytes never
  become backing reads or dirty publication.
- Fused and unfused execution produce identical bits. The fused contract must
  also prove the removed resource has no arena bytes, transfer bytes, cache
  rows, or native dispatch.
- Performance evidence reports backing bytes, page hits, promotion bytes,
  shared-memory bytes, dispatch count, and ready-edge stall. Geometry alone
  does not prove throughput.
# Resource-level page permutation

External input page remaps are part of the sealed graph footprint rather than
a second physical allocation policy. The footprint reserves the existing `K`
frame rows and stores at most `TiledGraphPortCapacity * PipelineLeafCapacity`
fixed remap entries. Each nonempty resource template is a full-page bijection;
source identity is resolved before frame matching, so next-use, pin, dirty, and
alias accounting still see one authoritative source page. The map changes
identity and binding order, not page bytes or the Pool footprint. Empty maps use
the ordinary ordinal footprint and identity.

The active tail is checked with `q = min(K, P-F)` before Authority mutation.
Partial-byte or multi-source Assemble footprints are intentionally not
implemented here.
