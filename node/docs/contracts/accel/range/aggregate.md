# Accel Range Planning Contract

Range is the source-private algorithm-selection authority for
associative one-dimensional range operations. It turns operation algebra,
window shape, and backend capabilities into one immutable physical plan. It
does not own graph semantics, buffers, an arena, or physical memory placement.

## Authority

Implementation authority:

- `/node/src/accel/range_aggregate/model/traits.hpp` owns the Range algebra,
  identity, shared vocabulary, and width/support constants.
- `/node/src/accel/range_aggregate/model/shape.hpp` owns affine shape
  validation and the frozen shape value.
- `/node/src/accel/range_aggregate/model/capability.hpp` owns CPU/GPU
  capability construction and validation.
- `/node/src/accel/range_aggregate/model/candidate.hpp` owns candidate, cost,
  stage, temporary, and fixed planner-capacity values.
- `/node/src/accel/range_aggregate/model/prefix.hpp` owns prefix execution
  stage derivation, its builder, and prefix factories.
- `/node/src/accel/range_aggregate/model/plan.hpp` owns the immutable selected
  or rejected Range plan and its stage/temporary projection.
- `/node/src/accel/range_aggregate/plan/arithmetic.hpp` owns checked u128/u64
  planner arithmetic and group-count formulas.
- `/node/src/accel/range_aggregate/plan/evaluation.hpp` owns the candidate
  evaluation record and bounded stage/temporary append operations.
- `/node/src/accel/range_aggregate/plan/traffic.hpp` owns affine boundary
  traffic projection used by direct and prefix candidates.
- `/node/src/accel/range_aggregate/plan/budget.hpp` owns shared-memory budget
  and radius-capacity formulas.
- `/node/src/accel/range_aggregate/plan/direct.hpp` owns Direct candidate
  construction.
- `/node/src/accel/range_aggregate/plan/shared.hpp` owns SharedHalo candidate
  construction.
- `/node/src/accel/range_aggregate/plan/tiled.hpp` owns TiledDifference
  candidate construction.
- `/node/src/accel/range_aggregate/plan/prefix.hpp` owns PrefixDifference
  candidate construction.
- `/node/src/accel/range_aggregate/plan/block.hpp` owns BlockPrefixSuffix
  candidate construction.
- `/node/src/accel/range_aggregate/plan/order.hpp` owns Pareto dominance and
  deterministic lexicographic ordering.
- `/node/src/accel/range_aggregate/plan/identity.hpp` owns source and
  execution identity construction.
- `/node/src/accel/range_aggregate/plan.hpp` owns final legality checks,
  candidate enumeration, Pareto filtering, and selected-plan orchestration;
  it owns no algorithm-specific candidate arithmetic.
- `/node/src/accel/range_aggregate/execution/model.hpp` owns the fixed host
  parameter/dispatch ABI (`RangeParams`, `RangeDispatch`, and
  `RangeIndirect`).
- `/node/src/accel/range_aggregate/execution/run.hpp` owns resident-count stage
  projection and indirect-row construction (`RangeRun`).
- `/node/src/accel/range_aggregate/execution/control.hpp` owns resident control
  buffer byte layout (`RangeControlLayout`).
- `/node/src/accel/range_aggregate/execution/scratch.hpp` owns immutable
  temporary-role slot and pair values (`RangeTempSlot`, `RangeScratch`).
- `/node/src/accel/range_aggregate/execution/projection.hpp` owns the complete
  backend-neutral `RangeExec` projection and its three small free projections.
- `/node/src/accel/metal/range/`
- `/node/src/accel/vulkan/range/`
- `/node/src/accel/scan/prefix.hpp` for the native Scan projection

Metal keeps one owner per native Range responsibility:
`capability.mm` projects device limits, `resources/execute.mm` owns temporary
and resident-count control lifetime, `pipeline/execute.mm` owns cache lookup,
compilation, and publication, `prepare.mm` owns admission and assembly, and
`encode.mm` owns command encoding. `source.cpp` and `source/` own the complete
MSL recipe.

Verification authority:

- `/node/tests/contract/accel/kernel/range/aggregate.cpp`
- `/node/tests/contract/accel/kernel/range/`

The entry point is

```cpp fragment
PlanRange(const RangeShape &, const RangeCaps &) noexcept
```

and is the only candidate-selection function. Callers project an already
validated primitive into `RangeTraits` and `RangeShape`,
then freeze the returned plan. Backend callers do not apply a second radius or
device-name threshold.

A shape is either descriptor-counted or resident-counted. A resident-counted
shape freezes the authored capacity `M` for legality, candidate selection, and
temporary reservation, while a U32 or U64 device scalar supplies the active
count `n` for one execution. The scalar is data, not a second planning API:
the selected candidate, width, source variant, and maximum stage topology stay
frozen. `0 <= n <= M` is required. `n=0` emits no data work, and `n>M` records
the canonical bounded-count failure before any output publication.

`RangeExec` is the one backend-neutral physical projection of that frozen
plan. It derives source identity, workgroup shape, stage parameters,
descriptor demand, shared allocation, dispatch topology, and typed temporary
bindings. Metal and Vulkan consume that value for source materialization,
pipeline/cache lookup, stage preparation, and dispatch. A primitive adapter
supplies only authenticated input/output bindings and maps its public
validation and result boundary; `MetalRangeBinds` and `VulkanRangeBinds` make
that handoff an exclusive two-binding value.

## Algebra, shape, and capabilities

The exclusive traits factory recognizes Sum, Minimum, and Maximum in the
signed, unsigned, or fixed integer domains and requires an explicit arithmetic
law. Modulo-width Sum is associative, has an identity, is commutative, and is
invertible. Saturating Sum is conservatively non-associative and non-invertible,
so it cannot acquire a prefix candidate. Direct and centered SharedHalo visit
logical window positions from left to right. Signed and unsigned integers wrap
at lane width; Fixed applies its declared overflow law after every addition.
Minimum and Maximum use an
order-only law; both are associative, idempotent, and ordered but are not
invertible.

The common one-dimensional affine shape is

```text
N = input count, Q = output count
K = logical window size, S = output stride, P = left padding
output j begins at jS-P and contains K logical positions

N,Q,K,S >= 1
P < K
(Q-1)S is representable in u64
(Q-1)S-P < N
E in {4, 8} bytes
NE and QE representable in u64
```

`P<K` proves that output zero intersects the input. The checked last-start
inequality proves that every later output also intersects, including a padded
output whose anchor is beyond `N`. Range admits arbitrary representable `K`;
semantic adapters own any tighter bound. The centered Stencil projection is
exactly `Q=N`, `K=2r+1`, `S=1`, `P=r`, Clamp. A pooling adapter may impose its
public `K<=N` law before projection without creating a second Range validator.

Clamp repeats the nearest endpoint outside `[0,N)`. Clip excludes an
out-of-range logical position, equivalently combining the operation identity.
Both are source-shaping boundary laws. The checked conceptual span used by
block preparation is `T=(Q-1)S+K`.

Capabilities are constructed as exactly one of unavailable, CPU, or GPU.
`cpu_reference()` admits Direct only for the standalone meaning oracle;
`cpu()` admits Direct, PrefixDifference, and BlockPrefixSuffix with a frozen
logical block width of 64. A
GPU capability names its Metal or Vulkan source class, the supported width set
within `{64, 128, 256}`, maximum threads per workgroup, shared-memory limit,
shared-memory occupancy budget, maximum first-dimension group count, and
maximum addressable storage element count, maximum bytes in one storage
binding, plus supported algorithm variants.
Metal and CPU retain the u64 count domain; the current Vulkan source variant
freezes `UINT32_MAX` as its storage element-count upper bound because every
storage-array subscript is u32. Unknown source classes, widths, operations,
domains, zero storage bounds, or candidate bits fail closed.

## Legal candidates

Every candidate requires `N`, `Q`, and each emitted stage element count to fit
the frozen capability's storage element-count bound. A candidate that emits a
global temporary additionally requires every individual temporary to fit the
frozen single-storage-binding byte bound. These legality checks occur inside
`PlanRange`; pipeline acquisition retains only defensive rechecks and never
becomes a second candidate authority. Thus an oversized PrefixDifference or
BlockPrefixSuffix temporary removes that candidate while Direct remains
eligible when its authenticated input and output bindings are legal.

Direct is the meaning-preserving fallback. For output `j`, let `Vj` be `K`
under Clamp and the number of in-range logical positions under Clip, and let
`V=sum(Vj)`. For a GPU width `W`, it is legal only when the width and
`ceil(Q / W)` groups are supported. Its exact modeled work is

```text
global reads  = VE
global writes = QE
combine ops   = V-Q
dispatches    = 1
```

Clamp therefore has `V=QK`; Clip subtracts the exact left and right excluded
arithmetic series. Direct is `Theta(QK)` and remains the semantic fallback
when no linear candidate is legal.

SharedHalo is considered separately at each supported width. With device
shared limit `L` and occupancy budget `q`, its frozen radius capacity is

```text
S = floor(floor(L / q) / E)
C = min(W, floor((S - W) / 2)) when q > 0 and S > W
C = 0 otherwise
```

SharedHalo is legal only for the centered Clamp projection (`Q=N`, `S=1`,
`K=2P+1`) and when `P <= C`, `(W + 2C)E <= floor(L/q)`, and the dispatch
group count fits. The allocation uses the frozen capacity `C`; the traffic
uses runtime padding `P`. For `G = ceil(N/W)` and final active tail
`T = N - (G-1)W`, exact distinct-union input reads are

```text
R = N                                      when G = 1
R = N + (2G - 3)P + min(P, T)             when G > 1
global reads = RE
combine ops  = (K-1)Q
```

TiledDifference is a one-dispatch, zero-global-scratch candidate for centered
Clamp modulo-width Sum. With physical width `W`, each lane owns 16 consecutive
outputs and a group owns `L=16W` outputs. For group base `b`, it first reduces
the in-range portion of window `b`, adding the exact scaled Clamp endpoints.
It then computes `d[i]=x[clamp(i+r)]-x[clamp(i-r-1)]` for `i>b` and `d[b]=0`.
An ordered prefix of these differences gives `y[i]=y[b]+sum(d[b..i])` modulo
lane width. No global carry or cross-group communication is needed. Each
output has one writer; input is immutable. This is not applicable to Fixed
saturating Sum or Min/Max.

The initial-window spans are the affine shape with `Q=ceil(N/L)`, stride `L`,
and the original padding/window. If that shape has `V` in-range samples and
`C` clamped endpoint corrections, and `G=ceil(N/L)`, modeled reads are
`[V+C+2(N-G)]E`; writes are `NE`; shared memory is `2WE`; scratch is zero;
and dispatch count is one. Work is `O(N+G min(N,K))`, not uniformly `O(N)` for
arbitrary growing radius. The existing Pareto/lexicographic cost authority
selects it only when its complete cost ranks ahead of the other candidates;
there is no product-shape, device-name, or radius threshold. Very large
radii retain PrefixDifference when its traffic is lower. The source geometry
constant lives in `model/traits.hpp`, and `plan/tiled.hpp` owns its cost and
legality. Metal uses SIMD sums and two threadgroup barriers for both stored widths;
Vulkan uses a shared-memory tree. Inactive lanes contribute
zero and all participating lanes cross the same barriers.

PrefixDifference is legal only for an associative invertible operation,
currently modulo-width Sum.
Each level performs a work-efficient, padded width-`W` local prefix, emits one
summary per group, recursively scans the summaries, and fixes lower levels in
reverse order above level zero. The per-element level-zero fix-up is fused
into the query: for endpoint `x`, the inclusive global prefix is
`local_prefix[x] + (x/W == 0 ? 0 : summary0[x/W-1])` modulo lane width.
The query retains summary zero through its final stage and reads at most two
additional summary values per output, instead of materializing corrected
prefix values for every input element. A one-block active input reads no
summary, including when its frozen resident capacity reserved many blocks.
Native Scan retains its fully materialized prefix contract. Metal's 32-bit modulo prefix
uses SIMD inclusive sums and SIMD totals, then one ordered scan of SIMD-group
totals. The SIMD width comes from the shader runtime attributes. All lanes,
including inactive zero-filled tail lanes, cross both threadgroup barriers;
there is no assumed 32-lane device width. Metal
64-bit prefixes use the same two-barrier hierarchy with exact split-word SIMD
arithmetic; Vulkan keeps the work-efficient shared-memory tree. The planner's combine count remains
an algebraic scan work model, not a hardware instruction counter.
The shared Metal wide-prefix recipe lives in `metal/simd/source.hpp` and is
also consumed by native Scan. Let `l_i` and `h_i` be an unsigned value's low
and high 32-bit words. Compute `p_i = sum(l_j, j<=i) mod 2^32` and
`c_i = [p_i < l_i]`. Since `p_i = p_(i-1) + l_i mod 2^32`, `c_i` is exactly
that sequential addition's carry. The high word is
`sum(h_j, j<=i) + sum(c_j, j<=i) mod 2^32`. This constructs the exact
modulo-64 inclusive prefix using only 32-bit SIMD prefix operations, including
low-word carry and full-width wrap. Group totals use two 32-bit shuffles from
the last participating lane. Width and partial subgroup boundaries come from
runtime attributes; inactive data lanes contribute zero. Metal Range source
identity version 6 covers PrefixDifference and TiledDifference; CPU and Vulkan
source versions are unchanged.

The [Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
leaves the mapping from threadgroup threads to SIMD groups implementation
defined. Kernels therefore assign logical work index
`simd_group * simd_width + simd_prefix_exclusive_sum(1u)`, using actual SIMD
attributes. This follows active-lane order rather than assuming a physical
thread index mapping. A partial subgroup uses `simd_max(physical_lane)` as its
last active lane for total shuffles. The maximum-index SIMD group is the only
possibly partial group, so these logical indices densely cover the workgroup.
The same construction is used by native Scan.

The final stage derives the in-range endpoints of
`[jS-P,jS-P+K)`, applies prefix difference, and under Clamp adds at most one
scaled left endpoint and one scaled right endpoint per output. Clip adds no
endpoint correction. Every hierarchy dispatch must fit, `WE` local
shared bytes must fit the occupancy budget, and all prefix and summary byte
requirements must be representable. If `n[0] = N` and
`n[j+1] = ceil(n[j]/W)`, the hierarchy is a decreasing geometric series for
`W >= 64`; its scan and fix-up work is `O(N)`. Query and endpoint work is
`O(Q)`, so total work is `O(N+Q)` independent of a multiplicative window-size
factor. Prefix values occupy exactly `NE` bytes. Both input/output counts and
every hierarchy stage element count must fit the frozen storage element bound.

The CPU plan uses the same candidate authority but an exact sequential
two-stage graph: one `PrefixSequential` pass over `N`, then one query pass over
`Q`. It retains only the `NE` prefix temporary, has no shared bytes or block
summaries, and models exactly `N+Q` launched logical lanes.

BlockPrefixSuffix is legal only for ordered idempotent Minimum or Maximum. It
forms the conceptual sequence

```text
T = (Q-1)S + K
```

Conceptual position `x` maps to input index `x-P`; Clamp materializes an
endpoint and Clip materializes the Min/Max identity outside the input. The
candidate partitions the sequence into `ceil(T/K)` blocks. Each block writes
forward prefix and backward suffix values; each output combines the suffix at
its left edge `jS` with the prefix at `jS+K-1`. General affine preparation plus
output is `O(T+Q)`, with exact scratch `2TE` for CPU/Vulkan and `TE` for Metal. It reduces to `O(N+Q)` under
adapter bounds such as Pooling's `K<=N` or centered Stencil's `Q=N`, `S=1`,
and `K<=2N+1`. Checked `T`, byte capacities, and both dispatch topologies must
fit. CPU and Vulkan preparation read exactly `2T` input elements under Clamp.
Metal reads its preparation span and its query-block span once each, as described below. Under Clip,
identity padding performs no input access, so replace `T` by `min(N,T-P)`
when counting source-level input reads. CPU/Vulkan query reads two prepared values per output; Metal reads one. A backend removes this candidate when `T` exceeds its frozen storage
element bound; for the current Vulkan source this is `UINT32_MAX` because its
scratch indices are u32. Another legal candidate may remain available.
Metal preparation assigns one workgroup to one conceptual block, replacing
the old one-thread-per-block dependent global recurrence. The first SIMD group
in each workgroup scans the block in contiguous SIMD-width tiles, broadcasts
the tile extremum as the next carry and writes forward prefixes. Other SIMD
groups return before any payload access; there is no workgroup barrier or
threadgroup memory. The frozen dispatch width remains part of scheduling and
launched-lane evidence; active lanes are `min(W,s)` for runtime SIMD width `s`.

The second dispatch fuses suffix construction with the final query. It owns
`R=floor((Q-1)S/K)+1` blocks containing query starts. Each owns exactly `K`
conceptual elements and scans backwards with the same register carry.
When a scanned position `x` equals `jS` for `j<Q`, it writes
`combine(suffix(x), forward[x+K-1])` directly to output. It never materializes a
global suffix array. Preparation prefix writes are visible before the query dispatch; the Window
binding contract keeps input and output byte spans distinct.
The generic two-slot encoder aliases the sole ForwardValues temporary in both
slots, as it does for a one-block prefix; this creates one allocation and the
shader consumes only buffer3. There is no BackwardValues temporary or argument
in Metal. CPU/Vulkan retain their separate two-array contracts.

Let `a=min(W,s)`, `B=ceil(T/K)`, and `U=RK<=T`. There are zero workgroup
barriers, zero shared bytes, and no inter-workgroup waits inside either
dispatch. Dependency depth per block is `O(ceil(K/a) log(a))` rather than the
old `O(K)` dependent global recurrence. Clamp input reads are `T+U`, scratch
reads `Q`, scratch writes `T`, and output writes `Q`. Clip input reads are
`min(N,T-P)+max(0,min(N,U-P))`. Global scratch is `TE`, half of the old Metal
contract, while dispatch count stays two. The conservative algebraic combine
bound over conceptual elements is `2T(log2(W)+1)+Q`; it excludes operations
on identity-only inactive tile lanes, like the Prefix algebraic work model.
These are source-level work/traffic models, not DRAM
transactions or exact instruction counts. Source identity is revised for
this topology. Dense (`S=1`) queries emit direct output indices and no integer
quotient; strided queries retain exact `x/S` divisibility checks. The dense
bit is authenticated in Metal Block source identity; different non-unit
strides reuse one parameterized source while execution identity retains the
complete affine shape.

Shuffle up/down distances are uniform within each SIMD group. Tile extrema
use `simd_shuffle` with a uniform source lane; varying up/down deltas are
invalid under the [Metal language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
section 6.10.2. Exact 64-bit shuffles split and recombine two unsigned 32-bit
words; comparisons retain the original signed or unsigned type. No floating
arithmetic, ordering-dependent reduction, or cross-workgroup synchronization
is introduced. Associativity and idempotence establish the same affine
Min/Max result for Clamp and Clip, including a partial final block.

`model/block.hpp` owns Metal/Vulkan preparation and query block groups projection.
Candidate admission, frozen `RangePlan` stages and resident `RangeRun` consume
that owner; Metal's control shader emits the matching one-group-per-block
geometry. The shader source boundary is declaration-only `metal/range/source.hpp`.
Shader recipes include the backend-neutral execution model directly and never
import adapter, resident-buffer, pipeline-cache or native resource-state definitions.

The CPU projection records two sequential stages with one logical group and
width zero: preparation performs `2T` forward/backward iterations and the
query performs `Q`, so its launched-work evidence is exactly `2T+Q`.

## Cost and deterministic selection

All candidate construction uses checked u128 arithmetic for modeled byte and
operation axes and checked u64 arithmetic for retained storage and dispatch
axes. The exact cost record contains global read bytes, global write bytes,
scheduled workgroups, combine operations, inverse operations, endpoint scale operations, shared
bytes, scratch bytes, dispatch count, and launched lanes.

A candidate dominates another only when every cost axis is no greater and at
least one axis is smaller. The planner first removes every dominated
candidate. If multiple Pareto candidates remain, it uses this documented
lexicographic order:

1. global read bytes;
2. global write bytes;
3. scheduled workgroups, summed over all stages;
4. combine operations;
5. inverse operations;
6. endpoint scale operations;
7. scratch bytes;
8. dispatch count;
9. shared bytes;
10. launched lanes;
11. candidate disposition, then width, then shared capacity.

Workgroup count exposes scheduling work that lane count alone hides: a query
with width 64 schedules four times as many groups as width 256 for the same
large divisible output. It is accumulated with checked arithmetic by the sole
stage append owner, participates in Pareto dominance and execution identity,
and applies to every candidate without a workload-size threshold. The ordering
prefers fewer scheduled groups at equal modeled traffic; it does not assert a
universal latency model.

This deterministic integer model consumes only operation algebra, exact shape,
and frozen capability fields. Candidate legality and ordering are therefore
reproducible across runs; the model is structural evidence and does not claim
universal wall-clock optimality.

## Stages, temporary requirements, and ownership

The selected plan exposes its exact ordered stages and typed temporary
requirements. A temporary records role, bytes, alignment, and inclusive first
and last live stage. PrefixDifference exposes prefix values and per-level block
summaries. CPU/Vulkan BlockPrefixSuffix exposes distinct forward and backward value
roles. Direct and SharedHalo have no global temporary.

For a resident-counted GPU plan, backend preparation adds one physical control
pass before the frozen algorithm stages. It is not an algorithm stage and does
not change `RangePlan::stage_count()`. The pass reads `n`, derives every
data-stage element and group count with checked integer formulas, and writes
one 64-byte `RangeParams` row plus one 32-byte dispatch/evidence row per frozen
data stage. Prefix levels above the hierarchy required by `n` and every data
stage at `n=0` receive all-zero dispatch rows.

For Pipeline execution, the existing Pipeline scratch planner remains the sole
physical placement owner of arithmetic temporaries. Standalone preparation
retains backend-native arithmetic buffers sized from the same frozen temporary
requirements; it introduces neither another placement planner nor another
arena. Control parameters, dispatch rows, and status are backend-native
prepared buffers because they have command-processor usage and lifetime, not
arithmetic scratch lifetime. Vulkan creates the dispatch backing with storage
and indirect-command usage, then places a compute-write to
compute/indirect-read barrier before the data stages. Standalone Metal likewise
uses indirect data dispatches after a buffer barrier. Metal Pipeline-private
capture records the frozen capacity grids because Metal indirect dispatch is
not an ICB command in this contract; the runtime parameter rows make inactive
lanes return before any payload access. Ordinary scratch barriers preserve the
algorithm stage order. No Range-owned arena exists.

For a centered resident Window with capacity `M`, active count `n`, radius
`r`, and width `W`, the active geometry is

```text
Q(n) = n
K = 2r + 1, S = 1, P = r
Prefix level 0: n0 = n
Prefix level j+1: nj+1 = ceil(nj / W), stopping after the first group
BlockPrefixSuffix span: T(n) = n == 0 ? 0 : n - 1 + K
```

All active byte and work terms substitute `n`, `Q(n)`, and `T(n)` into the
selected candidate's formulas. Reserved global scratch instead uses the same
formulas at `M`; therefore every active placement is a prefix of an already
retained allocation. Warm prepared execution allocates nothing. Vulkan and
standalone Metal launch the active topology, while Metal Pipeline-private
capture launches frozen capacity grids and performs active arithmetic only for
`n`. PrefixDifference active arithmetic remains `O(n)`. BlockPrefixSuffix
active arithmetic is `O(n+r)` and is `O(n)` exactly when the adapter constrains
`r=O(n)`; under the centered resident Window bounds `n<=M` and `r<=M`, its
frozen-capacity work and storage envelope is `O(M)`. The descriptor-counted
path has no physical control pass or dispatch buffer.

Vulkan retains one data PSO for all `S` frozen Range data-stage slots; a
resident-counted route additionally retains one control PSO. Stage slots and
their descriptor sets remain `S` or `S+1` because each dispatch needs its own
immutable binding tuple, while the unique native pipeline dependency count is
`U=1` or `U=2`. Cold native-object accounting therefore uses
`3U + D + U`—pipeline, pipeline layout, and descriptor-set layout per unique
PSO, `D` descriptor sets, and one descriptor pool per unique PSO—rather than
multiplying native PSOs or pools by aliased stage slots. Immutable preparation
acquires the data PSO once and borrows that exact pointer in every data slot.
Standalone preparation also acquires it once before materializing the stage
slots. The source recipe and cache hash/byte comparison execute once per
unique data PSO.

Backend shader source is privately split by durable responsibility. Metal and
Vulkan each own separate `source/{algebra,direct,shared,tiled,prefix,block}.hpp`
leaves, while `source/control.hpp` owns only the resident-count control pass.
Metal `source/build.hpp` is the short source-assembly owner; Vulkan
`range/source.cpp` performs the equivalent assembly. These owners emit the
exact source byte stream consumed by cache identity. Common Range source owns
no Stencil descriptor, radius policy, hash, or shader variant.

`RangePrefixExec` is the common source-private stage derivation for
associative prefix work. Its hierarchical form derives PrefixDifference's
recursive block, summary, and reverse-fixup stages. Its flat block-total form
derives native Scan's block, total-prefix, and offset stages from a frozen
`ScanPlan`. Scan is the prefix-only adapter: its native inclusive/exclusive
source and result law consume that shared flat derivation, while `RangeExec`
owns the window-family source and dispatch projection. Each primitive retains
its semantic result, overflow, and public-output authority.

The plan stores a compact immutable derivation rather than owning a vector or
allocating. These are placement-free requirements:

```text
Range planner
  -> variant + stage graph + typed temporary requirements
  -> existing Pipeline memory planner
  -> arena offsets, lifetime aliasing, alignment, and accounting
```

Dispatch-local shared bytes remain pipeline resource metadata and never enter
the global arena. The existing Pipeline memory planner remains the sole
physical placement and reuse authority.

Stencil and Window are the semantic adapters. Both project into the same
`RangeShape`, use the same physical `RangeBinds`, reject resident overlap, and
receive selected source, cache identity, temporary binding, and stage dispatch
from `RangeExec`. Stencil owns its centered Clamp descriptor/hash. Exact
rolling aggregates use the Window adapter with `Q=N`, `K=2r+1`, `S=1`, and
`P=r`. Pool validates `K<=N` and derives `Q=1+floor((N-K)/S)` for Drop or
`Q=1+floor((N-1)/S)` for Keep before entering the same Window adapter. Window
owns the resulting affine descriptor/hash and Clamp/Clip boundary; neither
public adapter owns physical scratch or backend source code.

## Identity

Source identity contains the backend source class, algorithm variant, width,
shared capacity, operation, numeric domain, arithmetic law, boundary, and
element width, plus the descriptor/U32/U64 count class and the dense-query
bit for Metal Block.
`N`, `Q`, `K`, `S`, and `P` are runtime parameters and are normalized out when
they do not shape that source variant.

Execution identity starts from source identity and adds `N`, `Q`, `K`, `S`,
`P`, exact stage topology, temporary roles and lifetimes, and the complete cost
vector. Thus runtime-equivalent source variants can reuse a pipeline while
different dispatch or storage plans remain distinct.

The Metal named-pipeline key serializes the complete source-shaping tuple
directly—candidate, width, shared capacity, operation, domain, arithmetic law,
boundary, element width, descriptor/U32/U64 count class, and the Metal Block
dense-query bit. Exact tuple text
is the lookup authority; the compact hash is only its index. One such
typed source variant accepts its affine `N`, `Q`, `K`, `S`, and `P` through
runtime parameters.

Rejected plans carry a stable positive failure boundary for invalid shape,
unavailable or invalid capability, no legal candidate, or checked cost
overflow. A rejected plan is not an implemented execution route.

The current-source focused diagnostic
`tools/measure/compute/run --collective <cpu|metal|vulkan>` measures a prepared
U32 Sum/Min/Max Window with `r=1024` at `N=4096` and `N=262144`. Its output
hash, warm-allocation counters, dispatch evidence, and resident wall time are
recorded together. Those observations measure the selected implementation on
the recorded host; the integer work and storage bounds above remain the
portable evidence.

## Planner entry ownership

The compiled `plan.cpp::PlanRange` owns runtime planning and delegates to the
single constexpr `plan/build.hpp::BuildRangePlan` implementation. Only that
production translation unit and Range static contracts include the candidate
implementation closure. The runtime declaration remains in `plan.hpp`; frozen
models, projection and backend recipes do not import candidate evaluation.
