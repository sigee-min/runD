# Accel RangeAggregate Planning Contract

`RangeAggregate` is the source-private algorithm-selection authority for
associative one-dimensional range operations. It turns operation algebra,
window shape, and backend capabilities into one immutable physical plan. It
does not own graph semantics, buffers, an arena, or physical memory placement.

## Authority

Implementation authority:

- `/node/src/accel/range_aggregate/model.hpp`
- `/node/src/accel/range_aggregate/plan.hpp`

Verification authority:

- `/node/tests/contract/accel/kernel/range_aggregate.cpp`

The entry point is

```cpp
PlanRangeAggregate(const RangeAggregateShape &,
                   const RangeAggregateCapabilities &) noexcept
```

and is the only candidate-selection function. Callers project an already
validated primitive into `RangeAggregateTraits` and `RangeAggregateShape`,
then freeze the returned plan. Backend callers do not apply a second radius or
device-name threshold.

## Algebra, shape, and capabilities

The exclusive traits factory recognizes Sum, Minimum, and Maximum in the
signed, unsigned, or fixed integer domains and requires an explicit arithmetic
law. Modulo-width Sum is associative, has an identity, is commutative, and is
invertible. Saturating Sum is conservatively non-associative and non-invertible,
so it cannot acquire a prefix candidate. Minimum and Maximum use an order-only
law; both are associative, idempotent, and ordered but are not invertible.

The current window shape is one-dimensional Clamp with

```text
N >= 1
1 <= r <= N
E in {4, 8} bytes
payload = N E representable in u64
```

Capabilities are constructed as exactly one of unavailable, CPU, or GPU. A
GPU capability names its Metal or Vulkan source class, the supported width set
within `{64, 128, 256}`, maximum threads per workgroup, shared-memory limit,
shared-memory occupancy budget, maximum first-dimension group count, and
supported algorithm variants. Unknown source classes, widths, operations,
domains, or candidate bits fail closed.

## Legal candidates

Direct is the CPU route and the meaning-preserving GPU fallback. For a GPU
width `W`, it is legal only when the width and
`ceil(N / W)` groups are supported. Its exact modeled work is

```text
global reads  = N(2r + 1)E
global writes = NE
combine ops   = 2rN
dispatches    = 1
```

SharedHalo is considered separately at each supported width. With device
shared limit `L` and occupancy budget `q`, its frozen radius capacity is

```text
S = floor(floor(L / q) / E)
C = min(W, floor((S - W) / 2)) when q > 0 and S > W
C = 0 otherwise
```

It is legal only when `r <= C`, `(W + 2C)E <= floor(L/q)`, and the dispatch
group count fits. The allocation uses the frozen capacity `C`; the traffic
uses runtime radius `r`. For `G = ceil(N/W)` and final active tail
`T = N - (G-1)W`, exact distinct-union input reads are

```text
R = N                                      when G = 1
R = N + (2G - 3)r + min(r, T)             when G > 1
global reads = RE
combine ops  = 2rN
```

PrefixDifference is legal only for an associative invertible operation,
currently modulo-width Sum.
Each level performs a work-efficient, padded width-`W` local prefix, emits one
summary per group, recursively scans the summaries, and fixes lower levels in
reverse order. The final window stage applies prefix difference and explicit
Clamp endpoint corrections. Every hierarchy dispatch must fit, `WE` local
shared bytes must fit the occupancy budget, and all prefix and summary byte
requirements must be representable. If `n[0] = N` and
`n[j+1] = ceil(n[j]/W)`, the hierarchy is a decreasing geometric series for
`W >= 64`; its scan and fix-up work is `O(N)`. Endpoint work is at most `2r`,
and `r <= N`, so total work is `O(N)` independent of a multiplicative radius
factor.

BlockPrefixSuffix is legal only for ordered idempotent Minimum or Maximum. It
forms the conceptual clamped sequence

```text
M = N + 2r
K = 2r + 1
```

and partitions it into `ceil(M/K)` blocks. Each block writes forward prefix
and backward suffix values; each output combines the suffix at its left edge
with the prefix at its right edge. Since `r <= N`, `M <= 3N`; preparation plus
output work is therefore `O(N)`. Checked `M`, byte capacities, and both
dispatch topologies must fit.

## Cost and deterministic selection

All candidate construction uses checked u128 arithmetic for modeled byte and
operation axes and checked u64 arithmetic for retained storage and dispatch
axes. The exact cost record contains global read bytes, global write bytes,
combine operations, inverse operations, endpoint scale operations, shared
bytes, scratch bytes, dispatch count, and launched lanes.

A candidate dominates another only when every cost axis is no greater and at
least one axis is smaller. The planner first removes every dominated
candidate. If multiple Pareto candidates remain, it uses this documented
lexicographic order:

1. global read bytes;
2. global write bytes;
3. combine operations;
4. inverse operations;
5. endpoint scale operations;
6. scratch bytes;
7. dispatch count;
8. shared bytes;
9. launched lanes;
10. candidate disposition, then width, then shared capacity.

This is a deterministic integer model. It consumes neither timing,
calibration, GPU model names, nor a fixed radius cutoff, and it does not claim
universal wall-clock optimality.

## Stages, temporary requirements, and ownership

The selected plan exposes its exact ordered stages and typed temporary
requirements. A temporary records role, bytes, alignment, and inclusive first
and last live stage. PrefixDifference exposes prefix values and per-level block
summaries. BlockPrefixSuffix exposes distinct forward and backward value
roles. Direct and SharedHalo have no global temporary.

The plan stores a compact immutable derivation rather than owning a vector or
allocating. These are placement-free requirements:

```text
RangeAggregate planner
  -> variant + stage graph + typed temporary requirements
  -> existing Pipeline memory planner
  -> arena offsets, lifetime aliasing, alignment, and accounting
```

Dispatch-local shared bytes remain pipeline resource metadata and never enter
the global arena. The existing Pipeline memory planner remains the sole
physical placement and reuse authority.

## Identity

Source identity contains the backend source class, algorithm variant, width,
shared capacity, operation, numeric domain, arithmetic law, boundary, and
element width.
`N` and `r` are normalized out when they do not shape that source variant.

Execution identity starts from source identity and adds `N`, `r`, exact stage
topology, temporary roles and lifetimes, and the complete cost vector. Thus
runtime-equivalent source variants can reuse a pipeline while different
dispatch or storage plans remain distinct.

Rejected plans carry a stable positive failure boundary for invalid shape,
unavailable or invalid capability, no legal candidate, or checked cost
overflow. A rejected plan is not an implemented execution route.
