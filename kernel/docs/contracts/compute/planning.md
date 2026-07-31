# Compute Planning And Graph Contract

## Planning

`ComputeMap`, `ComputeCaps`, `ComputeLimit`, `ComputeDispatchPlan`, and `ComputePlan` are the
kernel planning surface. `PlanComputeDispatch(...)` and `PlanCompute(...)` are pure and
deterministic: they consume only `TilePhaseDescription`, checked `ComputeMap`,
frozen `ComputeCaps`, and caller-provided `ComputeLimit`. Kernel planning and lowering
must never call node, OS, Metal, Vulkan, PMU, clock, affinity, driver, or
hardware discovery APIs.
Kernel planning also must not inspect runtime timings, runtime stats,
pipeline-cache state, buffer-reuse counters, measured throughput, or
host/device transfer measurements. Those records are node-owned evidence and
cannot become planner inputs.

`ComputeCaps::storage_alignment` is an immutable adapter capability for
node-owned resident suballocation. It is a nonzero power-of-two byte alignment
and does not enter `PlanCompute`, graph identity, operation order, numeric
policy, or lowering source. Node may consume it only after backend selection to
place physical Buffer subranges whose semantic shapes are already frozen.

[`kernel/core/checked.hpp`](../checked.md) is the single exact unsigned
64-bit arithmetic authority shared by Compute plans, references, binding
validation, Replay, and accelerator admission.
`checked::add(a, b)` and `checked::mul(a, b)` report whether the mathematical
sum or product fits `u64`. Output overloads publish only after success;
`checked::sub(a, b, out)` admits exactly `a >= b`, and
`checked::mul(a, b, c, out)` publishes the three-factor product only when both
intermediate products fit.
`checked::ceil(n, d)` computes `floor(n / d) + [n mod d != 0]`. A zero divisor
returns zero, the fail-closed count used by shape predicates; every successful
planner separately admits a nonzero divisor before consuming the result.
These inline `constexpr` predicates use one subtraction or division before the
represented operation. Their negation is the overflow query, so there is no
second overflow implementation. Consolidation adds no allocation, state,
workload-size branch, or arithmetic pass.

`reduce/operation.hpp` is likewise the one validity owner for `ReduceOp`.
Ordinary and segmented reduction planners consume `reduce::valid(op)` instead
of duplicating the four-value enum admission list; their shape and rejection
reasons remain separate planner responsibilities.

Each primitive `plan.hpp` also owns its complete descriptor-to-plan identity
predicate. `*PlanMatchesDesc(desc, plan)` recomputes the canonical plan from
the descriptor and compares every semantic plan field except the diagnostic
reason pointer. Runtime backends consume this predicate directly; they may add
resource identity, usage, alignment, and device-limit gates, but must not keep
a second projection of planner fields. This makes the accepted plan set exactly

```text
{ PlanPrimitive(desc) | PlanPrimitive(desc).ok }
```

instead of the larger set admitted when a backend mirror forgets a newly added
field. Replanning is allocation-free `constexpr` integer work performed only
at admission, not per element or inside a dispatch.

The planner admits a valid tile phase, declared-good caps, a nonzero operation
hash, a known Compute API, fixed scalar
authority, matching map/capability API, checked per-tile byte totals, nonzero
dispatch limits, and staging capacity for at least one tile plus parameters.
It chooses the largest admitted window

```text
S = min(caps.staging_bytes, limit.staging_bytes)
W = min(caps.max_window_tiles,
        limit.max_window_tiles,
        phase.tile_count,
        floor((S - param_bytes) / bytes_per_tile))
```

after proving `S >= param_bytes + bytes_per_tile`. Thus a large physical
dispatch bound is chunked by the actual byte budget instead of rejecting work
that fits as multiple windows. This is a pure integer calculation; allocation,
timing, cache state, and device execution cannot change `W`. A successful plan
proves only that declared work fits the frozen staging and window bounds and
records the runtime binding obligations later validated by the prepared
Compute path and backend.

Planner rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Invalid `TilePhaseDescription` | `compute_phase_invalid` |
| `ComputeCaps::ok == false` | `compute_caps_invalid` |
| Missing operation hash | `compute_op_invalid` |
| Non-fixed scalar | `compute_scalar_unsupported` |
| Unknown forged API enum | `compute_api_unsupported` |
| Map/caps API mismatch | `compute_backend_mismatch` |
| Zero input+output+metadata workset | `compute_workset_zero` |
| Workset byte sum overflow | `compute_workset_overflow` |
| Zero caps/limit/window tiles | `compute_window_zero` |
| Parameter plus one tile exceeds the common staging limit | `compute_staging_insufficient` |
| Dispatch byte arithmetic overflow | `compute_dispatch_overflow` |

A successful `ComputeDispatchPlan` records bytes per tile, staging bytes, dispatch
window tiles, dispatch count, `ok = true`, and reason `ok`. `PlanCompute(...)`
reuses that dispatch proof and records phase id, tile count, operation hash,
API, scalar, input buffer count, input/output/metadata bytes per tile,
parameter bytes, bytes per tile, staging bytes, dispatch window tiles,
dispatch count, `fixed_authoritative = true`, `ok = true`, and reason
`ok`.

## Graph Identity

`GraphBufferRef`, `GraphNode`, `Graph`, `ValidateGraph(...)`, and
`ValidateGraphIdentity(...)`
are the SDK-free kernel graph contract. Descriptors are caller-owned
pointer/count facts; they store no dynamic allocation and carry no node,
backend, OS, filesystem, clock, Metal, Vulkan, Foundation, or driver state.
Graph nodes are typed as `Map`, `Scan`, `SegmentedScan`,
`SegmentedReduce`, `Sort`, `Compact`, `Gather`, `Histogram`, `Partition`,
`Reduce`, `Scatter`, `Stencil`, `Transform`, `Matrix`, `Factor`, `Solve`, or
`Spectrum`. Reserved numeric kind slots `5` through `7` are invalid and are
not admitted node kinds. `Map`
nodes carry the checked map operation hash and no primitive fields. Collective
nodes carry primitive hash, element count, and no map operation hash; detailed
primitive planners define descriptor payloads in their own contracts.

Graph identity is kernel-owned, pure, and deterministic. Validation derives
`graph_id_hi` and `graph_id_lo` only from scalar and numeric policy, map
operation hashes, collective node kind, collective primitive descriptor hash,
collective element count, logical buffer ids, read/write roles, active
first-write initialization, and node/buffer order. `BufferInit::Preserve`
adds no identity extension; `BufferInit::Zero` adds one tagged binding fact.
The same descriptor facts produce the same id regardless of backend
runtime state, resident storage authenticity, pipeline caches, command queues,
measured timings, or adapter selection. Reordering nodes or changing logical
buffer roles changes the identity input. Collective nodes extend the identity
boundary without changing existing map graph ids.

Graph validation admits fixed scalar authority, at least one node, known node
kind, a nonzero operation hash for every `Map` node, a
zero primitive descriptor for every `Map` node, a nonzero primitive descriptor
hash plus nonzero element count for every collective node, and nonzero logical
buffer ids tagged with known read/write roles and initialization. `Zero` is
valid only on a Write binding; a Read binding always preserves its input.
Graph ids are stable contract values for the same descriptor facts. Fixed
format, rounding, overflow, approximation, scalar, numeric domain, operation
identity, resources, roles, initialization, and order form the complete
semantic identity boundary. The Zero-init domain tag is hashed only when that
semantic extension is active. The canonical two-map fixture is
`graph_id_hi = 0xf20519f7ca65f06e` and
`graph_id_lo = 0xa2881dd1b5722914`.

`ValidateGraphIdentity(...)` applies that same hashing algorithm to an admitted
semantic recipe whose root capacity is zero. It still requires at least one
known node, operation or primitive identity, valid buffers and roles, written
logical outputs, scalar/domain/fixed-format and numeric-policy metadata, and all descriptor
bounds. Unlike execution admission, recipe identity permits a node's element
count to be zero so the operation remains part of Program/cache identity
without inventing a dispatch. `ValidateGraph(...)` remains the execution
authority and rejects that zero-element node. Both validators reject a truly
empty graph with `compute_graph_empty`.

Graph descriptors are bounded before any pointer/count walk. A Program graph
admits at most `kMaxGraphNodeCount == 16384` ordered nodes and at most
`kMaxGraphBuffersPerNode == 64` buffers per node. The complete logical-value
construction envelope is

```text
kMaxGraphValueCount
  = kMaxGraphNodeCount * kMaxGraphBuffersPerNode
    + kMaxGraphOutputCount
  = 1,048,640
```

Every admitted resource must occur in a bounded node reference or the bounded
output set, so this is derived from the graph descriptor law rather than an
independently tuned resource number. Graph nodes and values use storage
proportional to the authored graph; the limits do not instantiate maximum-size
tables in every Program.

The Program schedule envelope is deliberately distinct from the per-Map
`kMaxComputeNodeCount == 1024` IR envelope. A large ordered Program therefore
does not inflate one shader or require the caller to split execution authority.
Fusion starts a new maximal region at the existing per-IR bound while retaining
the same Program graph, node order, fingerprint, and submission authority.
The finite Program bound keeps forged `u64` count descriptors fail-closed
before hashing.

Graph validation rejection reasons are contract vocabulary:

| Gate | Stable reason |
| --- | --- |
| Empty graph descriptor | `compute_graph_empty` |
| Node count exceeds graph contract bound | `compute_graph_node_count_invalid` |
| Missing nodes, unknown node kind, or missing map operation hash | `compute_graph_node_invalid` |
| Missing collective primitive hash, zero collective element count, or mixed map/primitive identity fields | `compute_graph_primitive_invalid` |
| Per-node buffer count exceeds graph contract bound | `compute_graph_buffer_count_invalid` |
| Missing buffers, zero logical buffer id, or unknown buffer role | `compute_graph_buffer_invalid` |
| Unsupported scalar, domain, fixed-format, rounding, overflow, or approximation authority | `compute_graph_numeric_invalid` |
