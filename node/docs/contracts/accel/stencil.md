# Accel Stencil Contract

Node owns resident backend execution for kernel-planned `Stencil` graph
steps. Kernel-owned descriptor, hash, pure planning, CPU reference, boundary
policy, and stable reason vocabulary stay in
[`/kernel/docs/contracts/compute.md`](../../../../kernel/docs/contracts/compute.md).

## Authority

Public support surface:

- `/node/include/node/accel/context.hpp`

Implementation authority:

- `/node/src/accel/stencil.hpp`
- `/node/src/accel/range_aggregate/{model,plan}.hpp` as the sole
  algorithm/capability/cost selector
- `/node/src/accel/range_aggregate/execution.hpp` as the generic `RangeParams`
  host parameter ABI owner
- `/node/src/accel/stencil/shape.{hpp,cpp}` as the sole Stencil-to-Range
  semantic projection and resident-span validation owner
- `/node/src/accel/primitive/shape.hpp`
- `/node/src/accel/cpu/stencil.cpp`
- `/node/src/accel/{metal,vulkan}/range/` for physical lookup, source,
  pipeline, scratch, and dispatch execution
- `/node/src/accel/{metal,vulkan}/stencil/` for the thin semantic adapter
- `/node/src/accel/collective*`
- `/node/src/accel/graph.cpp`
- `/node/src/accel/graph/collective/{bindings.hpp,defaults.cpp,desc.cpp,kind.cpp}`
- `/node/src/accel/kernel/bindings/range.cpp` for the shared physical
  two-buffer `RangeBinds`
- `/node/src/accel/kernel/plan/{compute,step}.cpp`
- `/node/src/accel/kernel/backend/run.cpp` for the canonical bound-step view

Verification authority:

- `/node/tests/contract/accel/kernel/stencil.cpp`
- `/node/tests/contract/accel/kernel/stencil/`
- `/node/tests/contract/accel/kernel/range/`
- `/node/tests/contract/accel/kernel/primitive/local.hpp`

## Contract

`CompileAccelKernel(context, graph)` admits `NodeKind::Stencil`
only when the node carries `AccelGraphNode::stencil`,
`PlanStencil(stencil).ok`, a primitive hash equal to
`HashStencil(stencil)`, default Sort descriptor, default non-stencil
collective descriptors, matching `element_count`, and exactly two bindings in
role order: `(read input, write output)`.

Stencil execution supports `Sum`, `Min`, and `Max` with clamp
boundary over u32 or u64 elements and radius values in `[1, element_count]`:

```text
left(i, d)  = i < d ? 0 : i - d
right(i, d) = i + d >= element_count ? element_count - 1 : i + d
output[i] = input[i] + sum(input[left(i, d)] + input[right(i, d)])
            for d in 1..radius
min/max output[i] = extremum over the same clamped window
```

Sum arithmetic wraps modulo the selected element width for every domain;
min/max compare the declared signed or unsigned domain interpretation. There
is no reduction, atomic accumulation,
schedule-dependent write, or floating-point authority.
Every semantic output element has one writer and reads only the frozen input
resident buffer, so CPU, Metal, and Vulkan may choose different physical lane
grouping while preserving the same output bits.

The admitted input and output resident spans must not overlap. They may share
one physical owner only when their exact byte spans are disjoint. This keeps
the input frozen for the complete dispatch and prevents CPU update order or GPU
inter-thread races from becoming Stencil semantics.

## Frozen Range execution

Graph admission projects the accepted Kernel descriptor into
`RangeTraits` and `RangeShape`, takes one backend capability
snapshot, and freezes `PlanRange(...)` before the execution token is
minted. `HashStencil`, the public graph descriptor, and the CPU reference stay
semantic authorities; the frozen plan is a resident execution authority only.

The planner may select the following exact execution families:

| Family | Stencil algebra | Resident stages |
| --- | --- | --- |
| Direct | Any admitted operation | One window stage |
| SharedHalo | Any admitted operation when its exact halo fits | One window stage |
| PrefixDifference | Modulo-width Sum | Prefix hierarchy, reverse fix-up, window output |
| BlockPrefixSuffix | Min or Max | Block forward/backward construction, window output |

`/node/docs/contracts/accel/range-aggregate.md` owns legality, cost formulas,
candidate dominance, tie-breaking, and the large-radius `O(N)` proofs. The
Stencil projection never ranks a second candidate set. `RangeExec` maps the
selected plan to a backend workgroup width, per-stage dispatch shape,
descriptor count, dispatch-local shared bytes, and typed global temporary
requirements. Stencil supplies only Clamp semantics, public resident bindings,
and public result/error mapping to that executor.

`RangeParams` carries the immutable logical window plus the active stage:

```text
input_count, output_count, window_size, stride, padding,
stage_element_count, stage_aux_count,
stage, reserved
```

Both backends consume that ABI for every selected stage. The width is always
one of `{64, 128, 256}`. A SharedHalo plan declares `(W + 2C)E` bytes and has a
uniform barrier before an inactive tail lane may return. Its loader reads the
expanded distinct input union exactly once; clamp endpoint duplicates are
lane-private loaded values fanned into the required halo slots. PrefixDifference
declares one width-sized scan array. Direct and BlockPrefixSuffix declare no
dispatch-local shared array.

Metal validates the frozen width against the device dimension limit and the
compiled pipeline total-thread limit. It validates modeled static shared bytes
against the pipeline's reported static allocation and the device threadgroup
limit under the planner's integer reserve. Vulkan validates width, invocation,
shared-memory, logical-index, and dispatch-group limits from the selected
physical device. These are capability and resource checks, not claims about
physical occupancy, register allocation, spills, or wall-clock performance.

## Scratch, stages, and publication

PrefixDifference supplies prefix values and hierarchy summaries. BlockPrefixSuffix
supplies forward and backward values. These are non-owning typed requirements
with exact bytes, alignment, and closed stage lifetimes. The existing Pipeline
scratch planner alone assigns their arena placements; serial stages reuse its
component-wise envelope and warm execution performs no allocation. Threadgroup
shared memory remains pipeline metadata rather than arena storage.

Each frozen Range stage owns one dispatch slot. Metal binds that slot's
descriptor-counted parameters and freezes its selected pipeline pointer;
Vulkan binds the shared data pipeline with one stage-specific parameter and
descriptor-set instance. A global-memory barrier separates dependent stages.
The backend manifest and immutable-template capacity record the exact frozen
stage and descriptor demand. A failed prepare publishes no partially usable
resident execution object. The exact unique-pipeline and descriptor
cardinality is owned by the Range planning contract.

Metal source-library and named-pipeline publication are transactional:
`Inserted`, `Existing`, and `Failed` retain canonical ownership and account
constructed compile work exactly once. A source-library owner retained after a
named-pipeline publication failure is reusable by a later retry; no stage
becomes visible until its pipeline owner is valid.

## Identity and parity

The Range source identity contains backend source class, family,
workgroup width, shared capacity, operation, domain, arithmetic law, boundary,
and element width. The execution identity additionally contains `N`, `Q`, `K`,
`S`, `P`, the complete stage graph, temporary requirements, and exact modeled
cost. Metal pipeline labels and Vulkan artifact/source matching consume the
source identity; manifest reservation, immutable-template reuse, parameter
dispatch, and scratch placement consume `RangeExec`. A pipeline is reusable
only when that complete source identity matches.

The stencil contract verifies Direct, SharedHalo, PrefixDifference, and
BlockPrefixSuffix results against the unchanged CPU reference across tails,
cross-workgroup windows, clamp boundaries, signed and unsigned domains, and
u32/u64 elements. It also verifies a capability-derived shared plan on each
available backend, immutable reuse of the matching resident stage, and
capability-derived source cache separation. The test names state the selected
algorithm family and capability-derived physical shape.

For the large-window Sum and Min/Max fixtures, the contract rebuilds the
same backend capability snapshot with `Direct` and exactly one selected prefix
family as its legal candidates. It executes both frozen plans against the same
resident input and requires their CPU-reference output bits to match. This is
an execution parity check, not a second selection policy.

The source contracts also authenticate the PrefixDifference local scan,
hierarchy scratch, inverse window stage, and the BlockPrefixSuffix
forward/backward block stages in the exact generated Metal and Vulkan source;
their exact source-byte recipes and identities are checked with the same
frozen plans.

The backend parity contract also executes a PrefixDifference window with
`N = 65,537` and `r = 257`.  That input creates at least three hierarchy
levels for each legal width in `{64, 128, 256}`, so the exact summary-role
bindings and reverse fix-up stages are exercised beyond the first block level.

Input and output buffers must exactly match the planned element width and
`element_count`. Resident Stencil runs do not stage host input and do not
download output implicitly; only explicit upload/download calls affect
user-facing transfer byte counters. `RunAccelKernel(...)` reports the exact
frozen Range stage count for Stencil.
