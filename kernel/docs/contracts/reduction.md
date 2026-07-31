# Reduction Contract

## Scope

This page owns `rund::kernel::FoldSlots`, `rund::kernel::FoldGraph`, ordered reductions,
and strict floating-point reference reductions.

Cross-layer numeric policy is owned by
[`/docs/architecture/numeric.md`](../../../docs/architecture/numeric.md).
This page owns only the reduction-local numeric classifications needed to
validate fold primitives.

Public authority:

- `/kernel/include/kernel/reduction/fold/slots.hpp`
- `/kernel/include/kernel/reduction/fold/operation.hpp`
- `/kernel/include/kernel/reduction/fold/primitive.hpp`
- `/kernel/include/kernel/reduction/fold/result.hpp`
- `/kernel/include/kernel/reduction/fold/strict.hpp`
- `/kernel/include/kernel/reduction/fold/graph/state.hpp`
- `/kernel/include/kernel/reduction/fold/graph/node.hpp`
- `/kernel/include/kernel/reduction/fold/graph/result.hpp`
- `/kernel/include/kernel/reduction/fold/graph/api.hpp`

There is no fold, type, or graph aggregate include. Callers include the
operation, primitive, result, or strict-policy leaf they consume. A graph
value-only owner includes state or graph result; only a caller that builds,
validates, or executes a graph includes the graph API.

Implementation authority:

- `/kernel/src/reduction/fold`
- `/kernel/src/reduction/graph`
- `/kernel/src/reduction/strict`

Verification authority:

- `/kernel/tests/reduction/fold`
- `/kernel/tests/contract/program/strict.cpp`

## Fold Slots

`rund::kernel::FoldSlots` owns generic worker-local fact storage and the ordered slot
protocol. The kernel guarantees slot storage, bounds checking, and ordered fold
helpers such as xor, max, min, saturating add, and fixed binary tree hash.

`rund::kernel::FoldOrderedSlots` is the generic reduction entrypoint for ordered slot
views.

## Fold Graph

`rund::kernel::FoldGraph` is the program-level canonical reduction graph IR. It
records:

- nodes and edges
- per-partition worker-local slots
- global ordered slots
- primitive, padding, and overflow laws
- no-growth scratch requirements

The validator fails closed on unsupported primitives, missing fixed
topological order, slot-bound errors, output redefinition, primitive mismatch,
padding-law mismatch, and result-slot mismatch.

`rund::kernel::FoldGraphReduce` executes standardized xor, max, min, saturating add,
fixed hash, and admitted strict floating-point primitives through the graph
using caller-owned scratch.

Public graph reduction treats every `FoldGraphView` as untrusted input. The
`dag_validated`, `slot_bounds_validated`, and `padding_law_validated` fields are
reported evidence, not authority to skip validation, because a view is a public
raw-pointer structure. The public reducer therefore validates fail-closed on each
call.

Generic fold-graph validation must remain linear in graph size. Slot-indexed
markers in caller-owned scratch record partition slots and reduction-edge output
ownership, so node validation performs constant-time membership and edge lookup
instead of scanning partition slots or edge lists.

## Fold Value Domain

`FoldValueDomain` is reduction-local classification. It exists so fold
validation, graph evidence, primitive descriptions, and strict-floating policy
can agree on the value class consumed by one ordered reduction primitive.

`FoldValueDomain` is not the universal numeric model for runD. It must not be
used to define public scalar domains, user value types, gameplay values,
simulation values, math API types, or caller-supplied operation types.
Repository-wide numeric policy and public numeric authority routing are owned
by `/docs/architecture/numeric.md`; deterministic integer and
fixed-point arithmetic law is owned by `/math32/docs` and `/math64/docs`.

## Strict Floating Point

Floating-point reductions are forbidden by default. Only `StrictFloat32Add`
and `StrictFloat64Add` with an explicit `StrictFloatReductionPolicy` are
admitted.

The reference strict mode fixes:

- nearest-ties-to-even rounding
- canonical quiet NaN
- signed-zero policy
- infinity and subnormal preservation
- software execution with FMA forbidden

Capability and telemetry fields do not imply that a hardware strict-FP
path exists or ran.

Strict floating-point reduction admission does not authorize arbitrary
authoritative floating-point state outside this reduction contract.
## Update Rules

- Fold primitive changes, graph validation changes, strict-FP policy changes,
  `FoldValueDomain` classification changes, or scratch requirements must update
  this page and the focused reduction tests in the same change.
