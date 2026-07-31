# Kernel Platform

## Boundary

Kernel owns deterministic packet scheduling, partition placement, workspace
capacity, worker dispatch, fixed-order reduction, execution telemetry,
`KernelProgram`, and rank-generic skeleton traversal.

Packet identity is immutable after schedule compilation. Worker pickup and
completion order are never semantic order. A no-growth run requires a checked
Workspace capacity proof.

## Module Ownership

- `CORE`: primitive fixed-width types and packet range records.
- `COMPUTE`: Compute binding validation, IR, graph signature, lowering,
  artifact admission, fusion, and metadata.
- `DISPATCH`: backend capability inspection, partition validation, execution,
  failure signaling, and prepared program orchestration.
- `PROGRAM`: program compile, executor/provider, physical tile planning,
  Compute tile execution, reports, and program state.
- `REDUCTION`: fold slots, fold graph, ordered reduction, and strict
  floating-point reference behavior.
- `SCHEDULE`: partition projection, placement, workspace reservation,
  no-growth capacity proof, and schedule telemetry.

Public headers live under `/kernel/include/kernel`. Internal headers under
`kernel/internal` and sources under `/kernel/src` do not become public
authority because a focused test includes them.

## Build Ownership

The source registry partitions all 121 Kernel translation units into six
non-empty, non-overlapping components:

| Component | Translation units |
| --- | ---: |
| CORE | 1 |
| COMPUTE | 8 |
| DISPATCH | 14 |
| PROGRAM | 18 |
| REDUCTION | 36 |
| SCHEDULE | 44 |

Every source has exactly one OBJECT owner. Configuration rejects missing,
unregistered, duplicate, or non-canonical sources.

The installed `kernel` archive is the union of all six components and retains
121 members. Node links three pairwise-disjoint views from the same objects:

- compute: `COMPUTE`, 8 translation units
- dispatch: `DISPATCH`, 14 translation units
- execution: `CORE + PROGRAM + REDUCTION + SCHEDULE`, 99 translation units

Their union is the complete 121-source registry. No Kernel implementation is
compiled twice for those views.

The dispatch and execution views form one static-link strongly connected
component: `PROGRAM` invokes prepared dispatch, while dispatch reads failure
state from `CORE` and records telemetry into `SCHEDULE`. CMake owns both
directed archive edges and a rescan multiplicity of two. A one-pass linker
therefore observes the same complete definition set as an archive-indexing
linker without merging the disjoint object owners or relying on command-line
order.

Timed compile flags are derived from semantic paths in the one registry.
There is no second source list.

## Contract Pages

- [Kernel Program](../contracts/program.md)
- [Kernel Skeleton](../contracts/skeleton.md)
- [Workspace](../contracts/workspace.md)
- [Reduction](../contracts/reduction.md)
- [Compute](../contracts/compute.md)

## Verification Surfaces

- `tools/test/run` runs the representative Kernel smoke routes.
- `tools/check/run` runs the complete disjoint Kernel contract groups.
- Build-graph verification requires the 121-member installed archive and the
  exact 1, 8, 14, and 99-member closure counts.
- Public surface changes additionally require `tools/release/run` and the
  external package consumer.

## Platform Rules

Keep module ownership here, stable behavior in the nearest contract page, and
verification routing in the repository process docs.
