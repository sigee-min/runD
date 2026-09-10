# Performance

This directory routes performance evidence. Runtime timing claims consume the
installed Release SDK and are valid only for an admitted host profile, exact
source manifest, unchanged workload identity, verified output, and the
measurement boundary defined by the method. Build measurement observes the
live compiler graph through its separate boundary.

## Owners

| Path | Owns |
| --- | --- |
| [GPU Workload Sizing](./gpu.md) | Arithmetic intensity, offload break-even, execution-shape guidance, and the interpretation of admitted GPU evidence. |
| [Metal Scatter Reduce](./metal-scatter-reduce.md) | Current-source warm indexed fold contention comparison and shared-memory bounds. |
| [Indexed Execution](./indexed-execution.md) | CPU Scatter Reduce scratch and native indexed preflight comparisons, Virtual route ownership, and claim-gate diagnostic limits. |
| [Metal Histogram](./metal-histogram.md) | Current-source warm Histogram contention/locality comparison and claim limits. |
| [Metal SIMD-group Min/Max](./metal-extrema.md) | Current-source warm GPU extrema comparison, exact SIMD packing and claim limits. |
| [CPU Affine Maps](./cpu-affine.md) | Current-source warm integer Map comparison, method and claim limits. |
| [Memory Observation](./memory.md) | Focused current-source memory observer allocation, stack, and serial cost evidence. |
| [Method](./method.md) | Workload schemas, sampling, semantic admission, comparison, evidence packets, and baseline update rules. |
| [Virtual Performance](./virtual/README.md) | Current-source VirtualPipeline measurement method and bounded observed results. |
| [`baseline.tsv`](./baseline.tsv) | Checked host profiles, semantic digests, units, exact values, and one-sided regression limits. |
| `/tools/measure` | Installed-SDK workload executables and public measurement commands. |

The table is data, not a second schema owner. Every command emits the units and
semantic identity defined by its parser; the comparator rejects a table row
whose profile, identity, metric, or unit disagrees.

## Commands

- `tools/measure/scheduler/run`
- `tools/measure/compute/run`
- `tools/measure/flow/run`
- `tools/measure/graph/services/run`
- `tools/measure/telemetry/run`
- `tools/measure/admit/run` — seal three observations per installed route and
  perform the Release median admission.
- `tools/measure/build/run [build [target]]`

These argument-free measurement commands consume the installed Release SDK and
never edit the baseline. Compute options such as `--resident`, `--sort`,
`--bulk`, `--pipeline`, and `--virtual-residency` build current-source
diagnostics and cannot publish Release baseline evidence. Virtual diagnostic
procedure and results are routed through
[Virtual Performance](./virtual/README.md); they add no Product row and do not
change the installed Compute route's 82-metric cardinality. A passing upper
bound is regression evidence, not a speedup claim. See [Method](./method.md) for
the generic acceptance and publication contract.
