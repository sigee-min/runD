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
| [Method](./method.md) | Workload schemas, sampling, semantic admission, comparison, evidence packets, and baseline update rules. |
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
`--bulk`, and `--pipeline` build current-source diagnostics and cannot publish
Release baseline evidence. A passing upper bound is regression evidence, not a
speedup claim. See [Method](./method.md) for the complete acceptance and
publication contract.
