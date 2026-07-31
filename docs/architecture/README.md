# Architecture

This directory owns cross-layer architecture. Subsystem details stay in the
subsystem docs listed below.

## Contents

| Page | Owns |
| --- | --- |
| [Layout](./layout.md) | Repository root categories, admission, generated/local state, and physical root changes. |
| [Topology](./topology.md) | Global kernel/node/cluster topology, kernel run identity, node admission boundaries, shard meaning, and single-node default. |
| [Numeric](./numeric.md) | Cross-layer numeric policy plus public numeric contract type model, invalid combinations, preset UX, identity, serialization, and kernel lowering. |
| [Verification](./verification.md) | Test tier ownership, inner-loop routing, release gate routing, and evidence policy. |
| [Source Ownership](./source/ownership.md) | Semantic file ownership and dependency-boundary rules. |
| [`/math32/docs`](../../math32/docs/README.md) | Public deterministic 32-bit real-time vector arithmetic law, geometry lane helpers, approximation law, prepared SoA range surfaces, and math32 contract evidence. |
| [`/math64/docs`](../../math64/docs/README.md) | Public deterministic 64-bit high-precision vector arithmetic law, geometry lane helpers, approximation law, and math64 contract evidence. |
| [`/kernel/docs`](../../kernel/docs/README.md) | Kernel-owned packet scheduling, workspace, dispatch, reduction, program capacity, telemetry, and contract evidence. |
| [`/node/docs`](../../node/docs/README.md) | Single-machine resource admission, topology evidence, worker backend construction, host, network, and replay. |
| [`/cluster/docs`](../../cluster/docs/README.md) | Optional multi-node shard-to-node placement and retry identity policy. |

Kernel no-growth execution capacity is owned by
[`/kernel/docs/contracts/workspace.md`](../../kernel/docs/contracts/workspace.md),
while Session memory evidence is owned by
[`/node/docs/contracts/runtime.md`](../../node/docs/contracts/runtime.md).
User payload storage and physics storage policy remain outside runD ownership.

## Architecture Rules

- Cross-subsystem behavior needs an explicit authority path from root docs to
  subsystem docs and then to code, tests, or measurements.
- Do not copy detailed subsystem contracts into architecture docs.
- Kernel behavior stays in `/kernel/docs`.
- Math arithmetic law stays in `/math32/docs` and `/math64/docs`.
- Cross-layer numeric policy and public numeric contract shape stay in
  [`numeric.md`](./numeric.md).
- Test tier routing and owner-local verification policy live in
  [`verification.md`](./verification.md).
- Repository root admission and physical relocation rules live in
  [Layout](./layout.md).
- Semantic ownership and dependency-boundary policy lives in
  [Source Ownership](./source/ownership.md).
- Page naming and shape rules live in [Naming](../process/naming.md).
