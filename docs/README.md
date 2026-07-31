# runD Docs

`/docs` is the repository source of truth for architecture routing, subsystem
contracts, development workflow, naming, evidence classes, and repo-local
acceptance framing.

## Read Order

1. This page for repository routing.
2. [Process](./process/README.md) for development workflow, evidence, and
   naming rules.
3. [Architecture](./architecture/README.md) for cross-layer boundaries.
4. The affected subsystem docs.
5. Public headers, implementation, contract tests, and measured evidence.

If repository docs disagree about product behavior, process, or ownership, the
task remains open until the owning page is fixed or the conflict is recorded as
a blocker. If repository docs and implementation disagree about product
behavior, update the docs and implementation together or record the blocker.

## Map

| Path | Owns |
| --- | --- |
| [`process/`](./process/README.md) | Development workflow, evidence, naming, and anti-duplication rules. |
| [`architecture/`](./architecture/README.md) | Cross-layer topology, numeric policy, and repository layout policy. |
| [`reference/`](./reference/README.md) | Public Compute guidance, stable evidence snapshots, and focused reference records. |
| [`/math32/docs`](../math32/docs/README.md) | 32-bit deterministic real-time vector math laws, prepared SoA range surfaces, and SIMD contract evidence. |
| [`/math64/docs`](../math64/docs/README.md) | 64-bit deterministic high-precision vector math laws and widened-lane contract evidence. |
| [`/kernel/docs`](../kernel/docs/README.md) | Kernel scheduling, workspace, dispatch, reduction, program, telemetry, and verification contracts. |
| [`/accel/docs`](../accel/docs/README.md) | Internal backend-neutral AccelKernel values and Node compute-bridge routing. |
| [`/node/docs`](../node/docs/README.md) | Local runtime, resource admission, topology evidence, scheduler, host, network, and replay. |
| [`/cluster/docs`](../cluster/docs/README.md) | Optional distributed placement and retry identity policy. |
| [`/package`](../package/README.md) | SDK/release artifact packaging, CMake package export, and external consumption policy. |

## Root Rules

- One behavior has one docs owner.
- Link to the owning page instead of restating its rules.
- Process pages own repository workflow. Subsystem docs own product behavior.
- Behavior, contract, naming, or acceptance changes update docs and
  implementation together.
- A behavior change updates or deletes conflicting docs in the same change.
- Performance claims require the evidence rules in
  [Evidence](./process/evidence.md).
- Document names and page shape follow [Naming](./process/naming.md).
