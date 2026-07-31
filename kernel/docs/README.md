# Kernel Docs

`/kernel/docs` owns `/kernel` behavior. Root docs may route here, but kernel
contracts, verification, and module ownership are defined here.

## Authority

1. [Platform](./architecture/platform.md)
2. [Contracts](./contracts/README.md)
3. Public headers under `/kernel/include/kernel`
4. Tests under `/kernel/tests`

If these layers disagree, fix the owning kernel authority or record the
blocker before closing the task.

## Boundary

Kernel owns packet scheduling, partition projection, workspace capacity,
backend validation, dispatch telemetry, ordered reduction, `KernelProgram`,
rank-generic skeleton execution, and the internal exact 64-bit capacity
arithmetic shared with Node.

The checked capacity owner is an implementation safety boundary, not public
arithmetic law. Kernel does not own public arithmetic law or cross-layer
numeric policy:

- arithmetic law: [`/math32/docs`](../../math32/docs/README.md) and
  [`/math64/docs`](../../math64/docs/README.md)
- numeric policy: [`/docs/architecture/numeric.md`](../../docs/architecture/numeric.md)

## Contents

| Page | Owns |
| --- | --- |
| [Platform](./architecture/platform.md) | Kernel boundary, module ownership, and verification surfaces. |
| [Contracts](./contracts/README.md) | Stable kernel behavior. |

## Kernel Rules

- Non-kernel behavior stays outside `/kernel/docs`.
- Use `tools/test/run` for the representative Orchestrator and Compute smoke
  owners. Use `tools/check/run` for those plus the disjoint
  Dispatch/Schedule/Workspace and complete Compute contract groups.
- Kernel code paths use semantic directory depth instead of repeated prefixes.
- General naming, linking, and anti-duplication rules live in
  [`/docs/process`](../../docs/process/README.md).

## SDK Artifact

Release packaging for this subsystem is owned by
[`/package`](../../package/README.md). This subsystem owns behavior and tests;
`/package` owns artifact layout, CMake package export, and external
consumption policy.
