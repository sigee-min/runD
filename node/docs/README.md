# Node Docs

`/node` is the single-machine runtime and SDK bridge. It owns resource
admission, topology projection, cooperative tasks, deterministic host and
network services, Replay, Session, telemetry, and concrete CPU/Metal/Vulkan
Compute execution. `/kernel` and `/accel` do not depend on `/node`.

## Authority

1. [Repository topology](../../docs/architecture/topology.md)
2. This page for the Node boundary
3. [Guide](./guide.md) for developer routing
4. [Contracts](./contracts/README.md) for product behavior
5. Source-private headers under `/node/include/node`
6. Implementation under `/node/src`
7. Contracts under `/node/tests/contract`

Kernel scheduling, partitioning, workspaces, reduction order, and telemetry
truth remain owned by [Kernel](../../kernel/docs/README.md). Backend-neutral
graph and run values remain owned by [Accel](../../accel/docs/README.md).
Distributed placement remains owned by
[Cluster](../../cluster/docs/README.md).

## Public Boundary

Applications enter Node behavior through the focused `<rund/*.hpp>` headers
and `runD::sdk`. Task names live only in `rund::task`; shared runtime failures
live in `rund::ReasonCode`. Direct `/node/include/node/...` paths and
`rund::node` names are implementation authority, not SDK entries. The exact
installed boundary is owned by [SDK Surface](../../package/docs/surface.md).

## Map

| Page | Owns |
| --- | --- |
| [Guide](./guide.md) | Runtime product map and developer entry points. |
| [Contracts](./contracts/README.md) | Node contract index. |
| [Runtime](./contracts/runtime.md) | Session lifecycle and product result. |
| [Scheduler](./contracts/scheduler/README.md) | Task, channel, timer, Reactor, and progress laws. |
| [Host](./contracts/host.md) | Deterministic host observations and replayable effects. |
| [Network](./contracts/net.md) | Socket ownership, readiness, I/O, and bounded server behavior. |
| [Replay](./contracts/replay.md) | Canonical input, checkpoint, scenario, retention, and reproduction. |
| [Storage](./contracts/storage.md) | Hierarchical allocated-capacity reservation, physical/allocated usage, refund, and reporting. |
| [Telemetry](./contracts/telemetry.md) | Levels, events, findings, and parity. |
| [Build Graph](./contracts/build/graph.md) | Translation-unit ownership and focused link closure. |
| [Compute Pipeline](./contracts/compute/pipeline.md) | Frozen dependent Program order, resident Buffer hazards, one prepared execution, claims, poison, readback, evidence, and Session integration. |

## Verification

Exact case names, backend selectors, link profiles, locks, and broad routes are
owned by [Repository Verification](../../docs/architecture/verification.md).
Use `tools/test/run --list`, run the narrowest exact case, then use
`tools/test/run --fresh ...` when a new observation is required. Local passes
are disposable edit-loop acceleration; `tools/check/run` and
`tools/release/run` remain the repository and installed-SDK closure.
