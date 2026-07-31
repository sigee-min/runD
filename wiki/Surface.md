# Public API Surface

The canonical direct-header, transitive-support, and release boundary is
owned by the [package API authority](./Stability.md). This page is the Wiki
reference map and intentionally repeats no normative header inventory.

| Product area | Reference |
| --- | --- |
| Session | [Run](./Run.md) |
| Cooperative tasks | [Tasks](./Tasks.md) |
| Host services | [Host](./Host.md) |
| Network transport | [Network](./Network.md) |
| Telemetry | [Telemetry](./Telemetry.md) |
| Replay and evidence | [Replay](./Replay.md), [Evidence](./Evidence.md) |
| Compute | [Compute](./Compute.md) |
| Fixed-width math | [Math32](./Math32.md), [Math64](./Math64.md) |
| Placement policy | [Cluster](./Cluster.md) |

Use runD from private implementation files unless your product intentionally
makes runD part of its own public API. Confirm every include and artifact
decision against the canonical package pages linked from
[API Stability](./Stability.md).
