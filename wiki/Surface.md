# Public API Surface

The canonical direct-header, transitive-support, and release boundary is
owned by the [package API authority](https://github.com/sigee-min/runD/wiki/Stability). This page is the Wiki
reference map and intentionally repeats no normative header inventory.

| Product area | Reference |
| --- | --- |
| Session | [Run](https://github.com/sigee-min/runD/wiki/Run) |
| Cooperative tasks | [Tasks](https://github.com/sigee-min/runD/wiki/Tasks) |
| Host services | [Host](https://github.com/sigee-min/runD/wiki/Host) |
| Network transport | [Network](https://github.com/sigee-min/runD/wiki/Network) |
| Telemetry | [Telemetry](https://github.com/sigee-min/runD/wiki/Telemetry) |
| Replay and evidence | [Replay](https://github.com/sigee-min/runD/wiki/Replay), [Evidence](https://github.com/sigee-min/runD/wiki/Evidence) |
| Compute | [Compute](https://github.com/sigee-min/runD/wiki/Compute) |
| Fixed-width math | [Math32](https://github.com/sigee-min/runD/wiki/Math32), [Math64](https://github.com/sigee-min/runD/wiki/Math64) |
| Placement policy | [Cluster](https://github.com/sigee-min/runD/wiki/Cluster) |

Use runD from private implementation files unless your product intentionally
makes runD part of its own public API. Confirm every include and artifact
decision against the canonical package pages linked from
[API Stability](https://github.com/sigee-min/runD/wiki/Stability).
