# API Reference

Use this folder to find the SDK module page for each documented direct include
header.

Include the headers listed in this table directly. Some SDK artifacts also
contain support-only transitive headers needed by these modules; those support
headers are not a separate include surface for application code.

| Module | Header | Use For |
| --- | --- | --- |
| [Run](https://github.com/sigee-min/runD/blob/main/wiki/Run) | `<rund/session.hpp>` | Configuring a Session, launching work, and reading its result. |
| [Replay](https://github.com/sigee-min/runD/blob/main/wiki/Replay) | `<rund/replay.hpp>` | Recording and checking runtime replay evidence from run results. |
| [Telemetry](https://github.com/sigee-min/runD/blob/main/wiki/Telemetry) | `<rund/session.hpp>` | Selecting Basic or Detail terminal observations for Session work. |
| [Evidence](https://github.com/sigee-min/runD/blob/main/wiki/Evidence) | `<rund/evidence.hpp>` | Building and encoding stable numeric evidence. |
| [Compute](https://github.com/sigee-min/runD/blob/main/wiki/Compute) | `<rund/compute.hpp>`; focused `<rund/compute/pipeline.hpp>`; opt-in `<rund/compute/async.hpp>`, `<rund/compute/math.hpp>`, and `<rund/compute/session.hpp>` | Typed Flow composition, one explicit `Target`, resident Job and dependent Pipeline reuse, async compilation, composite math, and Session-host submission. |
| [Composition](https://github.com/sigee-min/runD/blob/main/wiki/Surface) | `<rund/rund.hpp>` | Deliberately composing every focused runD domain in one translation unit. |
| [Tasks](https://github.com/sigee-min/runD/blob/main/wiki/Tasks) | `<rund/task.hpp>` | Spawning tasks, task groups, scoped work, yielding, sleeping, and ergonomic group waits. |
| [Host](https://github.com/sigee-min/runD/blob/main/wiki/Host) | `<rund/host.hpp>` | Host-facing environment reads, file IO, random streams, logical time, and timers. |
| [Network](https://github.com/sigee-min/runD/blob/main/wiki/Network) | `<rund/net.hpp>` | Sockets, listeners, readiness, server and peer byte lifecycle, flow accounting, bounded frame bytes, datagrams, options, and drains. |
| [Math32](https://github.com/sigee-min/runD/blob/main/wiki/Math32) | `<math32/math32.hpp>` | 32-bit deterministic math, fixed-point, SIMD, and vector families. |
| [Math64](https://github.com/sigee-min/runD/blob/main/wiki/Math64) | `<math64/math64.hpp>` | 64-bit deterministic math, fixed-point, SIMD, and vector families. |
| [Cluster](https://github.com/sigee-min/runD/blob/main/wiki/Cluster) | `<cluster/cluster.hpp>` | Cluster identity, placement, and retry APIs. |

`<rund/rund.hpp>` is the declaration-free convenience composition of every
focused `rund` domain above, not the default entry for one domain. For the
direct include list, see
[Public API Surface](https://github.com/sigee-min/runD/blob/main/wiki/Surface).
For release use, check [Platform Support](https://github.com/sigee-min/runD/blob/main/wiki/Platforms),
[API Stability](https://github.com/sigee-min/runD/blob/main/wiki/Stability), and
[Release Checklist](https://github.com/sigee-min/runD/blob/main/wiki/Checklist).

Network frame bytes are not a packet schema, not a protocol, and not a session;
application layers own byte meaning above the SDK byte carrier.
