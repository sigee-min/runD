# API Reference

runD exposes one C++20 SDK target with focused headers for deterministic
Compute, replay, bounded runtime work, networking, telemetry, and fixed-point
math. If you came for bit-identical CPU/GPU execution, begin with
[Compute](./Compute.md); the other modules supply its runtime, evidence, and
integration boundaries.

Use this folder to find the SDK module page for each documented direct-include
header.

Include the headers listed in this table directly. Some SDK artifacts also
contain support-only transitive headers needed by these modules; those support
headers are not a separate include surface for application code.

| Module | Header | Use For |
| --- | --- | --- |
| [Run](./Run.md) | `<rund/session.hpp>` | Configuring a Session, launching work, and reading its result. |
| [Replay](./Replay.md) | `<rund/replay.hpp>` | Recording and checking runtime replay evidence from run results. |
| [Telemetry](./Telemetry.md) | `<rund/session.hpp>` | Selecting Basic or Detail terminal observations for Session work. |
| [Evidence](./Evidence.md) | `<rund/evidence.hpp>` | Building and encoding stable numeric evidence. |
| [Compute](./Compute.md) | `<rund/compute.hpp>`; focused `<rund/compute/pipeline.hpp>`; opt-in `<rund/compute/async.hpp>`, `<rund/compute/math.hpp>`, and `<rund/compute/session.hpp>` | Typed Flow composition, one explicit `Target`, resident Job and dependent Pipeline reuse, async compilation, composite math, and Session-host submission. |
| [Composition](./Surface.md) | `<rund/rund.hpp>` | Deliberately composing every focused runD domain in one translation unit. |
| [Tasks](./Tasks.md) | `<rund/task.hpp>` | Spawning tasks, task groups, scoped work, yielding, sleeping, and ergonomic group waits. |
| [Host](./Host.md) | `<rund/host.hpp>` | Host-facing environment reads, file IO, random streams, logical time, and timers. |
| [Network](./Network.md) | `<rund/net.hpp>` | Sockets, listeners, readiness, server and peer byte lifecycle, flow accounting, bounded frame bytes, datagrams, options, and drains. |
| [Math32](./Math32.md) | `<math32/math32.hpp>` | 32-bit deterministic math, fixed-point, SIMD, and vector families. |
| [Math64](./Math64.md) | `<math64/math64.hpp>` | 64-bit deterministic math, fixed-point, SIMD, and vector families. |
| [Cluster](./Cluster.md) | `<cluster/cluster.hpp>` | Cluster identity, placement, and retry APIs. |

`<rund/rund.hpp>` is the declaration-free convenience composition of every
focused `rund` domain above, not the default entry for one domain. For the
direct include list, see
[Public API Surface](./Surface.md).
For release use, check [Platform Support](./Platforms.md),
[API Stability](./Stability.md), and
[Release Checklist](./Checklist.md).

Network frame bytes are not a packet schema, not a protocol, and not a session;
application layers own byte meaning above the SDK byte carrier.
