# SDK Surface

This page is the authority for what an SDK consumer may include. Installed
consumer builds through `runD::sdk` are the executable proof of the surface;
source-tree subsystem headers are not a parallel public API.

## Direct Headers

[`surface/headers.tsv`](./surface/headers.tsv) is the machine-readable owner of
the direct entries and the paths that must remain private. Focused runD
domains are `session`, `storage`, `task`, `host`, `net`, `replay`, `evidence`,
and `compute`; `rund/rund.hpp` composes those entries without declaring a
type, alias, or behavior. Math32, Math64, and Cluster retain their direct
entries.

Each `rund/<domain>.hpp` header is the focused entry for its domain.
`<rund/compute/pipeline.hpp>` is also a registered focused direct entry for
prepared dependent Program execution. `<rund/compute.hpp>` deliberately
excludes it, while the all-domain `<rund/rund.hpp>` composition includes it.
Compute has three deliberate opt-in extensions. `<rund/compute/async.hpp>` completes the
existing deferred Flow's asynchronous terminal without making another graph
or compilation authority. `<rund/compute/math.hpp>` owns composite expression
functions and matrix, transform, factor, solve, and spectrum Flow stages.
`<rund/compute.hpp>` includes neither extension nor Pipeline, so ordinary
`on(...).map(...).collect()` consumers parse neither standard-future
machinery, the advanced math graph, nor Pipeline binding templates.
`<rund/compute/session.hpp>` alone completes `Session::compute(Job&)` and
`Session::compute(Pipeline&)` plus their coroutine admission types, so
standalone Compute consumers do not parse the
Session scheduler surface.

The stable programmable accelerator contract remains the existing functional
`Flow` surface reached from `<rund/compute.hpp>`. Bounded input typestate,
count-aware indexed lookup, stable bounded emit/worklists, and deterministic
`scatter_reduce` extend that one language; there is no second public Kernel,
shader, backend graph, native module, or callback header. Prepared resource
views and state publication stay in the focused
`<rund/compute/pipeline.hpp>` entry. The installed
`example/device/program.cpp` consumer proves both entries using only
`runD::sdk`.
Other focused entries do not include a sibling merely for convenience. In
particular, `<rund/host.hpp>` and `<rund/net.hpp>` remain separate because Host
declarations do not depend on the Network surface; a translation unit using
both includes both. `<rund/session.hpp>` retains only forward declarations and
does not pull Compute definitions into lifecycle-only code. `<rund/rund.hpp>` is the only
all-domain convenience composition.
`<rund/storage.hpp>` is the focused entry for shared hierarchical storage
admission. It accounts storage retained by participating owners; it is not an
allocator, filesystem, or Replay policy mirror.
`<rund/replay.hpp>` follows the same rule internally: it is a declaration-free
composition of single-owner replay support headers. Its storage policy depends
on the independent `rund::storage::Budget` declaration and therefore reaches
that type transitively; `<rund/storage.hpp>` remains the direct entry for code
using generic storage accounting without Replay. Replay support paths let
producer objects include only the declarations they implement, but are not
direct SDK entries and carry no path-stability promise.
Their behavior is owned by the
[Session](../../node/docs/contracts/runtime.md) and
[Telemetry](../../node/docs/contracts/telemetry.md),
[Storage](../../node/docs/contracts/storage.md), and
[Replay](../../node/docs/contracts/replay.md) contracts, and by the
[Compute](../../docs/reference/compute.md) reference and
[Pipeline](../../node/docs/contracts/compute/pipeline.md) contract; this page
does not keep a second list of their types, numeric laws, or telemetry
semantics.

Public outcome names and their single-authority rules are owned by
[API Stability](./api/stability.md). This page owns header reachability only;
it does not mirror result or replay semantics.

## Hard Boundary

Kernel, Accel, and direct Node header paths are internal implementation
authority, even when the artifact stages a transitive header needed by a
direct entry. They are not consumer entry points. Selected runtime, storage,
host, replay, and evidence names plus the `rund::task` task surface and
`rund::net` network surface reached through their focused entries or
`<rund/rund.hpp>` are public and documented in the wiki; this does not
promote their owning `node/...` support-header paths. The Kernel Compute
umbrella, DSL, graph schema and graph construction tree, Accel graph
construction, and direct Node execution headers remain source-tree authorities
and are not separate installed consumer languages. The package exports no
subsystem CMake target.

The direct-header set is the versioned public boundary. At install time,
the producer compiler resolves each direct entry as an independent translation
unit. The union of those transitive project-local include closures is the
entire physical header tree. The sorted installed inventory is sealed as
`share/runD/sdk-headers.tsv`, and external consumption requires the manifest
and tree to match exactly. A transitive support file is still not a direct
consumer entry point.

The direct entries do not promote Kernel IR, plans, bindings, lowering, Node
adapter types, backend dispatch objects, or platform SDK types. Cooperative
task values are owned only by `rund::task`; implementation namespaces and
headers remain private. Session values are owned by `rund`, Session-host
Compute values by `rund::compute`, host evidence by `rund::host`, network
operations and outcomes by `rund::net`, hierarchical storage accounting by
`rund::storage`, and Replay values and policy by `rund::replay`, as documented
by
[API Stability](./api/stability.md).

## Artifact Rules

The installed artifact contains exactly the direct set and the transitive
support headers required to compile it; no manually maintained broad copy plus
exclusion list is a second install authority. Consumers link only `runD::sdk`.

| Page | Owns |
| --- | --- |
| [`acceptance.md`](./acceptance.md) | Release-facing product UX and evidence gate. |
| [`api/stability.md`](./api/stability.md) | Current public API authority. |
| [`surface/artifact.md`](./surface/artifact.md) | Installed artifact boundary. |
