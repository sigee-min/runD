# Execution Topology

Scope: global execution topology, naming, identity, admission, public
namespace, and ownership boundaries for `kernel`, `accel`, `node`, `cluster`,
`shard`, `kernel run`, and `AccelKernel`.

This page is the repository-level architecture authority for how work moves
from global intent into local deterministic execution. It does not own kernel
packet scheduling, kernel workspace internals, kernel dispatch contracts,
kernel reductions, `KernelProgram`, or kernel telemetry truth. Those contracts
remain owned by [`/kernel/docs`](../../kernel/docs/README.md).

## Authority Path

1. [`/docs/README.md`](../README.md) owns repository-wide documentation rules.
2. This page owns cross-layer execution topology and naming.
3. [`/kernel/docs/README.md`](../../kernel/docs/README.md) owns every kernel
   contract consumed by this topology.
4. [`/accel/docs/README.md`](../../accel/docs/README.md) owns the internal
   backend-neutral execution contract consumed by the Node compute bridge.
5. [`/node/docs/README.md`](../../node/docs/README.md) owns the implemented
   local node runtime product boundary, including backend selection,
   workspace pool ownership, discovery, lifecycle, telemetry, and trace.
6. [`/cluster/docs/README.md`](../../cluster/docs/README.md) owns implemented
   cluster placement and retry behavior.
7. [`numeric.md`](./numeric.md) owns cross-layer numeric policy
   routing for execution surfaces.

Node code owns the product-grade local runtime boundary for admission,
workspace pool leasing, backend selection, topology/resource discovery, kernel
handoff, completion, release accounting, telemetry sink, and run trace. It
does not claim transport, node membership, or distributed failover authority.
Accel code owns internal backend-neutral value contracts and execution graph
authority; it does not observe host devices, create command queues, compile
Metal/Vulkan pipelines, or own scheduler/replay envelopes.
Cluster code owns placement and retry identity only.

## Core Model

runD is a library-first deterministic execution runtime/substrate. Its release
consumption boundary is the SDK artifact produced by `/package` from this
private source repository. It provides kernel, node, and optional cluster
execution substrate APIs that other engines, simulations, tools, and hosts can
embed. It is not a gameplay framework, world model, scene system, asset
pipeline, rendering stack, or game engine.

runD may be used by game-server frameworks, but it remains meaning-neutral.
Game concepts such as tick loops, player input ordering, world snapshots,
rollback, input delay, lag compensation, ECS storage, and gameplay protocol
schemas are domain-layer concepts and are not runD layers. runD owns only the
deterministic execution substrate: run identity, scheduler ordering, admitted
host inputs, canonical evidence, numeric policy identity, and replay checks.
The node network surface follows the same cut: runD owns move-only socket
lifetime, borrowed socket identity, one-shot readiness transfer, byte-level
receive/send results, host-event evidence, and replay mismatch detection; the
embedding domain layer owns sessions, player ids, packet schemas, protocol
policy, ticks, rollback, lag compensation, and game-server meaning.

An embedding engine or simulation may be a white-box domain framework above
runD, but it consumes runD as a black-box release artifact. The normative SDK
consumer boundary is
[`/package/docs/consumption.md`](../../package/docs/consumption.md).
Consumers use `find_package(runD 1.0.1 EXACT CONFIG REQUIRED)` and the exported
runD target. They must not vendor, submodule, or `add_subdirectory` the runD
source tree, and their public headers must not expose runD types or include
runD headers.

runD uses four durable execution layers:

| Layer | Public namespace | Meaning | Default presence |
| --- | --- | --- | --- |
| `kernel` | `rund::kernel` | Meaning-neutral deterministic execution substrate under `/kernel`. | Always present. |
| `compute SDK` | `rund::compute` | Typed public device, buffer, program, and graph UX translated by compiled Node sources. | Always present. |
| `accel` | internal `rund::Accel*` support | Backend-neutral deterministic execution contract under `/accel`; node supplies concrete backend resources. | Always present internally. |
| `node` | `rund::{Session,task,host,net,replay}` | One machine-level runD runtime member whose product surface is split by capability, not by implementation layer. | Always present as an implicit local node. |
| `cluster` | `rund::cluster` | Optional distributed policy plane over multiple nodes. | Present only in explicit multi-node mode. |

The sole public compute namespace is `rund::compute`. Applications do
not author Kernel IR, internal DSL nodes, Accel graphs, adapter picks, or raw
backend resources. Compiled Node bridge sources translate the typed SDK into
the existing Kernel/Accel authorities. Runtime, node, math, and cluster names
retain their documented owners; top-level aliases without the `rund` prefix
must not be added.

The implementation dependency direction is Cluster policy to the numeric
evidence declaration contract, and Session/Compute/Task/Host/Net/Replay
implementation to internal Accel and Kernel authorities. Cluster owns its
distributed identities and must not consume Runtime, Kernel, or Accel headers
for placement authority.

Two related concepts are not durable layers:

| Concept | Meaning |
| --- | --- |
| `KernelProgram` | Compiled deterministic law or plan owned by the kernel. |
| `AccelKernel` | Backend-neutral compiled accelerator graph contract owned by accel and executed through node-owned backend resources. |
| `kernel run` | A node-admitted execution instance that applies a `KernelProgram` to one admitted work meaning. |

`shard` is also not a machine-level layer. A shard is a logical work or data
partition that may be assigned to a node and admitted into one or more kernel
runs.

The default execution shape is:

```text
local process
  -> implicit local node
    -> admitted kernel runs
      -> kernel-owned deterministic execution
```

Explicit cluster mode extends the same model:

```text
cluster
  -> node A
  -> node B
  -> node C
    -> admitted kernel runs
      -> kernel-owned deterministic execution
```

Cluster mode must not create a second execution model. The cluster
surface adds shard-to-node placement records and retry identity checks around
the same node admission and kernel execution path.

Cluster exposure is governed by [Execution Surface Exposure](#execution-surface-exposure).

## Term Contracts

### Kernel

The kernel is the deterministic execution substrate under `/kernel`.

The kernel is meaning-neutral. It does not know whether packets represent game
chunks, scenes, physics cells, AI agents, financial records, or any other
domain object. A packet has only kernel-owned execution meaning: identity,
range, work unit, placement unit, dispatch unit, and ordered reduction input.

The kernel owns its substrate contracts through `/kernel/docs`, including:

- packet identity and packet count
- schedule and program compilation
- internal partition placement
- worker backend validation
- ordered reduction
- no-growth capacity proof
- dispatch result shape
- kernel telemetry truth

This page relies on those contracts but does not restate their detailed rules.

### KernelProgram

`KernelProgram` is a compiled deterministic law or execution plan.

A `KernelProgram` is not a shard, not a scene, not a node task queue item, and
not a cluster placement record. It is the kernel-owned compile output that can
be run under the kernel's checked workspace, dispatch, reduction, and telemetry
contracts.

Repeated program compiles are stable only for the same checked kernel input.
If a node or cluster changes the effective compile request, fold policy,
numeric contract, worker evidence, or capacity contract, it must treat the
resulting execution as a distinct run identity unless it has explicit
equivalence evidence. Strict-floating reduction policy is part of the effective
numeric contract when it affects admitted authoritative work.

### Kernel Run

A kernel run is the deterministic execution unit for one admitted work meaning.
It is not a durable layer.

A kernel run binds:

- admitted work identity
- shard identity, when the work came from a shard
- deterministic input identity
- logical time, tick, frame, epoch, or checkpoint boundary
- `KernelProgram` or the checked compile request that produces it
- fold policy and numeric contract
- node-provided resource envelope
- workspace reservation or capacity contract
- worker backend evidence consumed by the kernel
- kernel dispatch result
- kernel telemetry envelope
- output identity
- retry epoch, when retried

The work meaning belongs to the run envelope and the domain layer that created
it. The kernel remains meaning-neutral.

Generic example:

```text
shard: shard_42
tick: 1001
program: deterministic update
kernel run: shard_42 tick_1001 deterministic update
```

The same `KernelProgram` may be reused by many kernel runs. Those runs still
have different work meanings when their shard, input identity, tick, epoch, or
output identity differs.

### Node

A node is one machine-level runD runtime member.

`node` must not be used as shorthand for hardware NUMA topology. When the
hardware concept is intended, use `NUMA node` explicitly.

A node owns local admission and resource policy:

- deciding which assigned work is admitted now
- mapping assigned shard work to one or more kernel runs
- managing local thread, memory, workspace pool, and concurrency budgets
- constructing kernel `WorkerBackend` instances
- providing verified topology, affinity, and worker-capacity evidence when
  available
- choosing whether to present unverified facts as hints or to fail closed
- wrapping kernel outputs and diagnostics into node-level envelopes

A node does not own:

- `KernelProgram` packet order
- kernel partition placement
- kernel fold order
- `rund::kernel::Workspace` internal layout
- kernel no-growth predicates
- kernel dispatch contract truth
- kernel failure reasons
- kernel telemetry schema or truth fields
- global distributed ownership

Node resource policy may decide the resource envelope. The kernel proves
deterministic execution inside that envelope.

### Cluster

The cluster is the optional distributed policy plane over multiple nodes.

The cluster owns:

- shard-to-node assignment records
- placement epoch identity
- retry identity checks over `rund::cluster::RunKey` values

The cluster does not own:

- global job graph identity
- node membership or health
- global admission pressure
- failover, replication, or checkpoint policy
- cross-node observability or telemetry aggregation
- a single kernel run's packet order
- a single kernel run's fold order
- a node's local topology truth
- kernel telemetry truth
- kernel workspace capacity proof
- packet-level placement inside a `KernelProgram`

Cluster exposure and local absence rules are owned by
[Execution Surface Exposure](#execution-surface-exposure).

### Shard

A shard is a logical work or data partition.

A shard may carry:

- stable shard identity
- domain or world identity
- input identity
- output identity
- retry epoch
- checkpoint boundary
- placement constraints
- locality hints

A shard does not imply:

- kernel packet identity
- kernel packet order
- kernel fold order
- worker pickup order
- NUMA placement
- one-to-one mapping to a kernel run
- one-to-one mapping to a hardware NUMA node

Shard-to-run cardinality is node admission policy. One shard may produce one
kernel run, many kernel runs, or no run for a given tick. Many shards may also
be batched into one admitted run only when the resulting work meaning and
output identity remain explicit.

## Ownership Matrix

| Concern | Owner | Notes |
| --- | --- | --- |
| Packet identity | kernel | Owned only by kernel contracts. |
| Packet and partition placement inside a compiled program | kernel | Node and cluster must not become second authorities. |
| Worker backend construction and kernel parallel runtime provider | node | Kernel validates backend truth it consumes. Kernel `par()` consumes a thread-local scoped provider installed by node runtime scope; it does not own worker-pool lifetime. |
| Lightweight task scheduling | node | Node owns `spawn` task ids, scheduler operation order, runtime-aware park/resume primitives, channels, timer/IO observation evidence, task-worker lanes, structured joins, stack budget, and scheduler evidence. Kernel skeleton order remains kernel-owned. |
| Workspace pool lifetime and budget | node | Kernel owns workspace internals and no-growth predicates. |
| Workspace capacity proof | kernel | Node may request or budget capacity; kernel records proof truth. |
| Local resource envelope | node | Includes width, memory budget, and topology evidence. |
| Topology evidence | node | Unknown or unverified topology must stay unavailable or hint-only. |
| Shard identity | domain layer | Cluster may reference shard identity in assignment records, but it does not own shard meaning. |
| Shard-to-node assignment | cluster in explicit cluster mode; local host otherwise | Local mode resolves directly to the implicit local node without exposing cluster placement. |
| Shard-to-run admission | node | Local decision under node resource policy. |
| Retry identity check | cluster | Compares source and target `RunKey` values and records retry epoch. |
| Ordered reduction | kernel | Node may wrap envelopes, not redefine fold order. |
| Kernel telemetry truth | kernel | Node may add envelopes, not rewrite truth fields. |

## Placement Vocabulary

Use precise placement terms. Do not use unqualified `placement` when more than
one layer is involved.

| Term | Owner | Meaning |
| --- | --- | --- |
| cluster placement | cluster | Assign a shard or work group to a node. |
| node admission | node | Accept assigned work locally and bind it to a resource envelope. |
| topology projection | node | Convert verified local topology/capacity facts into kernel-consumable truth, hints, or fail-closed decisions. |
| kernel partition placement | kernel | Place packets and partitions inside a compiled `KernelProgram`. |

This vocabulary prevents cluster or node policy from becoming a duplicate
authority for kernel packet placement.

## Execution Surface Exposure

This section owns how local and distributed execution surfaces are exposed to
engine, simulation, tool, hosting, and operator code.

Domain and engine-facing code submits work in domain terms owned outside runD:
world, scene, shard or chunk, system, logical time, deterministic input
identity, priority, budget, and locality constraints are examples, not runD
framework objects. The same domain work submission surface is used for local
and distributed mode.

Without explicit multi-node hosting configuration:

- cluster is absent as an execution surface
- the local host resolves shard work directly to the implicit local node
- ordinary local game and simulation code must not call cluster placement APIs
  to reach that node
- node admission remains the local resource boundary before kernel execution

With explicit cluster mode:

- cluster consumes a shard reference, candidate node ids, and placement epoch
- cluster owns the selected shard-to-node assignment record
- cluster may evaluate whether a retry preserves node run identity
- node admission and kernel execution remain the same semantic path used by
  local mode

Cluster APIs are exposed to hosting, operations, diagnostics, explicit
distributed-server code, tests, and host bridges. Direct node selection, such
as "place this shard on node N", is an escape hatch for those surfaces only.

## Node Admission

Node admission is the boundary between domain work and deterministic kernel
execution.

Admission input may include:

- assigned shard identity
- domain job identity
- tick, frame, epoch, or checkpoint boundary
- deterministic input identity
- desired concurrency or width
- memory budget
- topology preference
- required no-growth behavior
- required fold policy or numeric contract

Admission validates the kernel-run identity and selects resource state, or
returns a fail-closed reason. When admission succeeds, the node has selected a
resource envelope and a kernel path. When admission fails, no kernel run
exists. These two existing owners remain separate; there is no combined run
envelope value or duplicate state projection.

Node admission must keep these boundaries:

- It may choose how many runs a shard produces for a tick.
- It may choose local concurrency and resource budget.
- It may choose a worker backend and workspace pool lifetime.
- It may not define kernel packet order.
- It may not rewrite kernel capacity proof.
- It may not promote unverified topology to kernel truth.
- It may not merge semantically different work into one run without explicit
  output identity and replay rules.

## Resource Envelope

A resource envelope is the node-owned local execution budget offered to a
kernel run.

It may contain:

- requested worker width
- selected worker backend
- memory and workspace budget
- no-growth requirement
- topology and affinity evidence
- worker capacity evidence
- local admission reason

The resource envelope is not itself a kernel proof. It is input to a kernel
compile/run path that may pass, fail closed, or record unavailable truth.

The resource envelope is node-owned input to admission and kernel execution.
It is not cluster retry authority.

## Workspace Boundary

Node owns workspace lifetime and pool policy. Kernel owns workspace semantics.

The node may:

- allocate or pool workspace storage
- install a kernel parallel-runtime provider with node-owned backend
  selection and workspace lifetime
- schedule lightweight cooperative node tasks inside a runtime scope
- decide which admitted run receives which workspace instance
- enforce local memory budgets
- decide when a workspace is removed from service or reused
- request no-growth behavior

The kernel owns:

- workspace internal layout
- reservation requirements
- no-growth capacity predicates
- capacity proof fields
- capacity failure reasons
- dispatch contract checks
- telemetry schema derived from workspace/program state

## Lightweight Node Tasks

Node owns lightweight task scheduling inside a runtime scope. A task is a
node-level synchronous-looking callable spawned by `task::spawn`, with
deterministic logical id, scheduler-owned operation order, runtime-aware
park/resume points, and structured join evidence. It is not a kernel packet,
kernel partition, shard, or admitted kernel run.

The task scheduler fixes node-owned primitive order with a sequenced operation
log and external timer/IO observation log. OS worker pickup, wall-clock timing,
external IO readiness timing, and physical worker id are not semantic
authority. `rund::kernel::each(rund::kernel::par(), ...)` inside a node task
consumes the scoped node provider and then uses kernel-owned skeleton
validation, partitioning, and fold order.

Host-facing random, logical time, environment, and syscall wrappers are local
runtime authority. They enter deterministic work through `rund::host` APIs and host
event evidence. Raw std/libc host calls are outside replay authority; a checked
continuation proof must either reject them or rewrite them to admitted
`rund::host` surfaces.

The node task surface is cooperative and node-owned. Public code submits normal
`void` callables to `task::spawn` and uses scheduler primitives such as `yield`,
`sleep`, channels, and low-level IO waits for runtime-aware park/resume. Public
code uses only documented scheduler APIs; internal scheduler authority,
runtime contexts, preemptive task migration, work stealing, priority
scheduling, network/file wrappers, `select`, and distributed task movement are
not public contracts.

`rund::kernel::Workspace` is the sole workspace-policy authority.

## Topology And NUMA

NUMA is node-local resource topology, not kernel semantic meaning.

A node may admit:

- multiple kernel runs on one hardware NUMA node
- one large kernel run across multiple hardware NUMA nodes
- separate kernel runs pinned or preferred to different locality domains
- no NUMA-aware placement when topology evidence is unavailable

There is no one-to-one rule between kernel runs and hardware NUMA nodes.

Topology evidence must be projected conservatively:

- Verified worker capacity may be passed as trusted capacity evidence.
- Unverified capacity must remain a hint or cause fail-closed admission when
  truth is required.
- Verified affinity may be reported as affinity truth.
- Unverified affinity must remain hint-only or unavailable.
- Unknown topology must stay unknown.
- Node estimates must not be written into kernel truth fields.

The kernel may consume worker-capacity and affinity truth through its existing
public contracts, but the kernel does not own NUMA semantics.

## Identity And Replay

Determinism requires identity, not only repeatable code shape.

A kernel run identity must be stable enough to answer:

- what admitted work was run
- which deterministic input identity was used
- which shard or domain partition it came from
- which logical time, tick, frame, epoch, or checkpoint boundary it covers
- which `KernelProgram` or checked compile request was used
- which fold policy and numeric contract were used
- which no-growth or capacity contract applied
- which retry epoch produced the output
- which output identity was committed

Retry preserves the same run identity only when these conditions are preserved:

- admitted work id is the same
- deterministic input identity is the same
- `KernelProgram` or checked compile request is equivalent
- fold policy and numeric contract are equivalent
- required no-growth/capacity contract is equivalent
- checkpoint boundary is the same
- output identity is the same or explicitly replaced by retry epoch

If any of these change without equivalence evidence, the retry must use a
distinct run identity.

Cluster retry identity consumes the source `rund::cluster::RunKey`, the target
`RunKey`, and the retry epoch. It preserves run identity only when the
keys match.

## Single-Node Default

runD defaults to a single local node.

Without cluster configuration:

- the local process creates or uses an implicit local node
- `rund::run` is the typed implicit-local-node facade for local runtime scope
  execution; file-backed IaC parsing is intentionally separate
- kernel `par()` consumes node resources only inside a node runtime provider
  scope
- shard work resolves directly to that local node
- node admission remains the only local resource boundary
- kernel runs use the same kernel compile/workspace/dispatch path used in
  cluster mode
- no external cluster membership, consensus, replication, or failover service
  is required

The single-node default is not a simplified semantic mode. It is the base case
of the same topology. Cluster absence from the local execution surface is owned
by [Execution Surface Exposure](#execution-surface-exposure).

## Cluster Mode

Cluster mode adds shard-to-node placement and retry identity checks around the
same node and kernel model.
Its selection and API exposure are owned by
[Execution Surface Exposure](#execution-surface-exposure).

Cluster mode owns:

- shard-to-node placement
- placement epoch identity
- retry identity checks over Cluster `RunKey` values

Cluster mode must not add:

- a second kernel execution path
- a required cluster API dependency for local game or simulation code
- cluster-owned packet order
- cluster-owned fold order
- cluster-owned kernel telemetry truth
- distributed retry that hides changed run identity
- membership, failover, replication, checkpoint, or telemetry aggregation
  authority

## Telemetry Envelopes

Telemetry must preserve ownership.

Kernel telemetry fields are kernel truth. Node may wrap them, but it must not
overwrite their meaning. Node telemetry envelopes own a kernel telemetry
snapshot; they must not retain borrowed pointers into caller-owned kernel
telemetry storage.

Use separate envelope layers:

```text
node telemetry envelope
  -> kernel telemetry truth
```

Node telemetry may wrap topology availability and backend construction
diagnostics without rewriting kernel truth.

## Execution Flow

1. A domain layer creates work with explicit identity, input identity, and
   logical time.
2. The work is assigned to a shard or work group.
3. In local mode, the host resolves the shard directly to the implicit local
   node without a cluster control surface. In cluster mode, the cluster assigns
   the shard to a node.
4. The node evaluates local admission against resource budget, topology
   evidence, and workspace policy.
5. Admission validates run identity, selects resource state, or fails closed.
6. The node supplies the worker backend and workspace policy to the kernel
   compile/run path.
7. The kernel compiles or views the `KernelProgram`, validates dispatch
   requirements, proves capacity when requested, executes, reduces, and emits
   kernel telemetry truth.
8. The node wraps the kernel result without rewriting kernel truth.
9. The cluster, if present, owns shard-to-node placement and retry identity
   checks; node admission and kernel execution remain unchanged.

## Invariants

- Kernel is meaning-neutral.
- `kernel run` is the unit that carries one admitted work meaning.
- `KernelProgram` is a compiled deterministic law, not a domain object.
- Node chooses the resource envelope.
- Kernel proves deterministic execution inside the envelope.
- Shard identity does not imply packet identity.
- Shard assignment chooses where work is admitted, not how packets are ordered.
- NUMA topology is node resource evidence, not deterministic semantics.
- Unknown topology remains unknown.
- Single-node default and cluster mode use the same node admission and kernel
  execution path.
- Retry preserves run identity only when admitted work, deterministic input,
  checked program, fold policy, numeric contract, capacity contract,
  checkpoint boundary, and output identity are preserved or proven equivalent.
- Node telemetry may wrap kernel telemetry but must not rewrite it.

## Non-Goals

This page does not define:

- kernel packet scheduling details
- kernel workspace layout
- kernel reduction primitives
- node implementation APIs
- cluster wire protocols
- engine-facing cluster placement APIs for local execution
- storage formats
- network consensus
- asset streaming
- rendering
- audio
- game engine or gameplay framework APIs
- domain gameplay rules

Those concerns need their own owning docs when implemented.

## Failure Modes

- Duplicate placement authority: node or cluster attempts to own kernel packet
  or partition placement.
- Ambiguous run identity: retry changes the Cluster `RunKey` but reports the
  same run identity.
- Workspace authority leak: node policy rewrites kernel capacity proof,
  dispatch contract, failure reason, or telemetry schema.
- Topology fabrication: unverified NUMA, affinity, or capacity facts are
  promoted to kernel truth.
- Telemetry rewriting: node telemetry overwrites kernel truth fields.
- Shard semantic drift: shard identity starts implying packet order, fold
  order, worker pickup order, or NUMA placement.
- Split execution model: local mode and cluster mode use different semantic
  execution paths.
- Cluster UX leak: ordinary game or simulation code directly controls
  shard-to-node placement outside explicit distributed hosting, operations,
  tests, or diagnostics.
- Cross-shard nondeterminism: direct shared mutable state makes node scheduling
  or worker pickup order part of domain meaning.

## Verification

Verification command and route ownership lives in
[Verification](./verification.md). Topology contracts cover:

- admission pass/fail decisions under fixed inputs
- resource envelope construction from concrete discovery payloads
- workspace pool lifetime across prepared-state reuse
- topology evidence truth levels and unavailable states
- built-in fixed-pool backend construction and provided-backend validation
- scoped kernel parallel-runtime provider enter/exit behavior
- telemetry envelope preservation of kernel truth fields
- telemetry sink and run trace behavior
- lifecycle transitions: configure, start, drain, stop, fail
- lock-backed single-writer runtime execution and same-runtime reentry rejection
- stress, fixed-seed fuzz, concurrency, and long-run execution

Kernel, Node, and Cluster owners are selected through that one generated route
authority; this page does not maintain target lists or CTest-name mirrors.

## Known Rejected Meanings

- `kernel` must not mean the work item.
- `kernel run` must not be treated as a durable layer.
- `shard` must not mean a machine-level runtime layer.
- `node` must not mean hardware NUMA topology.
- `cluster` must not mean the local dispatch helper inside `/kernel`.
- `cluster` must not mean the default local game or simulation execution API.
- unqualified `placement` must not cross cluster, node, and kernel boundaries.
- NUMA placement must not become deterministic work meaning.

## Update Rules

- Changes to global execution naming or layer ownership must update this page.
- Node or cluster behavior changes must create or update owning subsystem docs and
  link back here when the cross-layer boundary changes.
- Kernel contract changes must update `/kernel/docs`, not this page, unless the
  cross-layer topology boundary also changes.
- If a rule appears both here and in a subsystem doc, one page must be declared
  the owner and the other must link to it.
