# Node Build Graph

This page owns Node production translation-unit ownership, component closure,
focused link profiles, and the relationship between internal test archives and
the installed `node` archive.

## Authority

1. `node/cmake/node/sources.cmake` owns component membership and the SCC
   condensation graph; files below `node/cmake/node/sources/` own their
   component source lists.
2. `node/cmake/node/profiles.cmake` is the sole profile table. It maps each
   profile to one SCC root, CPU projection, focus scope, and owner bound.
3. `node/cmake/node/contract/routes.cmake` is the sole group-to-target and
   group-to-resource table.
4. `node/cmake/node/library.cmake` creates one OBJECT target per component and
   assembles archives from those objects.
5. `node/tests/contract/cases.def` and its fragments own every case name,
   symbol, source owner, link profile, group, and verification tag.
6. `node/cmake/tests/contract/cases.cmake` parses the registry;
   `node/cmake/tests/contract/target.cmake` is the short target-orchestration
   owner and loads the responsibility leaves below `contract/target/`.

No filename convention, test body, generated cache, or parallel table may
infer any of these fields.

## Compilation Model

Every selected-platform `node/src/**/*.cpp` or `node/src/**/*.mm` file belongs
to exactly one component OBJECT target. The installed archive and internal
test archives contain unions of those objects; focused tests never compile a
production source through a second owner.

The complete `ACCEL_EXECUTION` SCC owns both native components. An exact
backend selection projects that existing component set before an internal
archive is materialized:

```text
N(full)    = {ACCEL_METAL, ACCEL_VULKAN}
N(cpu)     = {}
N(metal)   = {ACCEL_METAL}
N(vulkan)  = {ACCEL_VULKAN}
active(b)  = shared execution components union N(b)
```

`node/cmake/node/profiles.cmake` owns this mapping. The single compiled
`pick/catalog.cpp` owner receives matching catalog-membership definitions, so
its symbol edges equal `N(b)`; no generated catalog, unavailable stub, or
second backend picker is admitted. Product, Release, sanitizer, package, and
install configurations have no focused backend and therefore retain
`N(full)`. A focused tree is an internal verification tree, not an installable
single-backend SDK.

The `ACCEL_METAL` fragment owns its complete source list directly. Build setup
derives the Objective-C++ subset from that component list: only `.mm` owners use
Objective-C++ and ARC when the Metal SDK is present, while `.cpp` owners remain
C++. No other file owns a Metal source list.

The same component is compiled as ordinary C++ when the Metal SDK is absent so
the selected product archive retains its typed unavailable result without a
second stub library. Every Objective-C, Foundation, and Metal SDK declaration
is therefore admitted by the single expression
`__APPLE__ && RUND_NODE_HAVE_METAL_SDK`; source-private helper declarations use
that exact boundary as well. The no-SDK compile contains only portable C++
definitions and the unavailable result path. It may not import an Apple SDK
header, expose an incomplete native helper, or select another backend.

Let component `i` contain `n_i` translation units and let `closure(p)` be the
closed SCC set for profile `p`. Exact-profile production work is

```text
C(p) = sum(n_i for i in closure(p))
```

The product profile uses the full Node union. A broad test target takes the
union of its registered case profiles and selects the least SCC closure that
covers that union. Each distinct non-product closure has one internal STATIC
archive. Internal archives are not installed and are not SDK targets.

`node/tests/contract/main.cpp` owns argument parsing, backend selection,
timing, diagnostics, and ordered execution once in the
`node-contract-runner` OBJECT target. `dispatch.cpp` alone receives the case
definition, builds one constexpr table, and exposes it through
`case/table.hpp`. Changing a case table therefore rebuilds its thin dispatch
object and link, not the runner or semantic case sources. Registration has no
constructor path, alias, fallback lookup, or second table.

The runner also owns the executable backend domain. Its default domain is CPU
plus only the native SDK backends present in that target's compile context;
an exact backend selection projects that domain once. Every execution parity
inventory consumes `selected_compute_backends()`, and accelerator-only
inventories consume its `selected_accelerators()` subspan. A semantic case may
name a backend literal only when it is testing that backend's unavailable or
metadata contract without claiming execution parity. No executable case owns
a second CPU/Metal/Vulkan list.

The Runtime-base groups `runtime`, `runtime.task`, `runtime.task.host`,
`runtime.task.net`, `runtime.task.reactor`, and `runtime.task.replay` share the
`node-runtime` executable. Their ordered CTest rows preserve process
isolation, labels, parallel scheduling, and first-failure reporting. Focused
configuration retains the profile-declared one- or two-owner materialization
cohort plus owner-neutral helpers while using the same target and runner path.

Three owner-neutral test implementation units also have one OBJECT compile
owner each: Runtime product helpers, Compute allocation accounting, and Task
allocation/failure accounting. An executable that needs a process-global
allocation override links the same selected object; mutable state is still
isolated by process.

Verification-tag subsets do not recompile a semantic source that is already
required by their canonical routed executable under the same compile context.
The registry tag is the sole cohort authority: the group owner derives each
`(profile, resource, ordered cases)` cohort once, and both tagged target
construction and object sharing consume that result. When all cases in one
cohort route to one primary executable and the effective SCC root is
identical, their exact selected source intersection has one OBJECT owner.
Both the primary and tagged executable consume those objects; their dispatch
objects remain distinct because their case tables differ.

Configuration records the case/platform/root compile-context token and defers
a fail-closed comparison of definitions, normal and system includes, options,
features, linked usage requirements, C++ policy, interprocedural policy, and
precompiled-header state for the shared owner and both consumers. A differing
root, backend projection, platform, target-local definition, or link usage
therefore keeps separate compilation rather than silently reusing an
incompatible object. The derived source-to-object index lives only in the
CMake generation state; there is no checked-in source mirror or second tag
list. Both neutral-support and shared-source lookup use one SHA-256 property
key per canonical source/context, so materialization is `Theta(S)` rather than
rescanning a row list for each of `S` target sources.

Assertion failure has one repository test owner outside Node. Node contract
translation units consume the declaration-only `tools/test/assert.hpp`
boundary, and every Node test executable links the one compiled
`rund-test-assertion` implementation. No Node-local assertion header or inline
stdio implementation exists.

## Dependency Boundaries

Public schema edits keep their natural reverse dependency set. Private
implementation edits must depend only on the leaf that owns the changed state
or algorithm.

- `node/src/compute/type.hpp` solely owns private `Type` byte-width, validity,
  Kernel scalar-width, and arithmetic-domain projections. The functions are
  constexpr inline and every consumer includes the leaf directly.
  `device/state.hpp` owns Device and Buffer storage only and does not
  re-export type semantics. `map/compile` and `graph/build/primitive` consume
  `type_domain()` rather than retaining local domain switches.
- `node/src/compute/graph/scan.hpp` solely owns the constexpr
  `Type -> kernel::ScanElement` projection consumed by graph description and
  graph compilation. Neither path retains a local switch, so a type-table edit
  has one projection authority and cannot silently change only one graph hash
  path.
- `node/src/compute/cpu/view.hpp` exposes the narrow CPU View result and
  `node/src/compute/cpu/view.cpp` solely owns the overflow-free byte-footprint
  proof and host-pointer projection. Map, bounded control, collectives,
  primitives, reset, staging, and Pipeline publication include the leaf
  directly; Device and Job state do not re-export it.
- `node/src/runtime/task/scope/frame.cpp` owns scope admission, provider and
  scheduler restoration, exception containment, drain, and report capture.
  Public `Session::scope` contains only callback validation and a borrowed
  invocation thunk.
- `node/src/compute/run/state.hpp` is the sole complete `RunState` owner.
  Public `Run` uses fixed inline erased storage; lifetime and execution bridges
  live in `node/src/compute/run/value.cpp`.
- Operation-named headers below `node/src/accel/kernel/bindings/` own their
  individual binding schemas. `build.hpp` is the intentional full-variant
  assembly owner.
- `node/src/runtime/task/scheduler/state.hpp` owns the private Scheduler
  declaration and opaque state pointer;
  `node/src/runtime/task/scheduler/state/storage.hpp` solely owns the complete
  `SchedulerState` layout. Facade bridges that invoke Scheduler methods include
  only the declaration owner. Implementations that dereference state include
  the storage owner, so a storage-leaf edit does not invalidate declaration-only
  host, network, Runtime, and task bridges.
- Complete scheduler values live in the one-word ownership leaves below
  `node/src/runtime/task/scheduler/state/model/`. No model aggregate exists.
  C++ incomplete-container lifetime is closed by compiled constructors and
  destructors in the ready, lane, reactor, and Host I/O owners; the vector and
  `unique_ptr` representations, capacities, scheduler layout, and hot calls
  are invariant under physical ownership partition. In the focused Ninja
  dependency database, the largest
  complete ownership leaf, `model/task.hpp`, reaches 84 objects; the remaining
  leaves reach at most 33, and the Host I/O slot reaches 4. These are
  manifest-local measurements, not ABI constants, and must be regenerated from
  Ninja depfiles after graph changes.
- Completion identity, lease, slot, and waiter layout live in
  `task/completion/model.hpp`. Scheduler declarations and task records consume
  only that model. The callable `ResultRef` layout and the spawn result pair
  live independently in `rund/task/handle/ref.hpp` and
  `rund/task/handle/spawn.hpp`; concrete owners include those leaves without
  importing typed result construction. The complete pool enters scheduler
  storage, while typed result construction and observation enter only
  `task/completion/value.hpp`. An edit to the generic outcome implementation
  therefore has no transitive edge through scheduler state, completion
  storage, or `TaskRecord`; only translation units that construct or observe a
  typed result include that template owner. The same-manifest direct-header
  closure changed from 149 to 144 local headers for `scheduler/state.hpp`,
  from 7 to 5 for `task/completion.hpp`, and from 11 to 6 for
  `state/model/task.hpp`; all three no longer reach
  `rund/outcome/result.hpp`. These are dependency-graph counts, not wall-time
  claims.
- The Scheduler declaration forward-declares `ScopeEvidence`, the Host I/O
  operation/result types, `io::Fd`, and `random::RunSeed`. Complete storage
  includes only `rund/host/io/fd.hpp` and `rund/host/random/seed.hpp`; the full
  Host I/O and random operation aggregates remain confined to their compiled
  operation owners. `runtime.hpp` and the private Session result access
  declaration also forward-declare `ScopeEvidence`; only the scheduler
  snapshot and scope report owners import its complete definition.
- `<rund/task.hpp>` reaches task-owned `IoOp` and `IoResult` through
  `runtime/task/results.hpp`; it has no edge to the Host file-I/O surface.
  Editing `rund/host/io.hpp` therefore cannot dirty a task-only consumer.
- Node production and semantic-contract sources never include the public
  `<rund/task.hpp>` or `<rund/rund.hpp>` aggregates. They include the smallest
  top-level task owner required by the translation unit: `task/api.hpp`,
  `task/await.hpp`, `task/cancel.hpp`, `task/channel.hpp`, or `task/group.hpp`.
  The installed package consumers and installed scheduler measurement retain
  `<rund/task.hpp>` as the public SDK-entry proof; no test-only task aggregate
  exists.
- `runtime/reactor/readiness/state.hpp` owns readiness storage values,
  `mask.hpp` owns bit encoding and matching, and `handle.hpp` owns checked
  public/native conversion.
- Test-only Socket helpers live below `src/host/net/test/`. Each semantic
  source includes only `socket.hpp`, `ticket.hpp`, or `view.hpp` when it uses
  that capability. `view.hpp` alone reaches complete registry storage; no
  suite-wide forced include exists. Owner-neutral allocation accounting and
  unrelated Compute, Runtime, Task, Host, Reactor, and Replay contracts have
  no dependency edge to these test-only owners.
- Expression contract support imports only its semantic leaves.
  `support/values.hpp` owns fixed-value fixtures only;
  `support/record.hpp` and `support/single.hpp` own their Flow and expression
  function dependencies, while `part/cardinality.hpp` owns its narrower core
  expression dependency. Each support header is self-contained and a Compute
  product-surface addition cannot dirty these contracts unless they consume it.
- `rund/compute/backend.hpp` solely owns backend identity and observation
  values. `rund/compute/target.hpp` is the only public backend selector.
  Compute statistics, device information, and telemetry events consume that
  leaf directly; they have no edge to the unrelated operation vocabulary in
  `rund/compute/ops.hpp`. The operations header retains the backend leaf as its
  public prerequisite, but the dependency edge is one-way.
- `<rund/compute/expr.hpp>` is the thin public expression aggregate. The
  acyclic support chain below `rund/compute/expr/` preserves its declaration
  order: `model` owns declarations and symbolic nodes, `static` and `function`
  own compile-time expression construction, `value` owns `Expr` and
  `Predicate`, `record` owns typed field structure, `operation` owns runtime
  operations, `recipe` owns static materialization, and `select` completes the
  ordered selection overload set. Internal Flow and composite-function owners
  import `recipe` or `select` directly; they do not import the aggregate.
  Physical ownership changes without adding an alias, operation definition, or
  alternate expression authority.
- `<rund/compute/flow.hpp>` is the thin public Flow aggregate. Its support
  authority is partitioned by responsibility below `rund/compute/flow/`:
  `model` owns shared declarations, `stage` owns typed-stage composition,
  `zip` and `relational` own cross-input structure, `branch` owns conditional
  composition, `recipe` through `bounded` own deferred and bounded plans,
  `builder` owns Flow construction, and `resident` owns resident execution.
  Stage operations are grouped below `flow/stage/` by one-word capability.
  Every support header is a valid direct include. The default aggregate's
  support graph is acyclic; the optional `math` and `matrix` extension leaves
  bootstrap from the complete default Flow contract. The physical split
  changes edit invalidation ownership, not the public API, calculation graph,
  or default include closure.
- The leaves below `rund/compute/abi/` separately own the shared model, device
  and buffer bridge, expression bridge, job result materialization,
  observation bridge, graph bridge, and Flow bridge. There is no ABI aggregate
  include: product and private state headers must import the one leaf they
  consume. Moving a declaration between leaves must not add an alias,
  alternate implementation, runtime dispatch, or change public layout. The
  build measurement records every leaf's live reverse object reach so a hidden
  aggregate dependency cannot silently return.
- `rund/compute/pipeline/shape.hpp` owns the bounded leaf, transfer, binding,
  and resource dimensions used by Pipeline templates and compiled planning.
  Private plan, admission, and claim owners reach those dimensions through
  their narrow state boundary; they do not include the complete
  `rund/compute/pipeline.hpp`
  product aggregate merely to obtain a capacity. A facade edit therefore does
  not invalidate those private algorithm units unless it changes a leaf they
  actually consume.
- Pipeline arena declarations name `DeviceState`, `PipelineBuildStep`, and
  `PipelineMemoryPlan` through incomplete interface types. The complete
  Pipeline state enters only the implementation units that inspect it.
  Alignment and storage bounds likewise depend only on the compiled memory
  arena declaration. Recurrence-view comparison remains one inline owner in
  `plan/compare.hpp`; the projection and hash interfaces do not re-export the
  Pipeline state aggregate.
- `node/src/compute/program/output.hpp` is the narrow inline owner for logical
  output count and logical-to-physical index projection. Program
  introspection, cached execution, Job readback, and Pipeline planning consume
  that leaf instead of mirroring the conditional alias lookup. Its inline form
  preserves the original constant-time branch with no call, allocation, or
  copy while limiting a projection-law edit to those consumers.
- `node/src/accel/vulkan/command/model.hpp` is the sole native Vulkan
  command-state schema. The narrow `resources.hpp` interface and compiled
  `resources.cpp` owner create and destroy
  adapter-ring primaries, immutable Kernel secondaries, and reusable Pipeline
  primaries from one closed lifecycle plan. Kernel and Pipeline state retain
  that schema instead of mirroring pool, buffer, and fence fields. Timestamps
  remain outside it because slot timing and opt-in Pipeline profiling have
  different owners. Lifecycle algorithm edits therefore compile one Vulkan
  production object; the adapter ring's hot reset/record/submit path remains
  directly compiled in its existing command owner. In the current focused
  Accel depgraph the stable state model reaches 167 objects, while the lifecycle
  interface reaches 4 and the implementation source produces exactly 1 object.
  Regenerate these counts from Ninja depfiles rather than treating them as an
  ABI promise.
- Metal and Vulkan resident registries are opaque behind each backend's
  `resident/state.hpp`. Stable lookup rows contain only type-erased lifetime
  tokens and a native-buffer view; concrete owners and Vulkan native storage
  live in `resident/storage.hpp`. Each backend's batch lookup is one compiled
  implementation, not an inline hash-table copy in every primitive. In the
  current complete development depgraph, Metal concrete storage, resident
  state, and adapter state reach 2, 9, and 94 object files; the corresponding
  Vulkan counts are 2, 25, and 143. A concrete storage-layout edit therefore
  invalidates 97.9% fewer Metal objects and 98.6% fewer Vulkan objects than an
  adapter-state edit. Regenerate these counts from Ninja depfiles rather than
  treating them as an ABI promise.
- `node/src/accel/resident/model.hpp` is the sole resident identity, shape,
  capability, and owner-state schema shared by native registry entries.
  `resident/validation.cpp` is the sole descriptor and registry validation
  implementation. Backend lookup headers pass borrowed scalar state to that
  compiled boundary; they do not import kernel binding validation or duplicate
  its decision tree. Editing the validator implementation therefore compiles
  one production object plus closure links. `buffer/local.hpp` files do not
  re-export resident validation, reference, result, or usage headers into
  unrelated access, pool, and statistics owners.
- Metal and Vulkan executable-cache indices are likewise opaque compiled
  owners. Adapter state retains one pointer to each index; hash-table layout,
  bucket policy, and collision traversal stay in the cache leaf. Changing the
  lookup implementation must not invalidate every primitive that borrows an
  adapter merely because they share the same native device.
- `node/src/accel/vulkan/shader/module.{hpp,cpp}` is the sole Vulkan
  shader-module creation, move, counter, and destruction owner. Map and
  collective pipeline builders borrow its handle only for synchronous
  executable creation; cached executable state contains no module handle.
  Shader compilation, executable cache lookup, and pipeline layout remain
  separate owners.
- Native parameter layouts shared by Metal and Vulkan are owned by the
  operation, not by either backend. The `model.hpp` leaf under Gather,
  Histogram, Partition, Scatter, Segmented, or Stencil contains the sole C++
  host type; each backend local header imports it directly. The admitted
  layouts are:

  | Model | Field offsets in bytes | Size | Alignment |
  | --- | --- | ---: | ---: |
  | Gather | `0, 8, 16, 20` | 24 | 8 |
  | Histogram | `0, 8` | 16 | 8 |
  | Partition | `0` | 8 | 8 |
  | Scatter | `0, 8` | 16 | 8 |
  | Segmented Scan | `0, 8, 16, 24, 28` | 32 | 8 |
  | Stencil | `0, 8` | 16 | 8 |

  C++ `u64/u32`, Metal `ulong/uint`, and Vulkan
  `std430 uint64_t/uint` have the corresponding 8/4-byte scalar widths and
  natural alignments. Static assertions prove the complete host layout; the
  focused source contract requires the exact shader field sequence once in
  each generated source. The runtime uploads exactly `sizeof(P)` bytes through
  the same call. Shared parameter models add zero allocations, copies,
  branches, or uploaded bytes. Parameter models do not enter graph identity,
  lowering identity, or Program fingerprints.

For include graph `G` and changed header `h`, the scheduled
translation units are its reverse-reachable set `R(h)`. A leaf edge keeps
`R(h)` limited to consumers of that owner; an aggregate would replace it with
the union of unrelated owners.

Socket test access is owned by the explicit narrow `socket.hpp` leaf. Complete
registry storage remains behind `view.hpp`; no suite-wide forced include or
public aggregate may expose it. Production and semantic-contract sources do
not include `<rund/task.hpp>` or `<rund/rund.hpp>`. Build evidence for these
boundaries is regenerated from the current manifest through
`tools/measure/build/run`. Its live depfile rows include the Compute backend
identity and telemetry-event leaves so an operation-vocabulary regression is
visible as an increased reverse-reachable set.

The public `Run` inline-store size and reserve are owned by the
[Compute Memory contract](../compute/memory.md#program-host-ownership). This
page owns only the include-graph consequence: the private `RunState` remains
behind compiled leaves, and no public aggregate is introduced to expose its
layout. `compute.reuse` proves the frozen footprint and allocation contract.

Program convenience-execution templates and their compiled bridge declarations
live in `rund/compute/program/run.hpp`. Shared ABI consumers import only their
exact leaf under `rund/compute/abi/`. The build measurement route reports the
current reverse-reachable object set for both boundaries.

## Component Graph

The source graph condenses to:

```text
RUNTIME_PRODUCT (zero-object join)
  +--> CPU_RUNTIME_PRODUCT
  |      +--> RUNTIME_BASE ----+
  |      +--> CPU_COMPUTE -----+--> STORAGE
  +--> COMPUTE_EXECUTION
         +--> CPU_COMPUTE --> CPU_SIMD
         +--> ACCEL_EXECUTION --> CPU_ACCEL --> CPU_SIMD

CPU_COMPUTE --> TELEMETRY + WORKER_BACKEND
RUNTIME_BASE --> NUMERIC + TELEMETRY + WORKER_BACKEND
```

- `NUMERIC` owns numeric evidence encoding and validation.
- `STORAGE` owns the public hierarchical Budget/Reservation implementation as
  one dependency leaf shared by Compute admission and Runtime storage users.
- `CPU_SIMD` owns the CPU capability/SIMD translation units required by typed
  CPU Compute.
- `CPU_COMPUTE` adds typed backend-neutral and CPU Compute plus shared
  `STORAGE` and `WORKER_BACKEND`; it contains no native accelerator source or
  link edge.
- `CPU_ACCEL` adds backend-neutral Accel, the CPU adapter, and context owners.
- `ACCEL_EXECUTION` adds catalog composition, Fake, Metal and Vulkan.
- `COMPUTE_EXECUTION` adds native Compute adapter owners.
- `RUNTIME_BASE` owns host, replay, lifecycle, telemetry wrapping, topology
  projection, Kernel execution, Scheduler, and the selected platform; it
  consumes but does not own the shared `STORAGE` implementation.
- `CPU_RUNTIME_PRODUCT` adds the CPU NodeHost Compute bridge.
- `RUNTIME_PRODUCT` is an INTERFACE join over CPU Runtime and complete Compute;
  it adds no object.

Backend-neutral buffer, transfer, compile, run, resident Job, and memory code
cross one immutable `BackendOps` table sealed into the authenticated backend
pick. Its indirect cost is `O(1)` per admitted operation or prepared run, not
per element or tile; for `N` elements, its per-element contribution is
`O(1/N)`.

External linkage follows direct graph ownership. Internal CPU archives do not
link public Accel or the complete Kernel
archive. The Kernel compute, dispatch, and execution views are pairwise
disjoint; their exact union is the Compute Kernel closure, while Runtime uses
the complete Kernel closure. Configuration derives and verifies both unions
from their source owners. Dispatch and execution are also one declared
static-link SCC because prepared program execution calls dispatch and dispatch
records into execution-owned core/schedule state. CMake rescans that pair on
one-pass linkers; no object or source owner is duplicated.

Vulkan discovery uses only CMake's imported `Vulkan::Vulkan` target. When that
target is absent, Vulkan is unavailable. Metal retains its platform
framework/library owners. Installed package discovery resolves declared
dependencies again for the consumer and exports no build-tree path.

The base Runtime has no hard symbol edge to the NodeHost Compute bridge. The
bridge installs its close and retirement operations when Runtime Compute is
selected, so task, host, network, replay, and Kernel-only programs do not pull
public Compute or GPU implementation objects into their link graph.

## Link Profiles

| Profile | Exact-case use | SCC root | Installed archive |
| --- | --- | --- | --- |
| `header` | Header/filesystem contract | none | no |
| `numeric` | Numeric evidence codec | `NUMERIC` | no |
| `cpu-simd` | CPU capability and direct SIMD | `CPU_SIMD` | no |
| `cpu-accel` | Backend-neutral Accel, CPU adapter, and context | `CPU_ACCEL` | no |
| `accel` | Full Accel execution without public Compute | `ACCEL_EXECUTION` | no |
| `cpu-compute` | Typed CPU Compute | `CPU_COMPUTE` | no |
| `compute` | Public Compute and accelerators | `COMPUTE_EXECUTION` | no |
| `runtime` | Runtime without public Compute/Accel execution | `RUNTIME_BASE` | no |
| `cpu-product` | Runtime plus CPU NodeHost Compute | `CPU_RUNTIME_PRODUCT` | no |
| `product` | Complete installed Node product | `RUNTIME_PRODUCT` | yes |

The profile table owns the only backend projection. The backend-tagged
`compute.flow`, `compute.collective-modes`, and `compute.pipeline` cases accept
`cpu|metal|vulkan`: CPU projects `compute` to `cpu-compute` and removes
the accelerator resource, while native accelerators retain the canonical
Compute SCC root and resource but materialize only the selected native
component plus the shared and CPU oracle components. Selection never creates
a second source, component, catalog, or semantic owner. `compute.flow-primitives` is a native
`cpu-compute` row in the unlocked `node-compute` owner, not a projected
accelerator case. CPU-only targets must have no undefined native backend symbol
and, on Darwin, may link only `libc++` and `libSystem`.

Standalone Compute has two process owners over one semantic registry. The
unlocked `compute` group links `node-compute` through `CPU_COMPUTE`; the
accelerator group links `node-compute-accel` through `COMPUTE_EXECUTION` and
owns CPU-oracle comparison with Metal and Vulkan. Test translation
units are disjoint except for the allocation owner needed independently by
both processes.

The `compute.pipeline` case has one registry owner at
`tests/contract/compute/pipeline.cpp`. Its semantic contracts are separate
one-word translation units below `tests/contract/compute/pipeline/`; they link
into the same executable and expose no additional registry row. Shared
templates and compiled oracles have one `local`/`helpers` authority. A
semantic-leaf edit therefore rebuilds that leaf and relinks the existing
runner, while a shared-oracle edit intentionally invalidates its consumers.
No contract result depends on leaf initialization or execution order.

`compute.flow`, `compute.graph-services`, `compute.memory`, `compute.backend`,
`compute.bounded-parity`, and `compute.collective-modes` use the same
registry-runner boundary. Their registered source retains the only case row;
semantic units live below the matching one-word folder. Each leaf owns a
distinct surface, graph, resource, cache, memory, movement, bounded,
collective, numeric, or backend assertion. `local`, `model`, and the
fixed-result header are the only shared fixture and typed-oracle authorities.
Editing a semantic leaf therefore compiles that object and relinks its existing
accelerator runner. Linking all leaves into one case preserves the
domain/backend sequence and cannot create initialization or execution-order
dependencies.

The `runtime.task.host-io` case follows the same physical rule. Its registered
`host/io/run.cpp` owner is a thin ordered runner; `surface`, `admission`,
`replay`, `order`, and `signal` are independent semantic translation units, and
`fixture` is the sole temporary descriptor/path owner. Each semantic owner
creates and releases its own resources. A leaf edit therefore compiles that
leaf and relinks `node-runtime`; it does not recompile unrelated Host IO
semantics or introduce another registry row.

The Flow primitive semantic oracle belongs to the unlocked group. Its source
edit and exact execution therefore cannot dirty, link, lock, or open the
accelerator closure. Flow composition parity, collective parity, and boundary
parity remain in the accelerator group as the single physical backend
authorities.

Expression families live in separate implementation translation units.
Shared result construction and oracles live below
`compute/expression/support/`; semantic parts live below
`compute/expression/part/`. The 32- and 64-bit linear instantiations are
separate because each is independently expensive, while small fixed core
formats share one owner. Debug builds retain source-level line tables without
emitting recursively expanded template-local debug graphs.

## Focused Loop

Before configuration, `tools/test/run` consumes a sealed catalog produced only
by the canonical registry parser, profile projection, and route table. A
catalog cache hit rehashes every registry fragment plus the parser, profile, route, and
catalog-contract inputs. One validation pass checks the complete grammar while
projecting the input hashes and owner paths used by the independent byte and
existence checks; there is no second grammar or projection parser. Any missing
byte, input mismatch, malformed row, or seal mismatch regenerates the catalog
atomically through the canonical parser. The catalog is derived lookup data,
not an admission authority. A valid exact row
determines target, resource, profile, canonical materialization anchor, and
`.cache/focus/<profile>`. Focused CMake evaluates the same owners and emits an
index that must reproduce the requested route before and after Ninja
regeneration.

Harness acceptance compares the complete canonical list with the complete
catalog in one operation. It retains no sampled case-to-route literals. One
full-list proof plus the regex and unsupported-backend negative boundaries
form the complete live-root query set.

Focused configuration leaves Node-versus-Accel selection to the canonical case
registry while omitting unrelated Math, Kernel, and Cluster contracts from the
generated build tree. This does not remove a production dependency: the
selected profile still closes over the same Node, Kernel, and math object graph
required by its link root. A backend-capable exact route additionally projects
the native component set at that root. A cold Metal route has zero Vulkan
object producers and a cold Vulkan route has zero Metal object producers; the
CPU route closes at `CPU_COMPUTE` and has neither. Switching native backends in
the shared profile tree may leave unreferenced files on disk, but the live
Ninja target graph and archive inputs contain only the newly selected
projection.

Each profile has one mutable build tree and lock. Cases in one profile reuse
its production objects and serialize mutation; different profiles may build
concurrently. The canonical profile table also owns focus scope. Most profiles
retain one exact owner. The two-case `product` profile retains both owners and
one dispatch table because both cases share exactly one target, resource,
source suite, and link closure; its declared bound is two, so a third case
fails admission instead of silently widening the exact cold graph. Profile
scope rejects backend-selectable cases because their projected membership may
differ. The public runner sends only the requested case to the executable.
Execution selection is exact, and switching within the bounded profile cohort
performs no dispatch rebuild or product relink. An exact route
does not configure or mutate `.cache/dev`.
`.cache/case/catalog.tsv` is the sole disposable query acceleration artifact;
it cannot admit a case because the generated CMake index is still compared
after regeneration. A catalog miss parses twice to prove a stable input
identity, but does not change the cold build graph: the selected case owner or
bounded profile cohort, neutral helpers, production closure, dispatch
object, and link remain the same.

After that build and route revalidation, a separate successful-result cache
may skip direct exact-case execution only when the generated index, route,
backend projection, process runner, host, and executable bytes are identical.
Accelerator-resource cases always execute. Failed cases never populate this
cache, and `--fresh` bypasses it. This state is local edit-loop acceleration,
not semantic or evidence authority.

The configuration stamp covers source/build roots, profile, build type,
materialization anchor and route, caller arguments, tool/helper identities,
CMake command/version, and
the live `CMakeCache.txt`. Its topology witness is the sorted set
`T={(path, kind)}` for every live Node `.cpp` and `.mm`; add, delete, rename,
or file/symlink/other replacement changes `T`, while a content-only edit leaves
`T` unchanged for Ninja and compiler depfiles. An exact match skips only the
explicit configure; Ninja owns CMake-input regeneration. The
unfocused graph-sync path and the exact continuation both invoke the
configured repository Ninja owner directly. The exact route writes the stamp
only after build and route revalidation succeed.

Every registry row names one canonical `tests/contract/` source. It must exist
and occur exactly once in both its suite and routed target. For CMake graph
inputs `G`, case body `B_i`, and routed link `L_t`:

```text
change(G)   => regenerate + affected compile/link work
change(B_i) => compile(B_i) + L_t
```

Split semantic bodies have one additional checked-in authority:
`RUND_NODE_TEST_COMPANION_ROWS` maps each companion source to one or more
registered cases. Both exact-focus and tagged-subset selection call the same
resolver. Configuration validates and indexes the table once by canonical
source path; selection does not repeatedly scan the row set. A mapped source
is selected exactly when the requested case set
intersects its declared case set; a registry owner is selected exactly when
its case is requested; an unmapped non-owner retains the existing global
support meaning. For requested cases `Q`, source `s`, registered owner
function `owner(s)`, and companion set `companion(s)`:

```text
select(s,Q) =
  companion(s) intersects Q       when companion(s) is nonempty
  owner(s) is in Q                when s is a registry owner
  true                            otherwise
```

Configuration rejects malformed or duplicate companion paths, unregistered
sources, case-owner duplication, empty or duplicate case lists, unknown case
names, and any companion whose declared case belongs to another source suite.
Companion membership is explicit and fail-closed; path prefixes and filename
conventions never infer it. The Map, static Matrix, and two numeric
Flow cases are the first declared cohorts. Numeric Flow deliberately maps its
complete leaf set to both cases because one compiled model owns both ordered
runners; duplicating that model would create a second execution authority.

For a tagged cohort with `S` selected semantic sources whose primary and
tagged compile contexts are equal, the test compile action count is

```text
compile_actions = S
```

The live `thread` cohort has `S = 20`: Runtime storage, Runtime stress and its
lifecycle companion, plus the network-limits owner and companions. The shared
closure therefore has 20 TSan compile actions. The `telemetry:detail` cohorts
share their Runtime and CPU-product semantic owners. These are graph counts,
not latency claims; wall time depends on compiler cache state, parallel
scheduling, and the production closure.

Contract target construction is physically partitioned below
`node/cmake/tests/contract/target/`. `source` owns canonical relative paths and
owner counts; `companion` owns companion validation and source selection;
`suite`, `focus`, and `subset` own their respective case projections;
`support` owns neutral object rows and derived same-context tagged objects;
`context` owns the one shared compile context plus role-specific flags;
`runner` owns the runner and dispatch objects; `build` materializes
executables; and `closure` verifies the selected SCC link. The root
`target.cmake` only loads those owners in dependency order. The support table
and profile/backend selector each have one derived authority.

Source admission canonicalizes each candidate once into a function-local
SHA-256 count index. Suite, focus, and tagged-subset owner checks query that
index instead of normalizing and scanning the complete source list for every
case. For `S` candidate sources and `C` case owners, path projection and owner
counting therefore require `Theta(S + C)` operations rather than
`Theta(S * C)`. The index is discarded when the selecting function returns;
the registry, companion table, and suite source property remain the only
retained authorities.

Case bodies are ordinary compiler inputs, not configure dependencies. Adding
or removing a source changes the admission glob and regenerates the graph;
editing a body does not schedule CMake. Switching cases within one profile
does not change production commands. Within a profile-scoped focus it also
does not change the dispatch command, so with unchanged sources and depfiles:

```text
dirty(object) = changed(command | explicit inputs | discovered dependencies)
same profile + unchanged production inputs => 0 dirty production objects
```

A translation unit must own a distinct assertion, failure, backend, resource,
or scheduling boundary. A `.cpp` that only includes implementation headers is
not an admitted semantic owner; its assertions belong to the nearest owner in
the same target.

Native collective implementations follow that rule directly. Metal
compact, gather, partition, reduce, scatter, stencil, and sort preparation,
encoding, pipeline acquisition, and resource destruction are owned by each
operation's `execute.mm`; segmented-scan preparation and destruction are
owned by its `execute.mm` beside the independently compiled encoder. Vulkan
uses the corresponding `execute.cpp` owner, while Map uses its existing
`finish.cpp` run owner because it has no standalone execute translation unit.
Every native translation unit owns a distinct calculation or platform
boundary; forwarding-only translation units are not admitted.

The CPU SIMD `32/run.cpp` and `64/run.cpp` files are distinct compiled numeric
owners: each instantiates the shared executor under a different storage-width
configuration.

For a same-suite profile cohort of size `k`, cold case-owner work is
`Theta(k)` and a changed retained owner still relinks the shared executable;
warm sibling switch work is zero build actions. The product admission bound
fixes `k <= 2`.
The profile therefore admits at most one additional cold owner while making
an unchanged sibling switch schedule zero build actions. Current latency and
fan-out evidence belongs only to `tools/measure/build/run` and its sealed
packet.

`tools/measure/build/run [build [target]]` observes this boundary without
creating another graph authority. It reads the generated compile database,
live Ninja targets, and Ninja depfiles, then brackets the observation with one
unchanged source manifest. Materialized public-umbrella fan-out is reported
separately from selected private leaves, including the compiled assertion
boundary, and the Socket leaf's direct compile edge is reported separately
from both. A live but dirty object may still carry a depfile from its last
compile; zero target dirtiness is therefore required before a
materialized row is treated as a warm graph observation.

## ABI And Toolchain

The installed STATIC library is `node`. It exposes the Node include boundary,
C++20, public Accel/Kernel and dynamic-loader linkage, private math linkage,
and selected native backend dependencies. Native definitions, SDK headers, and
compiler usage apply only to their Metal or Vulkan OBJECT component.
Internal archives never repeat external links already owned by their SCC root.

Verification profiles require Ninja and reject a missing driver or non-Ninja
tree. CMake and direct builds use the repository Ninja owner, so graph, depfile,
and output mutation share one state lock.

## Admission

Configuration rejects:

- a missing, duplicate, unowned, or noncanonical production source;
- an unknown, duplicate, cyclic, empty, or incorrectly closed component/SCC;
- a malformed profile, root, or backend projection;
- a focused native projection whose catalog membership, archive component
  set, or native link set disagrees;
- any profile whose component union differs from its declared SCC closure;
- an empty registry group, missing case owner, duplicate suite ownership, or
  route/profile/resource disagreement;
- a CPU profile that admits native accelerator objects or links;
- overlapping or wrong-cardinality Kernel views, or an installed Kernel union
  that differs from the generated complete closure;
- an external dependency attached outside its owning SCC;
- a materialized object for the zero-object Runtime product join;
- a selected platform with zero or multiple reactor/IO/network owners.

Unix selects one native reactor and POSIX byte substrate. Other hosts select
the complete `runtime/platform/unavailable/` implementation, which links the
portable product and reports unsupported operations. The forced-unavailable
configuration exercises that same owner on Unix. Vulkan may be disabled
without changing source ownership.
