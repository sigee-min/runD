# Compute Virtual Residency

This directory owns the opt-in `<rund/compute/virtual.hpp>` contract. Its
pages partition one product authority; none is an alternate allocator,
planner, cache, scheduler, memory ledger, or telemetry owner.

The installed Virtual surface is physically partitioned by declaration
authority: `rund/compute/virtual/backing.hpp` owns backing capabilities and
transaction records, `virtual/buffer.hpp` owns typed buffers and backing
factories, `virtual/pipeline.hpp` owns pipeline handles and access, and
`virtual/prepare.hpp` owns preparation and overload templates. The stable
`virtual.hpp` entry is an include-only umbrella over those support leaves;
they do not introduce a second public product surface or duplicate declarations.

## Read Order

1. [Product](./product.md) for the public surface and admitted routes.
2. [Plan](./plan.md) for immutable demand, capacity, and identity.
3. [State](./state.md) for the sole mutable Authority and backing laws.
4. [Pool](./pool.md) for physical ownership and memory accounting.
5. [Footprint](./footprint.md) for canonical page, Window halo, Graph pin, and fusion alignment.
6. [Graph](./graph.md) for recurrent graph demand and arena lending.
7. [Cycle](./cycle.md) for rolling execution and overlap ordering.
8. [Execution](./execution/README.md) for the recurrent compound transaction.
9. [Backend](./backend/README.md) for CPU/Metal/Vulkan capability and terminals.
10. [Verification](./verify/README.md) for implementation and test ownership.

## Status

| Boundary | Status | Owner |
| --- | --- | --- |
| Device-global software residency authority | Implemented on CPU, Metal, and Vulkan | [State](./state.md) |
| Direct Map, Window, Reduce, and Scan routes | Implemented within the listed constraints | [Product](./product.md) |
| Recurrent U64 Map-prefix-to-Reduce graph subset | Service-free GPU evidence covers all four reductions at one, two, and maximum-width six inputs; the distinct Host Graph path has actual Metal/Vulkan evidence through six inputs and Authority aggregation through seven sources | [Graph](./graph.md) |
| Compatible Graph arena lending | Implemented for the documented storage class | [Pool](./pool.md) |
| Rolling compute-flight journal | Implemented for accelerator non-reduction; structural Plan has no native consumer | [Cycle](./cycle.md) |
| Recurrent whole-run transaction | Eligible ordinary all-staged, unary, nonresident Pointwise Q>=2 first probes `StagedLoop` when exact mapped Host-visible/coherent input and output views exist; it owns one native recurrence submit, Q GPU epochs, zero transfer submits/bytes, zero Host epoch submit/service/callbacks, and one Final/publication/version. A pre-lease structural `BackendUnsupported` cleanly declines to the legacy service; nonresident Persistent Pointwise R2 uses bounded `BackendChunked` two-coordinate chunks (`ceil(Q/2)` native submissions). Rolling remains the capability fallback and other shapes retain their truthful routes | [Persistent](./execution/persistent.md) |
| Resident service-free Direct recurrence | Public Metal/Vulkan terminal-output and `write_each` History `Pipeline::repeat<Q>::run()` use one GPU-owned fused dispatch, fixed retained common/native route storage, and one aggregate Final for Q=5/9/257; large-Q and backing-serviced Virtual recurrence remain open | [Service-Free Direct](./execution/service-free.md) |
| GPU-owned DeviceVsm | Explicit, resident-backed, required, and other statically admitted Metal/Vulkan DeviceVsm shapes use one native submit, zero Host epoch service/callbacks, Q device-generated page epochs, and one Authority/Pipeline/backing Final publication. Ordinary all-staged unary nonresident Pointwise Q>=2 has the separate `StagedLoop` probe and mapped-view contract above; it is not the generic callback-backed cohort. A mapped structural `BackendUnsupported` before owner/rearm/stage/lease/native acceptance is a clean decline to the legacy route, while any owner mutation or non-capability failure is terminal. The bounded resident GraphResident branch requires `input_count>=2` and a nonterminal stage with at least two distinct Internal/Intermediate/Transient reads plus one Internal/Intermediate/Transient write; it uses one physical GPU controller dispatch under the sealed geometry `2<=S<=8`, `2<=R<=9`, `P<=16`, `O<=9`, `S*ceil(Q/C)<64`. Existing all-staged evidence remains the sealed U64 Tile S=5,Q=5,C=2,B=3 case with 15 steps, one submit/controller dispatch/logical dispatch, and one aggregate Final/publication/version; the bounded shared `VirtualReadCohort` path also covers Q6. Generic callback-backed staged cohorts remain blocked by missing cross-input-wait-free reads, cross-backing concurrency budget, callback no-reentry/cycle rule, and dedicated bounded cohort owner/join. Whole-run transfer is separate; mixed endpoints are rejected and exact fallback handle/byte/version authentication is required. TilePartial, AddSat, S=7 outside planner-sealed static GraphResident/GraphPointwise shapes, schedules at or above the tile bound, dynamic Forecast/Promote/Drain/Persist, and dynamic nonresident GPU-native I/O remain unimplemented/blockers. The explicit/resident/required DeviceVSM contract covers the sole-input/sole-output pure total U32/U64 pointwise Map DAG, centered Clamp/Clip U32 Window Sum/Min/Max, canonical Map/Window compositions, pure unsigned Scan/Reduce, canonical total U64 Map-to-collective Graph, and admitted static `GraphPointwise` subsets. | [Persistent](./execution/persistent.md) |
| Native whole-run scheduler | Explicit/coherent or resident Direct schedule contracts remain available with one public handoff and backend-specific queue ownership; mapped all-staged ordinary Pointwise is selected by `StagedLoop` before the legacy Persistent service, while Persistent remains the clean-decline route. Pure unsigned U32/U64 Scan and Sum/CountNonzero/Min/Max Reduce retain the separate DeviceVsm scheduler. | [Scheduler](./execution/scheduler.md) |
| Continuous forecast/promotion split | Direct and the Graph Host product use distinct typed Forecast and Promote lifetimes; actual Metal/Vulkan Host contracts cover six inputs and Authority aggregates seven sources. The Host fallback consumes a fixed two-worker GraphPersist ring for exact Backing-output pages. Native runtime-ready-horizon consumption and a distinct noncoherent transfer-queue lowering remain open. | [Forecast](./execution/forecast.md) |
| Canonical page/epoch footprint alignment | Geometry, SharedHalo, fixed-state reuse, and ordinal Direct Window service-aware multi-page physical Forecast/Assemble binding are implemented. DeviceVsm centered Window seals the same fixed-size Authority footprint and validates GPU-returned boundary/checksum evidence. Arbitrary non-ordinal Graph footprint binding and general fusion remain partial. | [Footprint](./footprint.md) |
| Fixed-state continuous sliding controller | Fixed-storage model and adversarial contracts implemented; mapped all-staged ordinary Pointwise is selected by `StagedLoop`, while legacy Persistent/Window/Stream fallback owns cleanly declined or unsupported shapes | [Sliding](./execution/sliding.md) |
| Typed non-Direct native topology | Pure unsigned U32/U64 Inclusive/Exclusive Scan and Sum/CountNonzero/Min/Max Reduce are implemented through DeviceVsm; Local Window outside its Direct subset, other Reduce/Scan, general Graph, and noncoherent-transfer native lowerings remain open | [Routes](./execution/routes.md) |
| Vulkan bounded native window | Typed finite bare-I32/U32 `WindowRingFused` uses one dispatch/submit on Resident or fully staged endpoints; unsupported/noncoherent I/O retains the dedicated Window/Stream or rolling fallback | [Backend](./backend/README.md) |
| Persistent/Host/Device native state machine | Not implemented | [State](./state.md) |
| Global victims across incompatible native buffers | Partial | [Pool](./pool.md) |

For Persistent `BackendChunked`, the backend bound is two physical payload
slots. Vulkan may additionally retain four fixed authentication metadata
cells; those cells are not payload slots and do not claim same-submit GPU
ownership of nonresident data.

The finite `WindowRingFused` exception is a separate typed DeviceVSM route for
bare centered-I32/U32 Window programs. Its proof selects either the Resident
or all-staged endpoint before the Authority lease; all-staged inputs and the
output are fully staged before that lease. One physical dispatch and one
native submit own `Q` seed epochs, `Q` compute epochs, and `2Q` internal phase
steps, with zero Host epoch submit/service/callbacks and zero transfer
submits/bytes. The common terminal performs one Final and one
publication/version transition. The route does not imply I32 Map fusion.

Ordinary all-staged unary nonresident Pointwise with `Q>=2` probes `StagedLoop`
first. The probe requires exact mapped Host-visible/coherent input and output
views and carries the route enum into preparation; shape, tier, and read-lane
predicates are not recomputed downstream. It owns one native recurrence submit,
`Q` GPU epochs, zero transfer submits/bytes, zero Host epoch submit/service/
callbacks, and one Final/Authority/output publication/version. A structural
`BackendUnsupported` before any owner, rearm, stage, lease, or native acceptance
is a clean decline to the legacy route; every other failure after mutation is
terminal. The natural Q2/Q3/Q5 fixture is a passing functional E2E check:
`tools/test/run --fresh compute.pipeline-metal-persistent-sliding` completes
successfully. This is feature-contract evidence, not a performance result or a
60-sample timing claim. It verifies the public StagedLoop/DeviceVsm one-native-
submit/one-handoff path with Q GPU epochs, zero Host epoch callbacks/transfers,
and one Final/publication/version. Shared Persistent Q2/Q3/Q5 `BackendChunked`
and direct API Q5/Q9/Q257 checks are separate evidence and must not be
conflated with public StagedLoop. The range preflight rejects the reserved
sentinel and overflow request before native owner allocation; this does not
attribute every past IOGPU abort or establish safety for every finite-Q
device/driver. Authoritative native command-capacity/ICB migration for valid
finite-Q direct Metal O(Q) encoding remains a blocker.
An Unknown terminal closes the active control (`active=false`), sets
`quarantined=true`, retains the native owner, and suppresses publication,
fallback, and retry.

Persistent spatial Window is not a generic `OneSubmit` product claim. An
unsupported Persistent Window declines before lease; the dedicated bounded
Window (`Q<=4`) or Stream (`Q>4`) fallback owns one logical handoff, one Final,
and one output version (the current Metal fallback may make `Q` physical queue
calls). Explicit/resident DeviceVsm Window remains a separate contract.
Persistent Sliding accepts only finite coordinate counts: `UINT64_MAX` is the
reserved no-coordinate sentinel, and the final turn of every active role must
fit its uint32 control and uint64 descriptor generation.

The DeviceVsm row also includes exact parameter-free U64 `GraphPointwise`
products with one through seven public inputs at two stages, six public inputs
at three stages, five public inputs at four stages, four public inputs at five
stages, three public inputs at six stages, and two public inputs at seven
stages, while the one-input
form may contain two through eight stages. Every stage retains
their typed read/write authority even when their composed Program exceeds the
expression capacity; the GPU reads the external resident rows directly and
passes stage one's value to stage two in a register. Metal and Vulkan one- and
two-input Q=5,9,257 cases, plus the maximum seven-input/two-stage,
six-input/three-stage, five-input/four-stage, four-input/five-stage,
three-input/six-stage, and two-input/seven-stage Q=5
cases, use one
handoff, one native submit and payload dispatch, zero Host epoch control, and
one Final/publication. The four-stage case also executes exact
XOR/multiply/add continuations through the shared typed-Map scalar-operation
authority. Every point on the seven-to-two input diagonal uses nine planner
resources: inputs plus one intermediate per nonterminal stage plus one output.
The six- and seven-stage cases respectively prove 15 reads/1,752 bytes/six
stage terminals and 10 reads/1,168 bytes/seven stage terminals. This is not
general runtime-selected multi-input or
runtime-ready Graph scheduling.

The DeviceVsm Graph status includes one exact static wavefront boundary: the
proof retains planner identity, fixed frame/batch geometry, and separated
Dispatch/Release predecessor masks, while the one submitted GPU workgroup
uses the deterministic ready order to open the exact composed-Map and terminal
collective payload phases for every batch, then returns the step count and
trace. Canonically fused intermediate Map stages have no separate payload.
Arbitrary non-fusible physical stages and runtime backing-service turns still
do not execute at those selected cells; that general native Graph scheduler
remains open.

Public SDK usage stays in the
[Compute reference](../../../../../docs/reference/compute.md#virtual-working-sets).
Virtual performance procedure and measured results are owned separately by
[Virtual Performance](../../../../../docs/reference/performance/virtual/README.md).
