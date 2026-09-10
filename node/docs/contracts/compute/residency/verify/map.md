# Residency Implementation Map

```text
node/include/rund/compute/virtual.hpp          public facade
node/src/compute/virtual/{prepare,state}.*    prepared product owner
node/src/compute/virtual/run/*                 direct execution coordinators
node/src/compute/virtual/run/execution.cpp    asynchronous DeviceVsm lifecycle
                                               coordinator
node/src/compute/virtual/run/execution/seal.cpp
                                               backend-neutral recurrent plan
                                               seal authority
node/src/compute/virtual/run/execution/sliding.cpp
                                               prepared Sliding backend handoff
node/src/compute/virtual/run/backing/materialize.cpp
                                               input page reads, reuse, and
                                               boundary identity fill
node/src/compute/virtual/run/backing/prefetch.cpp
                                               Device/Host probe, admission,
                                               overlap alias, and worker submit
node/src/compute/virtual/run/backing/recovery.cpp
                                               poison validation and recovery
                                               clear policy
node/src/compute/virtual/run/backing/supply.cpp
                                               prefetched/direct page receipt
                                               validation and statistics
node/src/compute/virtual/run/device_vsm/route/graph.cpp
                                               Graph Host/resident endpoint
                                               eligibility authority
node/src/compute/virtual/run/device_vsm/route/probe.cpp
                                               pre-side-effect route selection
                                               and backend admission
node/src/compute/virtual/run/device_vsm/route/{owner,dispose}.cpp
                                               prepared-owner rearm/release and
                                               failure quarantine lifecycle
node/src/compute/virtual/run/device_vsm/route/{prepare,execute}.cpp
                                               backend prepare and single
                                               execution handoff boundaries
node/src/compute/virtual/run/device_vsm/prepare/graph_wavefront/
  {dependency,page_map}.cpp                    wavefront predecessor proof and
                                               external page-map projection
node/src/compute/virtual/run/device_vsm/prepare/graph_wavefront/resident.cpp
                                               ordered GraphResident preparation
                                               coordinator and publication proof
node/src/compute/virtual/run/device_vsm/prepare/graph_wavefront/
  {validation,resources,owners,stages}.cpp     resident shape, resource, physical
                                               owner/bank, and stage/port owners
node/src/compute/virtual/run/device_vsm/prepare/graph_wavefront/internal.hpp
                                               bounded transient resident draft
                                               declarations only
node/src/compute/virtual/run/device_vsm/prepare/buffers.cpp
                                               physical buffer allocation and
                                               fallback/mapping admission
node/src/compute/virtual/run/device_vsm/prepare/buffers/{match,graph}.cpp
                                               ordinary and GraphResident
                                               physical-binding authentication
node/src/compute/virtual/run/device_vsm/prepare/buffers/local.hpp
                                               declarations-only shared seam;
                                               no retained state authority
node/src/compute/virtual/run/projection/dispatch.cpp
                                               route-to-projection dispatch
node/src/compute/virtual/run/projection/{multi,ordinary,validate}.cpp
                                               multi-input and ordinary stream
                                               validation/projection owners
node/src/compute/virtual/run/projection/{frames,transfer}.cpp
                                               frame address and transfer
                                               binding owners
node/src/compute/virtual/run/projection/identity.hpp
node/src/compute/virtual/run/projection/identity.cpp
                                               materialization hash and
                                               input/output/transient identity
node/src/compute/virtual/run/projection/graph/{validate,regions}.cpp
                                               Graph resource/region contract
node/src/compute/virtual/run/projection/graph/{banks,materialize}.cpp
                                               physical-owner binding and
                                               materialization projection
node/src/compute/virtual/run/projection/graph/coordinator.cpp
                                               Graph projection coordinator
node/src/compute/virtual/run/projection/pages.cpp
                                               page/epoch projection and cache
                                               key projection
node/src/compute/virtual/run/scan/input.cpp  Scan admission and input snapshot /
                                               last-element capture
node/src/compute/virtual/run/scan/uniform.cpp
                                               carry arithmetic, overflow checks,
                                               and ordered uniform preparation
node/src/compute/virtual/run/scan/transfer.cpp
                                               CPU restore or native uniform
                                               upload and transfer statistics
node/src/compute/virtual/run/publication.{hpp,cpp}
                                               two-bank deferred publication
                                               cursor protocol and poison/restore
node/src/compute/virtual/run/transaction.hpp   VirtualBacking transaction state
node/src/compute/virtual/run/transaction/model.cpp
                                               captured region, Authority, and
                                               provider-identity validation
node/src/compute/virtual/run/transaction/record.cpp
                                               physical output journal merge
node/src/compute/virtual/run/transaction/{begin,prepare}.cpp
                                               provider admission and exact
                                               physical-row capture
node/src/compute/virtual/run/transaction/{commit,abort}.cpp
                                               mutually exclusive terminal
                                               publication/disposition
node/src/compute/device/residency/registry/transaction_owner.{hpp,cpp}
                                               physical output-row tagging,
                                               lease capture/cleanup and
                                               commit/abort over Authority
node/src/compute/virtual/run/overlap/model.{hpp,cpp}
                                               value-only direct epoch model,
                                               bounded leases, and shared
                                               timeline/statistics projection
node/src/compute/virtual/run/overlap/prepare.cpp
                                               input preparation and prefetch
                                               admission/cleanup
node/src/compute/virtual/run/overlap/submit.cpp
                                               native submit and completion
                                               terminal ownership
node/src/compute/virtual/run/overlap/flush.cpp
                                               output flush, reduction, and
                                               backing publication staging
node/src/compute/virtual/run/overlap/scheduler.cpp
                                               two-bank top-level epoch order
node/src/compute/virtual/graph/reduce/timeline.{hpp,cpp}
                                               canonical interval duration,
                                               intersection, transfer, and
                                               stall/overlap accounting
node/src/compute/virtual/graph/reduce/projection.cpp
                                               thin topology projection facade
node/src/compute/virtual/graph/reduce/projection/internal.hpp
                                               declarations-only projection seam
node/src/compute/virtual/graph/reduce/projection/{identity,effects,regions,
stage,ticket}.cpp                              graph identity/effects, physical
                                               region/key, stage scratch, and
                                               ordered ticket projection owners
node/src/compute/virtual/graph/*               recurrent Graph execution
node/src/compute/graph/compile/slice.{hpp,cpp} canonical graph slices

node/src/compute/device/residency/registry/cache_model.hpp
                                               immutable cache/frame values;
                                               it owns no mutable authority
node/src/compute/device/residency/registry/transfer.hpp
                                               immutable CacheBinding and
                                               CacheTransition transfer records
node/src/compute/device/residency/registry/graph.hpp
                                               Graph materialization, port,
                                               relocation, and freeze records
node/src/compute/device/residency/registry/credentials/{result,cpu,epoch,
direct,execution,sliding,transaction}.hpp       one semantic owner per
                                               Authority result/diagnostic,
                                               CPU persist reservation,
                                               epoch/alias lease, direct
                                               recurrence, execution ticket,
                                               Sliding final, and transactional
                                               lease value; no mutable storage
node/src/compute/device/residency/registry/credentials.hpp
                                               include-only compatibility
                                               surface; owns no credential
node/src/compute/device/residency/registry/model/{frame,lease,execution,
cycle}.hpp                                      exact private state layouts;
                                               no duplicate Authority storage
node/src/compute/device/residency/registry/frame/support.cpp
                                               canonical frame lookup, region
                                               predicates, and credentials
node/src/compute/device/residency/registry/frame/journal.cpp
                                               lease journal clear, rollback,
                                               and completion transitions
node/src/compute/device/residency/registry/frame/registration.cpp
                                               frame and resident-view
                                               registration with setup reserve
node/src/compute/device/residency/registry/frame/ownership.cpp
                                               region release and ownership
                                               proof
node/src/compute/device/residency/registry/model/cpu.hpp
                                               CPU pending/retry records and
                                               stateless quarantine owner seam
node/src/compute/device/residency/registry/cpu_graph_owner.{hpp,cpp}
                                               stateless CPU Graph reservation,
                                               epoch, and quarantine owner;
                                               borrows Authority state/gate
node/src/compute/device/residency/registry/cpu.cpp
                                               CPU-domain counters and
                                               PendingCpu transaction hooks
node/src/compute/device/residency/registry/model/view.hpp
                                               immutable view commit plan/state
                                               and forecast capacity records
node/src/compute/device/residency/registry/view_receipt.{hpp,cpp}
                                               receipt row layout and
                                               nontrivial receipt lifecycle
node/src/compute/device/residency/registry/view_owner.{hpp,cpp}
                                               stateless view protocol and
                                               receipt recycling over the
                                               Authority-owned gate/storage
node/src/compute/device/residency/registry/freeze/{stream,graph}.cpp
                                               schedule-specific immutable-use
                                               validation and application
node/src/compute/device/residency/registry/view/activation.cpp
                                               semantic-view validation and
                                               conflict eviction
node/src/compute/device/residency/registry/view/commit.cpp
                                               allocation-free receipt journal
                                               capture and atomic plan apply
node/src/compute/device/residency/registry/execution/close.cpp
                                               shared locked execution-row
                                               clearance invariant
node/src/compute/device/residency/registry/execution/close/{reject,abandon,abort}.cpp
                                               authenticated execution
                                               terminal and cleanup owners
node/src/compute/device/residency/registry/execution_owner.{hpp,cpp}
                                               stateless generic execution
                                               facet over Authority storage
node/src/compute/device/residency/registry/complete.cpp
                                               locked cache-epoch completion
node/src/compute/device/residency/registry/view/terminal.cpp
                                               receipt authentication, close,
                                               abort, and sticky quarantine
node/src/compute/device/residency/execution/graph_forecast.hpp
                                               GraphForecast page descriptors
                                               and callback-gated credential
node/src/compute/device/residency/registry/graph_forecast_owner.{hpp,cpp}
                                               stateless Graph Forecast
                                               issue/terminal/abort/quarantine/
                                               recovery/release/retire owner
node/src/compute/device/residency/registry/graph_promote_owner.{hpp,cpp}
                                               stateless Graph Promote
                                               issue/group validation/bind,
                                               terminal, and release owner
node/src/compute/device/residency/registry.hpp
                                               public Authority API and the
                                               one physical mutable storage
                                               composition; no state-layout
                                               mirror or receipt lifecycle
node/src/compute/device/residency/registry/graph_epoch.cpp
                                               graph transform and simple graph
                                               entry/locked projection
node/src/compute/device/residency/registry/graph_epoch/admission.cpp
                                               graph epoch lock/retry entry,
                                               state gates, orchestration, and
                                               final token/generation commit
node/src/compute/device/residency/registry/graph_epoch/validation.cpp
                                               request/remap/materialization
                                               validation and PageUse projection
node/src/compute/device/residency/registry/graph_epoch/assignment.cpp
                                               existing-frame scoring and
                                               deterministic Hungarian assignment
node/src/compute/device/residency/registry/graph_epoch/relocation.cpp
                                               rollback-safe permutation,
                                               relocation, and binding staging
node/src/compute/device/residency/registry/graph_epoch/lifecycle.cpp
                                               activate/resume epoch lifecycle
node/src/compute/device/residency/pool.cpp     live Pool execution operations
node/src/compute/device/residency/pool/        footprint, lifecycle, and
                                               atomic acquisition owners
node/src/compute/device/residency/executor.*   CPU worker and accelerator receipts
node/src/compute/device/residency/prefetch.*   metadata-only prefetch
node/src/compute/device/residency/registry/cycle_owner.{hpp,cpp}
                                               stateless rolling cycle behavior
node/src/compute/device/residency/cycle/*      structural temporal model
node/src/compute/device/residency/execution/plan/{seal,project,input,window}.cpp
                                               immutable execution Plan sealing,
                                               projections, input sources, and
                                               Window footprint reconstruction
node/src/compute/device/residency/execution/run/lifecycle.cpp
                                               Q=1 Run begin, issue, terminal,
                                               native, rejection, and abandon
node/src/compute/device/residency/execution/run/finalize.cpp
                                               host/native evidence merge and
                                               final close request construction
node/src/compute/device/residency/execution/run/authority_close.cpp
                                               Authority evidence validation,
                                               frame publication, and quarantine
node/src/compute/device/residency/execution/window.cpp
                                               bounded Window lifecycle and
                                               service ordering
node/src/compute/device/residency/execution/window/authority_close.cpp
                                               Window evidence validation,
                                               frame publication, and quarantine
node/src/compute/device/residency/registry/execution/issue.cpp
                                               recurrent issue lock/order
                                               coordinator
node/src/compute/device/residency/registry/execution/issue/validation.cpp
                                               plan, predecessor, identity, and
                                               slot-admission validation
node/src/compute/device/residency/registry/execution/issue/window.cpp
                                               Host-service cache staging and
                                               window transfer journal
node/src/compute/device/residency/registry/execution/issue/output.cpp
                                               cache-aware output service
                                               admission
node/src/compute/device/residency/registry/execution/issue/finalize.cpp
                                               issue journal state and ticket
                                               publication
node/src/compute/device/residency/registry/execution/admit.cpp
                                               lease/sliding admission ordering
node/src/compute/device/residency/registry/execution/admit/regions.cpp
                                               frame-region and representative
                                               epoch validation
node/src/compute/device/residency/registry/execution/admit/direct.cpp
                                               Q=1 direct cache journal and
                                               no-mutation preflight
node/src/compute/device/residency/registry/execution/{terminal,close,
accept,release}.cpp                            recurrent transaction lifecycle

node/src/compute/pipeline/residency/authority.* private-resource proof
node/src/compute/pipeline/residency/integration/append/support.cpp
                                               shared frame-binding projection
node/src/compute/pipeline/residency/integration/append/direct.cpp
                                               cold direct residency append
node/src/compute/pipeline/residency/integration/append/graph.cpp
                                               cold generic Graph expansion
node/src/compute/pipeline/residency/integration/append/semantic.cpp
                                               cold semantic Graph expansion
node/src/compute/pipeline/residency/integration/support.cpp
                                               shared physical-region projection
node/src/compute/pipeline/residency/integration/semantic.cpp
                                               semantic Graph port binding
node/src/compute/pipeline/residency/integration/graph.cpp
                                               generic Graph port binding
node/src/compute/pipeline/residency/integration/direct.cpp
                                               Direct/stream region binding
node/src/compute/pipeline/residency/integration.cpp
                                               public residency route coordinator
node/src/compute/pipeline/residency/planner/stream.cpp
                                               one-resource Stream plan admission
node/src/compute/pipeline/residency/planner/graph.cpp
                                               ordered Graph planning coordinator
node/src/compute/pipeline/residency/planner/graph/{support,validation}.cpp
                                               resource lookup, remap, and stage
                                               port normalization/validation
node/src/compute/pipeline/residency/planner/graph/liveness.cpp
                                               graph liveness and next-consumer
                                               sealing
node/src/compute/pipeline/residency/planner/graph/physical.cpp
                                               interval coloring and physical
                                               class assignment
node/src/compute/pipeline/residency/planner/graph/dependencies.cpp
                                               same/prior-batch wavefront edges
node/src/compute/pipeline/execution/*           Pipeline terminal owner
node/src/compute/pipeline/plan/publication/{common,controls,window,
                                               terminal}.cpp
                                               shared resolution, Window
                                               controls, and publication-kind
                                               planning authorities
node/src/compute/pipeline/plan/publication/identity.cpp
                                               public/backend publication
                                               identity projection
node/src/compute/pipeline/plan/publication/coordinator.cpp
                                               ordered publication-plan
                                               assembly and fingerprint fold
node/src/compute/pipeline/snapshot/{hash,metadata,payload,restore,storage,
                                               coordinator}.cpp
                                               canonical hash/schema, payload
                                               capture, restore, reusable
                                               storage, and public coordination
node/src/compute/pipeline/snapshot/publication.cpp
                                               publication-state acquisition,
                                               rebase/copy restore, and latest
                                               publication observers
node/src/compute/pipeline/generation.cpp       native generation seed and
                                               failed-submit rebase authority
node/src/compute/backend.hpp                   DeviceOps capability
node/src/compute/backend/accel.cpp             immutable DeviceOps composition
node/src/compute/backend/accel/transfer.{hpp,cpp}
                                               download/upload/copy, host
                                               views, and buffer resolution
node/src/compute/backend/accel/program.{hpp,cpp}
                                               compile, range projection, and
                                               scratch planning
node/src/compute/backend/accel/pipeline.hpp
node/src/compute/backend/accel/pipeline/{memory,preparation,submission,
                                               window,schedule,sliding}.cpp
                                               prepared pipeline, residency
                                               bridge, memory, and capability
node/src/compute/backend/accel/execution/*     Plan-free cold owner alias only
node/src/accel/kernel/terminal.hpp             exact-generation terminal cell
node/src/accel/kernel/prepared/completion.cpp  ordinary Run/Pipeline submit,
                                               completion, and sync wait
node/src/accel/kernel/prepared/completion/residency/{validation,terminal,
                                               window,stream,stream_lifecycle,
                                               control}.cpp
                                               residency Window/Stream
                                               validation, terminal, submit,
                                               signal, abort, release, and
                                               quarantine lifecycle
node/src/accel/backend/ops/table.hpp           backend capability table
node/src/accel/metal/kernel/pipeline/residency/*
                                               retained Metal window/schedule owners
node/src/accel/metal/kernel/pipeline/residency/persistent/prepare/validation.mm
                                               issue-to-check coordinator only
node/src/accel/metal/kernel/pipeline/residency/persistent/prepare/validation/
  {issue,role,request,identity,capability}.mm
                                               issue, native-role/sequence,
                                               request/range, prepared identity,
                                               and capability proof facets
node/src/accel/metal/kernel/pipeline/prepare/spatial_window/
                                               stateless spatial-Window proof model,
                                               equality, geometry/binding proof,
                                               diagnostics, validation, and
                                               admission proof
node/src/accel/metal/kernel/pipeline/prepare/program/
                                               ordered Metal program encoding:
                                               recurrence, entry capture,
                                               window controls, body/status,
                                               publication, and composition
node/src/accel/vulkan/kernel/pipeline/residency/selection.cpp
                                               preparation generation/reset,
                                               selection destruction/lifecycle
node/src/accel/vulkan/kernel/pipeline/residency/plan.hpp/.cpp
                                               immutable selection-entry and
                                               dispatch plan/admission authority
node/src/accel/vulkan/kernel/pipeline/residency/materialize.cpp
                                               allocation, command recording,
                                               and residency install transaction
node/src/accel/vulkan/kernel/pipeline/residency/status.cpp
                                               status/readiness/commit and
                                               submission projection
node/src/accel/vulkan/kernel/pipeline/residency/graph_direct.{hpp,cpp}
                                               aggregate GraphStageDirect proof
                                               construction and identity match
node/src/accel/vulkan/kernel/pipeline/residency/graph_direct/local.hpp
                                               declarations-only private seam
node/src/accel/vulkan/kernel/pipeline/residency/graph_direct/bindings.cpp
                                               resident range/canonical identity
                                               predicates
node/src/accel/vulkan/kernel/pipeline/residency/graph_direct/execution.cpp
                                               graph roles, aliases, controls,
                                               Map/Reduce dispatch
node/src/accel/vulkan/kernel/pipeline/residency/graph_direct/{map,reduce}.cpp
                                               ordered operation-specific binding
                                               proofs
node/src/accel/kernel/residency/device_vsm/projection.cpp
                                               public projection coordinator
node/src/accel/kernel/residency/device_vsm/projection/internal.hpp
                                               bounded candidate declarations
node/src/accel/kernel/residency/device_vsm/projection/candidate.cpp
                                               request and route selection
node/src/accel/kernel/residency/device_vsm/projection/pipeline.cpp
                                               pointwise/graph-map pipeline
                                               matching
node/src/accel/kernel/residency/device_vsm/projection/validate.cpp
                                               route geometry/topology/peer
                                               validation
node/src/accel/kernel/residency/device_vsm/projection/artifact.cpp
                                               route artifact construction,
                                               owner retention, proof sealing
node/src/accel/kernel/residency/device_vsm/projection/window.hpp
                                               private Window authority records
                                               and narrow projection seam
node/src/accel/kernel/residency/device_vsm/projection/window.cpp
                                               exact Window/map binding, plan,
                                               and Window identity projection;
                                               homogeneous proof and capability
                                               query
node/src/accel/kernel/residency/device_vsm/graph_resident.hpp
                                               GraphResident layouts and
                                               declarations-only public proof
                                               boundary
node/src/accel/kernel/residency/device_vsm/graph_resident/{type,identity,digest}.cpp
                                               type codes, owner identity, and
                                               canonical digest
node/src/accel/kernel/residency/device_vsm/graph_resident/{root,owners,resources,
stages,lifetime,tails}.cpp
                                               decomposed proof invariants and
                                               unused-tail closure
node/src/accel/kernel/residency/device_vsm/graph_resident/validation.cpp
                                               ordered public proof validator
node/src/accel/vulkan/timeline/owner/validation.cpp
                                               timeline capability, native
                                               proc validation, and value bounds
node/src/accel/vulkan/timeline/owner/lifecycle.cpp
                                               physical feature query and
                                               semaphore create/destroy reset
node/src/accel/vulkan/timeline/owner/generation.cpp
                                               generation reserve/cancel/close
                                               and capability authority
node/src/accel/vulkan/timeline/owner/point.cpp
                                               point reservation and ready
                                               preflight/signal
node/src/accel/vulkan/timeline/owner/submit.cpp
                                               single command-batch submit
node/src/accel/vulkan/timeline/owner/batch.cpp
                                               window/stream batch lowering and
                                               one queue submission
node/src/accel/vulkan/timeline/owner/observe.cpp
                                               done wait and counter read
```

## Authority contract evidence

The CPU Authority contract keeps one dispatcher order while each independent
contract has its own translation unit:

```text
node/tests/contract/compute/pipeline/residency/authority/cache.cpp
                                               cache and basic lease lifecycle
node/tests/contract/compute/pipeline/residency/authority/migration.cpp
                                               Device→Host migration and dirty
                                               overwrite/writeback admission
node/tests/contract/compute/pipeline/residency/authority/transform.cpp
                                               transform identity/role/tier,
                                               transient materialization, and
                                               rollback/retirement
node/tests/contract/compute/pipeline/residency/authority/capacity.cpp
                                               planner frame capacity and
                                               concurrent lease admission
node/tests/contract/compute/pipeline/residency/authority/lifetime.cpp
                                               transient/output retention and
                                               missing-transient rejection
node/tests/contract/compute/pipeline/residency/authority/graph.cpp
                                               multi-resource Graph admission,
                                               common-local validation, and
                                               atomic failure/recovery
node/tests/contract/compute/pipeline/residency/authority/dispatcher.cpp
                                               ordered migration/transform/
                                               capacity/lifetime/graph routing
node/tests/contract/compute/pipeline/residency/authority/relocation.cpp
                                               cross-layout relocation
node/tests/contract/compute/pipeline/residency/authority/views.cpp
                                               semantic view switching
node/tests/contract/compute/pipeline/residency/authority/prefetch.cpp
                                               metadata-only Prefetcher lifecycle
```

The virtual-product active-prefix contract keeps its single fixture owner and
publishes one case family per invariant, with the dispatcher retaining the
original initialization, cold, growth, and invalidation order:

```text
node/tests/contract/compute/virtual/product/active/support.cpp
                                               fixture construction and shared
                                               backing/statistics/value helpers
node/tests/contract/compute/virtual/product/active/base.cpp
                                               partial, shape-rejection, and
                                               cold identity/content checks (1–7)
node/tests/contract/compute/virtual/product/active/growth.cpp
                                               active-prefix sizes, cross-active
                                               cache behavior, and warm runs (8–10)
node/tests/contract/compute/virtual/product/active/evidence.cpp
                                               allocation, capacity, and output
                                               evidence (11)
node/tests/contract/compute/virtual/product/active/evidence/stats.cpp
                                               exact residency-stat predicate
                                               and first-failure diagnostics
node/tests/contract/compute/virtual/product/active/cache.cpp
                                               invalidation/recovery evidence
                                               (12–13)
node/tests/contract/compute/virtual/product/active/dispatcher.cpp
                                               public CheckProductActiveCount
                                               order and return routing
```

## CPU Host graph evidence

The CPU-only Host fixture reuses
`src/compute/virtual/graph_resident/workload.hpp::Spec` for topology, input,
expected output, hashes, and digests. `graph_resident_host/{program,fixture,
 evidence,oracle}.cpp` split construction, one invocation, capture, and
comparison. The accepted map is `GraphPointwise`/`CpuRolling`, frame capacity
2, owner mask/count 0, no accelerator callback or command submit, fifteen
epoch/page-in events, three input backing reads of 584 bytes each, and five
584-byte output writes. The proposed single verification command is:

```text
tools/test/run --fresh compute.virtual-graph-residency-product --backend cpu
```

The accelerator graph-resident fixture keeps only the shared backing seed in
`graph_resident/fixture.cpp`; `fixture/prepare.cpp` owns canonical U64 resident
construction, `fixture/staged.cpp` owns staged/dynamic preparation and loss
evidence, and `fixture/run.cpp` owns invocation routing. Their `local.hpp` is a
declaration-only seam. The independent U32
scalar-width fixture and its compile diagnostics are owned by
`graph_resident/fixture/u32.cpp`; both consume the single byte-write helper
declared in `internal.hpp`. Program construction, captured evidence, and the
oracle remain in their existing compiled owners, so the split creates no
second workload or expected-value authority.
