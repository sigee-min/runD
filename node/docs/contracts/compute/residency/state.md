# Residency State

This page owns the one Device-scoped mutable residency Authority, tier/frame
state transitions, backing serialization, poison, and execution ownership.

## Authority

`DeviceState::residency` owns one Registry. The Registry owns the only
`Authority` and execution gate across every `VirtualPipeline` and registered
Pool layout on that Device. Planner freezes demand; Authority alone chooses
eligible frames and victims and issues physical locals. Execution validates
and consumes those locals but cannot compact them, choose victims, run an
unleased prefix, or publish validity.

The implemented “global” scope is one Device. It is not a native scheduler
spanning Persistent backing, Host memory, and Device memory. Pool and extent
ownership are specified in [Pool](./pool.md).

The Authority header is an API and one-storage composition boundary. Its gate
and physical frame table remain in the façade, while
`registry/model/state.hpp` is the sole data owner for the five lifecycle
components: `ExecutionAuthorityState`, `CycleAuthorityState`,
`CpuGraphAuthorityState`, `ViewAuthorityState`, and `CredentialsState`.
Their members are the only execution, cycle/epoch, CPU graph, view receipt,
and identity/counter storage; the model owners contain no mirror table, gate,
or alternate credential source. Every friend and lifecycle owner reaches the
same component through the Authority façade and still acquires the Authority
gate first.

The detailed method laws live in this State page and the Execution pages,
not as a second prose specification in `registry.hpp`. That header retains a
compact declaration surface for frame registration, Graph and epoch
admission, and stateless lifecycle facets. The components are data-only records under
that one gate; they do not create competing gates or journals. Compiled
algorithms remain split by lifecycle phase while the single Authority storage
boundary stays explicit.

CPU Graph reservation, epoch confirmation/close, and quarantine disposition
are implemented by the stateless `CpuGraphOwner` returned by
`Authority::cpu_graph()` in `registry/cpu_graph_owner.{hpp,cpp}`. It borrows
Authority's gate, frame table, credentials, cycle epochs, and
`CpuGraphAuthorityState`; it owns no state mirror or alternate gate. The
Authority keeps the locked Graph admission seam, while `registry/cpu.cpp`
retains only CPU-domain counters and `PendingCpu` transaction hooks. The
internal `CpuQuarantineOwner` still owns the allocation-free snapshot/commit
plan used by the facet's discard path.

Ordinary and transform lease admission are compiled separately under
`registry/lease/ordinary.cpp` and `registry/lease/transform.cpp`. Their shared
victim selection and dirty-retirement policy is owned once by
`registry/lease/support.cpp`; all three borrow Authority's one gate, frame
table, and lease journal, so this split introduces no second selection policy,
frame ledger, or rollback authority.

View-plan mutation is owned by stateless `ViewOwner`, returned by
`Authority::views()`, without duplicating Authority storage or its gate.
`registry/freeze/{stream,graph}.cpp` own the two schedule-specific
freeze validators and applications, `registry/view/activation.cpp` owns raw
view selection and eviction, `registry/view/commit.cpp` owns the bounded
receipt journal transaction, and `registry/view/terminal.cpp` owns receipt
validation, close, abort, and sticky quarantine. Their shared
`registry/freeze/internal.hpp` is declarations-only; it owns no inline
algorithm, frame table, gate, stamp, or receipt storage. The removed
no-receipt `Authority::freeze` overloads are not a second live mutation path;
schedule changes use the same authenticated `commit_view_plan` as the product.

`VirtualPipelineState::graph_pipelines` is the canonical stage-major,
bank-minor table for Graph preparation. The state has no named
collective/alternate-collective aliases: terminal selection reads the final
stage through the narrow state accessor, which returns a `shared_ptr` copy so
the selected Pipeline remains alive for its caller. `pipeline` and
`alternate_pipeline` remain the ordinary two-bank fields shared by non-Graph
routes; Graph preparation retains their stage-zero views because generic
runtime and publication paths consume that common pair.

Virtual run cache behavior is compiled under `virtual/run/cache/`: `input.cpp`
owns the direct input view, `output.cpp` owns the output view and output-bank
reservation/lease/cancel path, `transform.cpp` owns CacheUse projection,
`supply.cpp` owns CPU/coherent/deferred/transfer supply and upload accounting,
`retention.cpp` owns retained-output validation and its one-shot failure hook,
and `writeback.cpp` owns transactional stage/publish/writeback. `cache.hpp`
is declarations-only; these leaves borrow the existing Pipeline, Pool, and
VirtualBacking owners and introduce no cache mirror or second publication
authority.

Frames move through validated Empty, Mapping, Pinned, Resident, Dirty, and
Writeback states with explicit tier and role. Mapping, Pinned, and Writeback
are non-evictable. Successful execution installs exact dirty extents. A
Backing-domain page drains through Writeback; a physical Device output drains
through Migrate to its Host output; a consumed Transient page drains through
Discard. One drain token stays live until success or fail-closed cleanup.

The two fixed GraphPersist Authority slots also have a `RetryReady` state. A
CPU Known failure rolls back its exact journal into `RetryReady` without a Book
retry mirror. The first CPU graph Authority mutation uses
`begin_cpu_graph_epoch_retry`: it validates the complete prepared identity and
all rows under the same gate, then atomically converts their authenticated
Dirty frames to Empty while publishing the new epoch. A domain-only consume is
not an execution API. UnknownMayWrite or an identity contradiction remains
terminal and retained, never RetryReady. Other RetryReady rows do not block an
already-issued exact terminal close.

Each slot is bounded to `GraphPersistCapacity` pages, while teardown and retry
planning use one derived aggregate frame claim set of
`GraphPersistCapacity * GraphPersistSlotCapacity` (currently 64). Every active
Unknown `Drain` row must be terminal; every `RetryReady` row must be
nonterminal. The Authority preflights all rows together, including token,
generation, domain, full identity, region, transition/undo correspondence,
alias claims, and cross-row frame uniqueness, before any rollback or clear.
`Free` rows have no residual credential or journal metadata. Overflow, a
duplicate frame, or a partial row is a no-mutation failure.
Every GraphPersist row also carries the same nonzero projected `coordinate` in
the Ticket and Authority LeaseSlot; terminal, release, recovery, and Unknown
snapshots authenticate that locator exactly. Quarantine teardown is physically
split into Book snapshot, Authority-only check/commit, and post-unlock Book
scrub; the Authority gate never dereferences Book state.

Before any Drain rollback, RetryReady discard, or Unknown teardown, the shared
allocation-free undo validator checks full transition/undo sizes, bounded
unique frames, Writeback witnesses, frame/key/dirty correspondence, and the
immutable assigned/tier/role/extent/view/registration/next-use/retain/claim
fields. Drain requires current `Writeback` plus prior `Dirty`; RetryReady
requires an exact current/prior `Dirty` snapshot. Both require Host/Output
regions. Direct/public rows require an empty identity and nonzero `row.plan`;
CPU-domain rows require
a valid identity whose plan equals `row.plan`. Malformed or partial rows return
without mutation. Unknown with domain zero becomes an Authority-owned sticky
terminal Drain after its authenticated ticket is cleared; a nonzero CPU domain
keeps the ticket for Book Unknown retention and reports failure to the caller.

On physical output migration, Authority atomically empties the Device source
and publishes the corresponding Host frame as the sole dirty copy. On a
backend-proven coherent route, the Device output itself remains the sole dirty
copy through backing writeback. The terminal distinction is owned by
[Backend](./backend/README.md). Every successful non-Reduce run flushes dirty output
and leaves it clean. Reduce consumes and invalidates partials before its final
scalar. Dirty retention across successful terminals is not implemented.

Deterministic replacement uses frozen future use. A dirty victim must complete
its exact drain before reuse. Failed physical copy invalidates the complete
touched transaction; no stale metadata hit may survive.

## Native Terminal Uncertainty

Authority rollback is legal only after a backend proves that submitted native
work can no longer write the frames. `UnknownMayWrite` is a terminal result for
the Host waiter but not physical completion. The backend therefore keeps the
complete may-write owner tree self-retained, permanently quarantines that
Device's virtual-residency capability, and reports `DeviceLost`. Registry
metadata cannot make those frames reusable, and another prepared owner cannot
enter callbacks or Authority transitions on that Device. Same-owner, peer,
and new-owner use observes the Device-wide `DeviceLost` fact; this boundary is
stronger than an owner-local poison result. Exact generation and backend
mechanics are owned by [Backend](./backend/README.md).

## CPU Graph quarantine teardown

Authority credential values are partitioned by lifecycle under
`registry/credentials/`: `result.hpp` owns common failures and diagnostics,
`cpu.hpp` owns CPU persist identity and reservation keys, `epoch.hpp` owns
epoch and alias leases, `transaction.hpp` owns the locked virtual transaction,
`sliding.hpp` owns Sliding final credentials, `direct.hpp` owns service-free
direct recurrence credentials, and `execution.hpp` owns recurrent execution
tickets and close evidence. `registry/credentials.hpp` is include-only and
owns no second credential definition.

CPU Graph preparation creates one fixed `shared_ptr<CpuReceiptBook>` before
Authority or Pool mutation. `VirtualPipelineState`, `AbortController`, and the
quarantine holder share that exact Book; the holder stores only raw
Pipeline/Pool identity and a temporary self-reference while Authority retain
is attempted, so a normal Authority-owned sticky quarantine has no strong
Pipeline/Pool cycle. A valid lifecycle makes retain rejection unreachable, but
the rejection path keeps self and drops stack handles defensively.

If a state is destroyed with an active CPU quarantine, its destructor first
holds the Pool `execution_gate`, then asks `Authority::cpu_graph()` to discard
the exact quarantine identity. The facet preflights its pointer/Book/domain,
every CPU
epoch and GraphPersist credential, pending/execution/writeback/cycle/native
quiescence, all frame bounds/undo sizes/duplicate rows, and zero alias claims.
Only after that complete check does an allocation-free commit rollback and
invalidate CPU rows, retire or roll back Known GraphPersist rows, clear Book
slots, and remove the Authority quarantine. A contradiction terminates rather
than allowing Pool destruction to drop a live credential.

## Backing

One source-private synchronization/poison state belongs to the
`VirtualBacking` object, not to a `VirtualBuffer` view. All views therefore
agree on callback serialization and recovery. Cross-connected Pipelines
acquire the complete backing-authority set without lock-order dependence;
callbacks over shared backings do not overlap, while disjoint backings remain
independent.

Before a run's first output callback, recovery publishes the largest logical
extent that callback may partially modify. A fresh view, zero run, or smaller
retry remains `BufferPoisoned`. Only a same-or-larger successful overwrite
clears the obligation. Failed terminal writeback discards all affected cache
mappings so another Pipeline or backing cannot inherit uncertain bytes.

An output backing implementing the separate `VirtualBackingTransaction`
interface starts its provider-owned shadow after route admission and before
the first output stage. Bounded page ranges stage into that shadow while
public bytes and version remain unchanged. After Authority and workers drain,
`prepare_commit` validates complete coverage without mutation and one
`commit` swaps the generation and increments the version once. Known failure
calls `abort_known` for same-state retry; unknown may-write calls
`quarantine_unknown` and prevent reuse. Legacy backings retain direct
writeback and have no atomic visibility claim.

One VirtualPipeline admits one active run. A concurrent call returns
`PipelineBusy` before callback, transfer, or dispatch. Destruction requires all
prefetch lanes and Authority tokens to be terminal.

## Tier Boundary

`VirtualBackingTier::Host` and `Persistent` are callback capabilities, not
native storage states. Persistent may admit two parallel read producers; Host
defaults to one. Accelerator Host input pages are capacity-bounded reusable
Authority frames, and physical fallback Host output pages are also registered.
Native NVMe scheduling, complete per-tier occupancy/traffic, terminal-persistent
dirty state, and one Persistent/Host/Device scheduler are not implemented.

## GPU-owned nonresident continuation — blocked

The implemented Metal/Vulkan bounded-W path preencodes O(Q) work and uses
Host-driven service; it must not be described as GPU-owned nonresident
Forecast/Promote/Drain/Persist. The intended private ring has fixed R
independent of Q and authenticates operation/ticket/coordinate/turn/page,
next-use/bank/frame/resource/offset/bytes/generation, and plan digest. Its ack
publishes result/may-write under release/acquire ordering, contiguous ticket
and full/empty laws, a Host-only owner-nonce shadow, page ownership transfer,
one Final, and Unknown retention.

The missing cross-layer primitive is device-to-Host demand wake,
Host-to-device acknowledgement wait, system-scope visibility, and dynamic
continuation without a second submit. Vulkan timeline/host events and Metal
SharedEvent/ICB/MTLIO limitations, plus the noncoherent transfer DAG, remain
unresolved. No unused ABI is an implementation of this state machine.
