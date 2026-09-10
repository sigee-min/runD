# Execution Authority

Authority authenticates the sealed Plan against its registered frame table
and returns one token and run generation. The mutable journal is bounded to
the two recurrent banks, fixed Host-service cells, and the current four-entry
native release journal. It never copies Q frame states or restores stale cache
metadata over bytes that native work may have changed.

The rolling accelerator flight submachine is a stateless `CycleOwner` returned
by value from `Authority::cycles()`. It takes the same Authority gate for every
bind, advance, terminal, and close, and mutates only Authority's existing
epoch/frame/cycle storage. `complete_cycle` calls the shared `complete_epoch`
transition while that gate is held; no second cycle journal or frame authority
exists.

Graph HostReady issue, terminal evidence, authenticated abort, fixed
quarantine/recovery, callback-return release, and `GraphReady` retirement are
directly owned by the stateless `GraphForecastOwner` returned by
`Authority::graph_forecasts()`. The facet borrows this same gate, frame/epoch
table, and four fixed Forecast holders; Authority retains their physical
storage, quarantine predicates, and destructor check.

Graph Device-output to Host-output migration issue, terminal validation, and
callback-return release are directly owned by the stateless `GraphDrainOwner`
returned by `Authority::graph_drains()`. It borrows the same Authority gate,
frame/lease table, and writeback rows; the move-only `GraphDrain` credential
still authenticates both tokens and retains its Authority identity.

Graph Host-output to backing persistence and same-domain CPU Graph retry
admission are directly owned by the stateless `GraphPersistOwner` in
`registry/graph_persist_owner.{hpp,cpp}`, returned by
`Authority::graph_persists()`. Issue, terminal, callback-return release,
Known rollback, Unknown recovery, and retry admission borrow the same gate,
frame table, fixed two-slot `CycleAuthorityState::graph_persists` journal, and
pending CPU reservation; the facet has no second retry or quarantine state.

CPU Graph reservation, epoch confirmation/close, and quarantine disposition
are directly owned by stateless `CpuGraphOwner` in
`registry/cpu_graph_owner.{hpp,cpp}`, returned by `Authority::cpu_graph()`.
It borrows that same gate, frame table, cycle epochs, credentials, and
`CpuGraphAuthorityState`. Authority retains only the locked Graph admission
seam and the storage boundary; neither owner keeps a protocol-state mirror.

The complete rolling Sliding lifecycle is directly owned by the stateless
`SlidingOwner` in `registry/sliding_owner.{hpp,cpp}`, returned by
`Authority::sliding()`. Admission, bound cleanup, physical Fetch/Promote/
Native/Drain/Persist tickets, coordinate disposition, and two-phase Final
preparation/acceptance all borrow the same Authority gate, frame table, and
execution slot. Authority retains the storage and lock authority; the facet
holds no protocol state or mirrored predicates.

The service-free Direct recurrence lifecycle is directly owned by the stateless
`DirectRecurrenceOwner` in `registry/direct_recurrence_owner.{hpp,cpp}`,
returned by `Authority::direct_recurrences()`. Registration binding, lease
admission, Final preparation/staging, pending release, publication, and both
abort paths borrow the same Authority gate, frame table, and ExecutionSlot.
Authority exposes no parallel Direct transition methods; only its private
construction rollback remains available to the two registration owners.

The generic service-backed execution lifecycle is directly owned by the
stateless `ExecutionOwner` in `registry/execution_owner.{hpp,cpp}`, returned
by `Authority::executions()`. It borrows Authority's gate, physical frame
table, shared ExecutionSlot, and credentials; it owns no journal, counters,
or mirrored protocol state. The private `admit_execution` seam, region
validation, direct-admission predicate, and `close_rows_clear_locked`
cross-protocol predicate remain on Authority because Sliding and the close
checks use those shared locked invariants. Authority exposes no parallel
generic execution transition methods.

Pooled Stream/Graph view activation, commit, Known abort, close, and Unknown
quarantine are directly owned by stateless `ViewOwner` in
`registry/view_owner.{hpp,cpp}` and the semantic `registry/view/` leaves,
returned by `Authority::views()`. The facet borrows the single frame table,
gate, credentials, and receipt storage. Shared active/quarantine predicates
and setup-time receipt capacity remain private Authority seams because frame
registration and other protocols consume those same locked invariants.
Receipt identity remains the Authority address, never the temporary facet.
`registry/freeze/{stream,graph}.cpp` owns shared schedule validation/projection
only: the obsolete no-receipt `Authority::freeze` entrypoints are removed.
Both product and cache-replacement tests use the authenticated commit path.

Pooled Stream/Graph view commits use one Authority-owned idle
`ViewCommitReceipt`. Its bounded `Row` storage is created or grown only by
frame registration, before the resulting frame table is mutated;
`commit_view_plan` is allocation-free. A successful close or Known abort clears
row credentials and returns the same holder to the Idle slot. Unknown or
conflicting disposition moves that active holder to sticky quarantine instead
of recycling it. Frame registration and release require the Authority to be
Idle, so setup-time growth cannot race an active receipt.

Within epoch `e`, ordering is Input -> Dispatch -> Output. Dispatch terminal
releases the Input bank, so Input(e+2) may overlap Output(e). Output(e) must
terminal before Dispatch(e+2) reuses the output bank. Queue acceptance is not
a successful terminal.

An out-of-order later-bank Output terminal cannot close an earlier outstanding
Output ticket. The execution contract explicitly attempts that premature
close, requires `Busy` without consuming the journal, then completes the exact
remaining ticket and verifies the full service/completion counts.

Host-service tickets name the exact Plan node, token, generation, bindings,
transitions, masks, and monotonic sequence. Input bindings are K Host locals
followed by K Device locals. The backing mask names Host misses; the transfer
mask names Device supply obligations. Window/Stream admission treats both Host
input banks as one cache domain while keeping Device targets bank-local. When
no Host victim is safe and the exact Device target misses, the Host binding is
an authenticated sentinel and the coherent mask authorizes backing fill
directly into that Device local. The same bit is then backing work, not a
transfer; no H2D receipt or byte count may be fabricated. A coherent Device
hit may also use the sentinel with neither mask. Output bindings are
Device/Host pairs and carry canonical Map then Writeback transitions.

Native adapters cannot choose pages, victims, banks, or publication. They
return bounded dispatch facts. Compute owns Pipeline start, control generation,
finish, publication, and reseed before exposing a Release to Authority.

Final success never resurrects pre-run keys. Cache-aware Q1 can retain exact
clean inputs from its fixed tickets and retire outputs after backing service.
Opaque window/stream paths conservatively invalidate mutated reserved rows
unless exact service receipts prove their terminal materialization.

Continuous Sliding binds its internally minted owner to the admitted execution
before the first Fetch or native ticket. Final is a two-phase capability:
Sliding first freezes a nonce-bound immutable Final; Authority verifies that
exact plan/token/generation/owner and freezes the physical reservation without
mutation; Authority accept then conservatively retires every reserved row and
mints one receipt for the exact Sliding nonce. The controller consumes that
receipt once. Cross-Authority, cross-controller, stale-nonce, late-bind, and
replay attempts are rejected. `UnknownMayWrite` cannot enter this Known/success
close and remains quarantined.

The fused Persistent Final has an explicit private `Inflight` subphase after
`commit_sliding_frames` and before the callback continuation. Authority and
Sliding gates are released during that no-fail `noexcept` continuation, but
the plan/token/generation/owner/final nonce and physical rows remain live.
Authority begin, abort, close, or duplicate fused-finish attempts therefore
return Busy/Invalid with no mutation; they do not clear or quarantine the live
slot. The continuation reacquires Authority then Sliding and authenticates the
same credential before closing exactly once. Pipeline, publication, and
backing gates remain held by the caller throughout; Pipeline API reentry is
not part of this private contract. If reauthentication fails after
publication, the owner is sticky Quarantined and no callback or rollback is
attempted.

Every post-begin exit is terminal. Known failures invalidate exact mutation
regions. `UnknownMayWrite` quarantines all possibly writable owners. Internal
credential contradictions use an authenticated abandon path that invalidates
the reservation, publishes nothing, and leaves nontransactional backing
recovery set. Before Sliding Final freeze, a known synchronous rearm/submit
rejection sets `CloseRequirement::Required` and is handed synchronously to
`close_generation`. After Final freeze, a restored failure closes as `Closed`;
rollback or control uncertainty is `Quarantined`. A close is `Closed` only
after authenticated quiescence and token clearing; related live ambiguity
remains `Quarantined`. `CloseRequirement::Required` is a synchronous handoff
marker, not retry evidence: `RestoredFailure` may close the generation, while
`Poisoned` requires quarantine and maps to `DeviceLost`.

Region release is an owner gate: a requested row, or any row with the same
nonzero `PhysicalArena` extent, is rejected while an active owner-bearing
journal row/reference holds it or that extent has an alias claim. Active cycle
membership is part of the same gate; evidence and progress alone never own a
release. Extent-zero proves only ordinary exact-region reuse. Arbitrary
`BufferState` or resident aliases remain unsupported.

## Direct registration credential

The CPU Graph path reserves its quarantine spare and one shared
`shared_ptr<CpuReceiptBook>` before the first CPU Authority mutation. State,
Abort, and the quarantine holder all reference that same non-movable,
non-copyable Book; its address and domain never change. The fixed Book owns
stage epoch credentials and only fixed Unknown credential locators. Authority's
`CycleAuthorityState::graph_persists` rows own the GraphPersist page journal
and rollback state; the
Book never mirrors those pages or transitions. It authenticates the complete
candidate set, snapshots the destination mapping, and only then performs its
noexcept adoption. The private GraphPersist generation is nonwrapping. Known
close rolls the exact slot back into Authority `RetryReady` and leaves the run
retryable; the first same-domain CPU graph calls
`begin_cpu_graph_epoch_retry`, which validates the complete prepared identity
and consumes all matching RetryReady rows in the same Authority-gated
preflight/commit as its first epoch begin. A standalone domain-only consume is
not an execution path. Other RetryReady rows do not block the already-issued
exact close. Unknown recovery
arms the same Book with raw Pipeline/Pool identity and a temporary
self-reference, calls Authority retain first, then drops only that
self-reference after successful retain before dropping stack receipt handles.
A retain rejection keeps self and drops handles defensively; valid lifecycle
prechecks make that rejection unreachable. The Authority-owned sticky holder
has no strong Pipeline/Pool root cycle. If the Pipeline state is destroyed
while the holder is active, its destructor takes the Pool execution gate and
performs one exact, allocation-free Authority discard before Pool destruction;
any preflight contradiction is a lifecycle invariant breach.

GraphPersist terminal, release, and recovery first authenticate the exact
Authority row `{owner, token, generation, coordinate, plan, book-domain,
region, full identity, pages}`. Terminal status, completion evidence, and identity failure
flags are not changed before that check succeeds; a mismatch leaves the row
owned by Authority for the existing Unknown policy.

The same allocation-free undo preflight is then required by terminal recovery,
Known rollback, RetryReady discard, and quarantine teardown. It validates
unique bounded frames, Writeback transition witnesses, exact Dirty undo rows,
all frame identity fields, and Host/Output region ownership before rollback;
partial issue journals use the same credential-independent core before their
pre-publication rollback. Direct rows require an empty identity and nonzero
plan, while CPU
rows require a valid identity whose plan equals the Authority row.
An authenticated Unknown with `book-domain == 0` clears its Ticket and returns
the sticky Authority terminal as handled; a nonzero CPU domain preserves the
Ticket so the Book can retain the exact Unknown locator and reports failure for
the quarantine path.

Resident Direct registration uses one immutable credential: a process-unique
nonzero nonce, a Release or Retain policy, and a fixed ordered copy of every
binding's registration, region, view, and usage. The wrapper, lease, Final, and
Authority slot carry the same credential identity and nonce; a caller-only
span is never the release authority. Registration construction rolls back a
partial ordered binding set only through the Authority's exact construction
rollback; a mismatch returns Busy and leaves every row untouched. Runtime
observation uses one wrapper-gated value Snapshot, never an unlocked proof or
binding span.

The State credential also seals the wrapper's shared-owner control block and
object pointer through a weak owner reference. Authority requires both the
same shared control block (bidirectional `owner_before` equivalence) and the
same wrapper object pointer; aliasing a subobject or reusing the pointer from a
different control block is not accepted. An owner-less State cannot be
constructed. With no active direct execution slot, a direct abort is
`DirectAbort::Invalid`; once a slot is active, foreign, stale, wrong-phase,
Pending, or Inflight abort credentials return `Busy` with no row, lifecycle,
publication, or quarantine mutation.

The lifecycle is `Active -> Admitted -> Frozen -> Pending -> Inflight`, then
`Released` for service-free Release or `Active` for a successful Retain owner.
Known stage work only enters Pending. Same-credential Pending release is
idempotent. Foreign credentials and duplicate Inflight operations return Busy
without mutation or quarantine. Dirty, contradictory, or Unknown authenticated
rows quarantine and retain their metadata and physical rows.

Only finish invokes the no-throw publication callback, with the Authority gate
released and exactly one callback attempt. A post-callback credential failure
preserves the already-published terminal and makes the owner sticky
Quarantined; no fallback publication is allowed. Retain owners retire through
the exact credential check after returning to Active. Registration nonce
allocation permanently fails at zero or maximum rather than reusing an
identity.

A stage rejection after Final freeze is handed to the authenticated frozen
abort path. Known terminal work closes the registration and clears the slot;
Unknown or contradictory authenticated evidence is Quarantined. Foreign,
stale, duplicate, or wrong-phase credentials return the DirectAbort Busy result
without clearing live rows or converting the caller into DeviceLost.
