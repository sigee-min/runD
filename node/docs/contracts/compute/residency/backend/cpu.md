# CPU Residency

The CPU Host graph product uses the canonical graph-resident workload as a
single Host invocation, but its selected route is `GraphPointwise` with
`CpuRolling`, not the accelerator `GraphResident` owner. Its fixed evidence
checks remain Host-owned: three distinct input backings, one output backing,
frame capacity two, fifteen input page reads, five output page writes, and no
accelerator owner or command submission. The same evidence requires zero
upload, download, external-roundtrip, and physical page-in bytes; CPU dispatch
count is intentionally not used as a zero-valued route predicate.

CPU graph failure ownership is preallocated before the first Authority
mutation. State, Abort, and its quarantine holder share one
`shared_ptr<CpuReceiptBook>`; the Book is never moved or copied, so its
address/domain cannot change. The fixed receipt book reserves each role/bank
slot before `begin`, minting a flat opaque `{Authority-owner, book-domain,
role-slot, nonce}` key.
The owner identity is Authority-minted, nonwrapping, and immutable; a foreign,
zero, or exhausted owner is rejected before mutation. The move-only `Permit`
carries that key through `Free -> Reserved -> Prepared -> Armed -> Free`:
the Authority holds exactly one pending reservation while the serialized CPU
`begin`/`begin_graph_epoch` attempt is in flight; the exact key is required,
every failed attempt clears that pending reservation, and a successful begin
moves it into the returned LeaseSlot before clearing pending. Other already
Armed CPU slots may remain live concurrently, but a pending or Reserved slot
is never selected or overwritten by another begin.
`Prepared` records the exact nonzero Authority token/generation, Authority
confirms the same key under its gate, and the book finalizes only after that
confirmation. A failed begin cancels only the reservation; a Busy/quarantined
close retains the prepared credential. Authority's fixed
`CycleAuthorityState::graph_persists`
LeaseSlot is the sole retry/journal owner and authenticates the complete
prepared identity before retry consumption. A Known failure rolls back the
authenticated GraphPersist journal into the Authority slot's `RetryReady`
state and remains retryable; the first CPU graph's
`begin_cpu_graph_epoch_retry` consumes every matching row in the same
Authority-gated preflight/commit that publishes its first epoch. A domain-only
consume is not exposed. An Unknown or stale credential remains sticky in the
Authority-owned quarantine. RetryReady is not a Ticket or Book mirror. The
Book retains only a fixed, detached Unknown credential snapshot, never a
move-only `GraphPersist` capability. The prepared identity stores the topology as an
exact `(hi, lo)` pair, not an XOR scalar; equality remains collision-free even
when the two components are equal.

The receipt implementation is physically split by lifecycle:
`virtual/graph/reduce/receipts/book.cpp` owns fixed bind/reserve admission;
`book/lookup.cpp` owns credential queries, `book/unknown.cpp` owns Unknown
retention/recovery, and `book/lifecycle.cpp` owns handle/permit detachment and
slot reset. `permit.cpp` owns move/cancel permit transitions, `receipt.cpp`
owns receipt handles and queries, and
`quarantine.cpp` owns the raw quarantine holder lifecycle. `receipts.hpp`
retains only stable layouts, declarations, and trivial accessors. Stateless
`CpuGraphOwner` owns CPU reservation, epoch confirmation/close, and quarantine
entry/discard algorithms while borrowing Authority's sole gate, frame table,
credentials, and `CpuGraphAuthorityState`; Authority remains the sole
registry/quarantine state owner and locked Graph admission boundary.

Unknown capability. The quarantine stores only raw Pipeline/Pool
identity plus its exact typed Pool/Authority binding and a temporary
self-reference during `Ready -> Armed -> Held`; it does not retain a strong
Pipeline/Pool root. A retained holder returns to `Ready` only after the
authenticated teardown commit. The book is run-thread confined and is never
touched while the Authority gate is held.

Graph diagnostic provenance is first-wins for `(epoch, batch, stage, phase,
check, token, generation)` and may carry one flat Authority `CloseInfo`.
Reservation failure occurs before Authority mutation; a failed `begin` cancels
the reserved permit, while a successful begin publishes only its exact
token/generation into the already sealed `Prepared` slot. A Reserved,
Prepared, or Armed slot is non-idle and cannot be reset or reused until the
matching permit/receipt is cancelled or an authenticated CPU close clears it.
A slot may retain its one quiet bound receipt handle while `Free`, but `idle`
still rejects any live permit, active key snapshot, or credential; authenticated
clear/reset_cred preserves that handle, while only explicit receipt detach,
final reset, `drop_handles`, or receipt destruction invalidates it before the
slot is rebound.
`live_permits` counts the live move-only Permit objects exactly: cancellation,
authenticated close, and Reserved recovery clear the Permit local only after
the Book has reset the exact key, and `reset` itself never changes the count.
Normal epoch close uses `reset_cred`: it clears only key/token/generation,
permit, and Armed state while preserving the fixed bank/role/Authority and the
quiet `CpuEpochReceipt` handle. The same bound handle can reserve the next
epoch; only explicit receipt destruction/detach or final Book `drop_handles`
severs the Book/slot binding.
The quarantine snapshot records `live_receipts`, `live_permits`, every Slot's
state/key/token/generation, and each Unknown locator's terminal/quarantined
flags. Stale or nonzero Free metadata rejects teardown before mutation. The
GraphPersist coordinate is nonzero and equal in its Ticket, Authority row, and
Unknown locator.
Drain rollback, RetryReady recovery/discard, and Unknown teardown share the
same no-allocation undo check. It requires Host/Output bounds, unique
Writeback frame witnesses, exact current/undo keys and dirty extents, and
unchanged frame identity fields. Direct rows carry an empty identity; CPU rows
require a valid identity whose plan equals the Authority plan. A domain-zero
Unknown is cleared from its Ticket after the Authority row becomes sticky
terminal; a nonzero CPU-domain Unknown keeps the Ticket for Book retention.
Unknown termination arms the shared Book/raw-identity/self holder, retains it
through Authority first, then drops self only on successful retain before
dropping stack handles; a valid lifecycle makes retain rejection unreachable.
If the state later tears down with an Authority-owned holder, the destructor
holds Pool execution serialization and invokes the exact Authority discard;
the discard preflights every CPU epoch, GraphPersist row, frame claim, and
native-inflight marker before rollback/invalidation and row clearing. This is
diagnostic only and does not alter terminal ownership.

CPU is positively admitted and executes directly in authoritative Host banks.
It retains the existing cold worker route, has no H2D/D2H payload, and reports
no transfer overlap. Ordinary dense CPU Buffers remain valid non-virtual
storage but are not a second residency authority.

The CPU Graph epoch contract is Direct/GraphPointwise-only: its exact bound
receipt is the only `activate`/`resume` path, while cycle execution and the
generic `complete` path reject CPU-tagged leases. A CPU epoch therefore closes
through the authenticated CPU bridge, never through a competing cycle or
generic fallback.

Direct and Reduce rolling execution enter a separately emitted CPU overlap
symbol. The shared policy body is specialized with accelerator and cycle
capabilities both false, so its epoch loop contains no coherent Device view,
prefetch-transfer receipt, Device migration/download, or cycle-journal call.
The one backend dispatch remains outside that loop. CPU also passes the
whole-execution gate before any bounded-window or Q=1 prepared owner is
constructed. This is structural isolation; a no-regression throughput claim
still requires paired measurement from the same source revision.
