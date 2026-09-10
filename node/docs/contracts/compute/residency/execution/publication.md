# Execution Publication

Authority owns atomic cache metadata, while its stateless
`VirtualTransactionOwner` facet owns the bounded physical output-row lease and
an opt-in backing transaction owns all-or-none external output visibility.
Successful Output service means the exact backing write or generation-private
staging write already completed. Authority close follows that terminal; it
never precedes persistence.

External all-or-none visibility uses the separate
`VirtualBackingTransaction` capability without changing the `VirtualBacking`
vtable/layout. The implemented atomic product is deliberately narrow:
accelerator ordinary aggregate `Scan`, with no Graph/Reduce/DeviceVSM route,
a full output extent, and an output that dynamically implements this
capability. Admission does not make resident, nontransactional, or other
routes atomic; those remain explicit legacy/blocker paths. The move-only token
binds backing id, base version, logical extent, page geometry, and run
generation. The descriptor keeps physical `backing_page_bytes/count` separate
from logical `write_page_bytes/count`; ordinary `write`/`write_batch` calls
remain nontransactional.

For an admitted Scan, `begin(spec, token)` succeeds before epoch zero and
recovery covers the complete output extent. Every epoch stages its exact
writeback through the bound provider; no epoch calls the public backing
version writer. The final coordinator first prepares the provider and
preflights both Pipeline banks and the bounded physical output-cache regions,
then calls `commit(token)` exactly once. Only after a successful provider
commit does it rekey or retire the bounded rows, clear recovery, and apply the
claim-owned per-bank terminal deltas.

For this transactional Scan path, the provider keeps recovery set while a run
is incomplete or internally abandoned. A Known-no-write or preparation
failure calls `abort_known`, empties all matching provisional physical rows,
restores private control to the captured bases, and leaves bytes/version and
canonical generation unchanged. An unknown terminal calls
`quarantine_unknown`, empties the bounded rows, publishes no version or
generation, and poisons the owner. A legacy nontransactional backing instead
retains the established per-successful-write version advance and does not
provide atomic byte visibility; a failed write may leave a partial externally
visible range.
An internal cleanup-authentication conflict performs no partial deletion,
quarantines the provider and owner, and retains recovery.

The separate nontransactional Direct persistent-product success path uses this final
transaction order:

```text
all Persist writes complete with recovery retained
-> freeze Sliding and Authority Finals; externally visible owner, bytes,
   lease, and publication state remain unchanged
-> rebase every native Pipeline control owner to its post-attempt selector
-> validate backing version/recovery and both Pipeline attempts
-> fuse Authority/Sliding receipt consumption and seal private Inflight phase
-> release Authority/Sliding only; retain Pipeline/publication/backing gates
-> invoke the private noexcept continuation exactly once
   -> publish both bank Pipeline terminals while retaining both publication gates
   -> increment backing version and clear recovery under the same callback
-> reacquire Authority -> Sliding and authenticate the same plan/token/nonce
   (this authorizes final close/reset; it cannot undo visible publication)
-> release Pipeline-publication and outer backing gates
```

Any failure before receipt consumption keeps backing recovery and publishes no
new backing version. A pre-freeze rejection sets `CloseRequirement::Required`
and synchronously hands off to `close_generation`; a restored post-Final
failure is a `Closed` terminal, while rollback or control uncertainty retains
the owner as `Quarantined`. A failure to consume the receipt retains the
complete owner in quarantine. Pipeline terminals are not backing publication;
their only role here is closing the two bank attempts. The fused callback is
`noexcept`; it may not re-enter a Pipeline API, while Authority reentry sees
the retained Inflight credential as Busy and performs no mutation. A
post-callback credential contradiction preserves the already-published
terminal and quarantines the retained owner; it never retries the callback or
rolls back publication. Fallible terminal and receipt validation precedes any
may-write, lease, or public byte/version mutation. Authority may still rewrite
retry-recomputed idle `Frame::next_use` and `Frame::retain_until`; a successful
clean conflicting-view eviction is an intentional committed cache transition,
not `UnknownMayWrite` mutation. Concurrent Pipeline and backing observers
remain blocked by their existing gates, while an Authority probe returns Busy
rather than deadlocking.
Transactional W4
selection remains closed until both parity owners have this same guarantee.
Known no-write abort leaves old bytes/version intact; `UnknownMayWrite`
quarantines the provider and publishes no version. Legacy backings retain the
direct fallback and make no atomic visibility claim.

The finite `WindowRingFused` route is outside this transaction provider. Its
staged or resident recurrence has one common Final and one ordinary
output-publication/version transition after the single physical dispatch;
physical dispatch/submit counts must not be confused with the logical `Q` seed,
`Q` compute, and `2Q` internal phase counts.
