# Transactional Scan State

This page owns the private state boundary for the one implemented
transaction-capable Virtual route. External provider state remains in
`compute/virtual/run/transaction.{hpp,cpp}`; physical output-row admission,
tagging, cleanup, and lease disposition are directly owned by the stateless
`compute/device/residency/registry/transaction_owner.{hpp,cpp}` facet over the
single Authority gate/frame table.

The guarantee applies only to an accelerator ordinary aggregate Scan that is
non-Graph, non-Reduce, non-DeviceVSM, covers the complete output backing, and
whose output dynamically implements `VirtualBackingTransaction`. Admission
binds one move-only provider token before epoch zero and requires recovery for
the complete output extent. Each epoch stages exact Authority writeback into
the provider shadow; no epoch publishes a backing version.

The final coordinator runs provider preparation, deferred two-bank Pipeline
preflight, bounded physical output-row preflight, and base backing identity /
version checks before one provider `commit`. The post-commit core is
allocation-free: it rekeys or retires the bounded rows, clears recovery, and
applies the actual per-bank terminal deltas. No logical-Q cache scan or second
publication owner is permitted.

Known no-write and preparation failures call `abort_known`, empty every
matching provisional physical row, restore private control to the captured
bases, and preserve bytes, backing version, canonical generation, parity,
commit/discard counts, and observation state. An unknown terminal calls
`quarantine_unknown`, empties the bounded rows, publishes no public version or
generation, and poisons the owner; private control is not restored.
An internal cleanup-authentication conflict performs no partial deletion,
quarantines the provider and owner, and retains recovery.

Multi-input GraphResident staged inputs use the public
`VirtualBackingReadCohort` member side interface. Every member must return the same shared
`VirtualReadCohort`, with one valid `VirtualCohortId` and lane limit 1..2; the
runner retains the provider through one canonical-order synchronous join and
reauthenticates all members before native preparation completes. A missing or
mismatched provider declines to ordinary Host execution before preparation;
after preparation, identity uncertainty is terminal. The out result uses
`joined=true` only after the cohort has joined, reports completed-byte prefix
and first failed member/page for a joined known failure, and uses
`UINT64_MAX` failure sentinels on success. Single-input staged inputs use the
existing synchronous scalar `read_pages` contract and do not require a cohort.
Providers retain neither request spans nor callback reentry. Resident Graph endpoints bypass this cohort and
use their direct authenticated bindings. The dynamic U64 Tile proof includes
the all-staged `S=5,Q=6,C=2,B=3` case with 15 steps, one submit/controller,
one Final/publication/version, and zero Host epoch callbacks.

Graph Forecast failures use the same Authority-owned bounded capability rule:
the exact owner/plan/token/generation/coordinate remains retained when alias
cleanup cannot close it. While retained, admission and publication are
sticky-blocked; only the post-cancel, post-ticket-cleanup recovery caller may
close the authenticated epoch. The stateless
`registry/graph_forecast_owner.{hpp,cpp}` owns that lifecycle over the same
Authority gate/table; it adds no state authority. DeviceVSM registration release crosses its
private accelerator callback boundary and requires `Done`.

Resident backings, non-provider or partial-extent backings, Graph/Reduce/
DeviceVSM routes, nontransactional writeback, and readers that bypass the
backing/provider authority remain legacy or blocked. This state does not
claim atomicity for those paths or for an external direct reader.
