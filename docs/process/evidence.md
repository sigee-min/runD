# Evidence

Evidence-based work is the highest priority in this repository.
This page owns the repository evidence classes and completion-claim discipline
required to support those claims.

## Accepted Evidence

- Checked-in docs that own the affected subsystem.
- Public headers and implementation files that define contracts.
- Contract tests, focused CI checks, and generated or measured evidence.
- Measurements with command, platform, input, and output recorded.
- External references only when they are primary, active, and cited.

## Discipline By Claim Type

| Claim | Required Evidence |
| --- | --- |
| Mathematical | Formula, invariant, bound, proof sketch, or counterexample. |
| Computer architecture | Execution model, memory model, cache behavior, ordering, synchronization, or platform constraint. |
| Digital circuit | Bit width, overflow behavior, fixed-point law, state transition, rounding, latency, or data path reasoning. |
| Determinism | Ordering, identity, schedule, seed, reduction tree, or hash evidence. |
| Performance | Measurement or explicitly labeled model with assumptions. |

## Repository Rejection Rules

- Do not treat intuition as evidence.
- Fix documentation that disagrees with the checked implementation before
  using it as evidence.
- Do not claim performance from code shape alone.
- Do not treat a task-local plan, handoff, or copied checklist as current
  process authority or completion evidence.
- Do not claim a performance hard cut targets the right layer without an
  immediate baseline and a bottleneck recheck. If duplicate authority,
  adapters, or fallback paths are on the measured hot path, classify that cost
  before replacing the architecture around it.
- Do not claim determinism without proving ordering and identity.

## Verification Manifest Identity

Fresh verification owns two product-source snapshots: `before`, captured
before configure/build, and `after`, captured after the route executes. Each
ordered row contains the file SHA-256, executable identity, and path. Source
stability is byte-for-byte equality of those manifests, so content, admission,
path, and execution-mode changes are all visible. A mismatch is failure
evidence and cannot be repaired by taking another snapshot.

Contracts selected between those boundaries reuse `before` when they need to
inspect the registered live-root manifest. This preserves their membership
assertions without a third O(product-source bytes) traversal. A direct CTest
run without a fresh-route boundary forms its own live snapshot instead.

The stability check brackets creation and sealing of the exact `after` file
with the Git revision and full porcelain worktree bytes. Any change to that
identity fails closed. It then records the bracketed revision and dirty bit in
a second sealed identity file. Evidence recording verifies both seals, copies both
artifacts into a staging packet, and publishes the complete packet with one
rename. It never re-reads live revision or worktree state. `run.tsv` records
`source_manifest_kind=verification_after`; its `source_manifest_sha256` must be
the SHA-256 of the packet's `source-manifest.tsv`, and
`source_identity_sha256` must name `source-identity.tsv`. Recording never
hashes the live product source tree, so a post-verification edit or commit
cannot be represented as part of the completed verification.

`package/cmake/identity.cmake` is the single structured owner for source
manifest and source identity sealing, the source-identity row schema,
canonical atomic writing, and validated reading.
`tools/internal/source/verify` supplies the bracketed facts and
`tools/internal/source/manifest/adopt.cmake` consumes the validated fields;
neither mirrors
the row grammar or seal policy.

`tools/evidence/status` is the launch-closure diagnostic. It forms one current
product-source manifest, selects the local host's required Debug, sanitizer,
platform, Release, artifact, leak, and measurement routes, and validates the
newest immutable packet for each route. A route is closed only when its newest
packet passed, its copied manifest and identity still match the hashes in
`run.tsv`, and its manifest equals the current manifest. Measurement packets
also require the current host, sealed raw-log and comparator-result hashes,
and a byte-identical fresh comparison against the checked-in baseline. An
atomic measurement-attempt marker exposes `in-progress` and setup failure
before a newer packet exists, so an older pass cannot mask the latest attempt.
Missing, in-progress, failed, corrupt, and stale results remain distinct
states. The diagnostic does not build or execute a workload; replaying the
pure comparator only validates recorded evidence and cannot turn an older
packet into evidence for current bytes.
