# Run Identity Contract

## Scope

This page owns Cluster run identity and retry attempts.

Public authority:

- `/node/include/rund/evidence/numeric/contract.hpp`
- `/cluster/include/cluster/run/identity.hpp`

Implementation authority:

- `/cluster/src/run/identity.cpp`

Verification authority:

- `/cluster/tests/contract/run.cpp`

## Contract

A `RunKey` identifies the admitted deterministic work meaning. It must
include admitted work identity, deterministic input version, program identity,
fold policy identity, numeric contract identity, capacity contract
identity, and output identity. Shard identity is optional, but when present it
must be complete.

Work, input, program, fold policy, numeric contract, capacity contract,
and output identities must be non-zero before the run key is complete. Zero is
reserved for missing identity.

`RunKey::complete()` is the sole completeness observer. Canonical `RunKey`
equality is owned by `operator==`: every base identity field and the `sharded`
discriminator must match, while the stored shard payload is compared only when
`sharded` is true. `RunAttempt::operator==` adds the
retry epoch to that canonical key equality. No parallel completeness or
same-key helper is part of the contract.

When shard identity is absent, stored shard payload is canonical non-meaning.
Completeness, equality, and derived run identifiers must ignore shard payload
unless the run key explicitly marks shard identity as present.

A retry epoch creates a distinct run attempt. It does not by itself change the
underlying deterministic run key.

Changing input version, program identity, fold policy identity,
numeric contract identity, capacity contract identity, output identity,
shard identity, checkpoint, or logical time changes the run key unless a higher
layer provides explicit equivalence evidence.

Run-bound random seed identity is part of the effective deterministic input
contract. Changing the admitted run seed or an external input record hash
changes replay evidence and must be represented as a different caller-owned
input identity by domain layers.

Replay evidence may carry this value, but Cluster does not own replay bundles,
host events, scheduler observations, IO readiness, timers, random payloads, or
trace storage.

## Update Rules

- Run-key fields or retry semantics must update this page and
  `/cluster/tests/contract/run.cpp`.
