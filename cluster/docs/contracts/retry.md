# Retry Identity Contract

## Scope

This page owns Cluster retry identity checks over canonical `RunKey` values.

Public authority:

- `/cluster/include/cluster/retry/identity.hpp`
- `/cluster/include/cluster/run/identity.hpp`

Implementation authority:

- `/cluster/src/retry`

Verification authority:

- `/cluster/tests/contract/retry.cpp`

## Contract

Cluster retry preserves run identity only when the previous and next complete
run keys match. If the run key changes, the retry must be
reported as a changed run identity.

The retry epoch records the attempt. It does not hide a changed deterministic
run key.

Checkpoint changes are run key changes.

`RetryCode` is the single policy outcome. `preserves_identity()` and `reason()`
derive from that code. A false predicate is a valid identity-policy outcome
rather than an operational failure, so this record deliberately does not
publish result-style `error()` or `exit_code()` observers. `attempt` always
records the requested next key and epoch.

## Update Rules

- Retry identity or equivalence-proof changes must update this page and
  `/cluster/tests/contract/retry.cpp`.
