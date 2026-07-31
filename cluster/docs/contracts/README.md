# Cluster Contracts

Stable `/cluster` behavior lives here. Cluster owns its distributed identities
and consumes only the numeric evidence id contract, never Kernel internals.

## Contents

| Page | Owns |
| --- | --- |
| [Run identity](./run.md) | Canonical distributed run keys and attempts. |
| [Placement](./placement.md) | Shard-to-node placement behavior. |
| [Retry](./retry.md) | Retry identity checks over Cluster run keys. |

## Rule

Contract pages describe distributed policy only and must name the public and
verification surfaces that prove them.
