# Placement Contract

## Scope

This page owns shard-to-node placement behavior.

Public authority:

- `/cluster/include/cluster/placement/selection.hpp`

Implementation authority:

- `/cluster/src/placement`

Verification authority:

- `/cluster/tests/contract/placement.cpp`

## Contract

Cluster placement assigns a logical shard to a node only when explicit
distributed hosting is active. It does not admit work into the kernel and does
not choose kernel packet order.

Ordinary local game and simulation code must not call cluster placement APIs to
reach the implicit local node. Local mode resolves domain work to the implicit
local node before node admission without exposing cluster as a control surface.

The placement functions support:

- first valid candidate node selection
- fail-closed reasons for missing shard or missing node

`PlacementResult` stores completion once in `PlacementCode`. `ok()`, the
explicit truth conversion, `error()`, and `exit_code()` derive from that code.
This makes `Placed`, `ShardRequired`, `NodeRequired`, and `NotPlaced` mutually
exclusive instead of permitting contradictory boolean and reason pairs.

`place_shard` selects the first candidate whose node id is
nonzero. Candidate order is caller-owned deterministic policy. The function
does not probe liveness, capacity, health, or network availability and its name
must not imply those observations.

## Update Rules

- Placement policy or result-shape changes must update this page and
  `/cluster/tests/contract/placement.cpp`.
