# Cluster API

Entry header: `<cluster/cluster.hpp>`

Namespace: `rund::cluster`

[Back to API Reference](https://github.com/sigee-min/runD/blob/main/wiki/API)

## When To Use

Use this optional header for distributed shard placement and retry identity.
The Cluster domain owns both the policy values and the run identities they
compare; ordinary single-node Session code does not need this surface.

## Major Types

| Family | Public Names |
| --- | --- |
| Run identity | `NodeId`, `JobId`, `ShardId`, `WorkId`, `InputVersion`, `LogicalTime`, `CheckpointId`, `ProgramId`, `FoldId`, `CapacityId`, `OutputId`, `RetryEpoch`, `ShardRef`, `RunKey`, `RunAttempt` |
| Placement | `PlacementEpoch`, `PlacementCode`, `ShardPlacement`, `PlacementRequest`, `PlacementResult` |
| Retry | `RetryCode`, `RetryRequest`, `RetryDecision` |

## Function Families

| Family | Use For |
| --- | --- |
| Placement | `place_shard` selects the first nonzero node id from the caller's deterministic candidate order. |
| Retry identity | `evaluate_retry` compares complete `RunKey` values and records the requested retry epoch. |

## Result Rules

- `PlacementResult::code` is the sole placement outcome. Check the truth
  conversion before reading `placement`; `error()` is empty on success and
  `exit_code()` returns `0` or `1`.
- `RetryDecision` is a policy decision, not an operational result. Inspect
  `code`, `preserves_identity()`, `reason()`, and `attempt`.
- `RunKey::complete()` requires every base identity. `shard` participates only
  when `sharded` is true, and then the complete `ShardRef` is required.

## Example

```cpp compile
#include <cluster/cluster.hpp>

#include <array>

int main() {
  const rund::cluster::ShardRef shard{
      .job = rund::cluster::JobId{7},
      .shard = rund::cluster::ShardId{3},
  };
  const std::array candidates{
      rund::cluster::NodeId{11},
      rund::cluster::NodeId{12},
  };
  const auto result = rund::cluster::place_shard({
      .shard = shard,
      .candidates = candidates,
      .epoch = rund::cluster::PlacementEpoch{1},
  });
  if (!result) {
    return result.exit_code();
  }
  return result.placement.shard == shard ? 0 : 2;
}
```
