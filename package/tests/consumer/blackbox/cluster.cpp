#include "model.hpp"

#include <cluster/cluster.hpp>

namespace package_blackbox {

[[nodiscard]] int CheckCluster() {
  const rund::cluster::ShardRef shard{rund::cluster::JobId{3u},
                                      rund::cluster::ShardId{7u}};
  const std::array<rund::cluster::NodeId, 3u> candidates{
      rund::cluster::NodeId{},
      rund::cluster::NodeId{13u},
      rund::cluster::NodeId{17u},
  };
  const rund::cluster::PlacementResult first =
      rund::cluster::place_shard(rund::cluster::PlacementRequest{
          .shard = shard,
          .candidates = std::span<const rund::cluster::NodeId>{candidates},
          .epoch = rund::cluster::PlacementEpoch{19u},
      });
  if (!first) {
    return first.exit_code();
  }
  if (!first.error().empty() || first.exit_code() != 0 ||
      first.placement.node != rund::cluster::NodeId{13u} ||
      first.placement.epoch != rund::cluster::PlacementEpoch{19u}) {
    return Mismatch("cluster-placement");
  }

  const rund::cluster::RunKey key{
      .work = rund::cluster::WorkId{1u},
      .sharded = true,
      .shard = shard,
      .input = rund::cluster::InputVersion{4u},
      .time = rund::cluster::LogicalTime{5u},
      .checkpoint = rund::cluster::CheckpointId{6u},
      .program = rund::cluster::ProgramId{7u},
      .fold = rund::cluster::FoldId{8u},
      .numeric = rund::evidence::identify(rund::evidence::i32()),
      .capacity = rund::cluster::CapacityId{9u},
      .output = rund::cluster::OutputId{10u},
  };
  const rund::cluster::RetryDecision preserved = rund::cluster::evaluate_retry(
      rund::cluster::RetryRequest{key, key, rund::cluster::RetryEpoch{1u}});
  rund::cluster::RunKey changed = key;
  changed.input = rund::cluster::InputVersion{11u};
  const rund::cluster::RetryDecision rejected = rund::cluster::evaluate_retry(
      rund::cluster::RetryRequest{key, changed, rund::cluster::RetryEpoch{2u}});
  return key.complete() && preserved.preserves_identity() &&
                 preserved.reason() == "run_identity_preserved" &&
                 preserved.attempt.run == key &&
                 !rejected.preserves_identity() &&
                 rejected.reason() == "run_identity_changed"
             ? 0
             : Mismatch("cluster-retry");
}

} // namespace package_blackbox
