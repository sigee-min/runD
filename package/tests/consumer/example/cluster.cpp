#include <cluster/cluster.hpp>

#include <array>

int main() {
  const rund::cluster::ShardRef shard{
      rund::cluster::JobId{10u},
      rund::cluster::ShardId{20u},
  };
  constexpr std::array<rund::cluster::NodeId, 2u> candidates{
      rund::cluster::NodeId{40u},
      rund::cluster::NodeId{50u},
  };
  const rund::cluster::PlacementResult placed =
      rund::cluster::place_shard(rund::cluster::PlacementRequest{
          shard, candidates, rund::cluster::PlacementEpoch{7u}});
  if (!placed) {
    return placed.exit_code();
  }
  return placed.placement.node == candidates.front() ? 0 : 2;
}
