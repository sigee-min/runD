#include "test/assert.hpp"

#include <cluster/placement/selection.hpp>

#include <array>
#include <string_view>
#include <type_traits>
#include <utility>

static_assert(std::is_same_v<decltype(rund::cluster::PlacementResult{}.code),
                             rund::cluster::PlacementCode>);
static_assert(
    noexcept(std::declval<const rund::cluster::PlacementResult &>().ok()));
static_assert(
    noexcept(std::declval<const rund::cluster::PlacementResult &>().error()));
static_assert(noexcept(
    std::declval<const rund::cluster::PlacementResult &>().exit_code()));

int RunClusterPlacementContract() {
  constexpr rund::cluster::PlacementResult pending{};
  static_assert(!pending.ok());
  static_assert(pending.code == rund::cluster::PlacementCode::NotPlaced);
  static_assert(pending.error() == "not_placed");

  const rund::cluster::ShardRef shard{rund::cluster::JobId{10u},
                                      rund::cluster::ShardId{20u}};

  const std::array<rund::cluster::NodeId, 3> candidates{
      rund::cluster::NodeId{},
      rund::cluster::NodeId{40u},
      rund::cluster::NodeId{50u},
  };
  const rund::cluster::PlacementResult first =
      rund::cluster::place_shard(rund::cluster::PlacementRequest{
          shard, candidates, rund::cluster::PlacementEpoch{7u}});
  TEST_ASSERT(first.ok());
  TEST_ASSERT(first);
  TEST_ASSERT(first.error().empty());
  TEST_ASSERT(first.exit_code() == 0);
  TEST_ASSERT(first.code == rund::cluster::PlacementCode::Placed);
  TEST_ASSERT(first.placement.node == rund::cluster::NodeId{40u});
  TEST_ASSERT(first.placement.epoch == rund::cluster::PlacementEpoch{7u});

  const std::array<rund::cluster::NodeId, 2> invalid_candidates{
      rund::cluster::NodeId{}, rund::cluster::NodeId{}};
  const rund::cluster::PlacementResult missing_node =
      rund::cluster::place_shard(rund::cluster::PlacementRequest{
          shard, invalid_candidates, rund::cluster::PlacementEpoch{8u}});
  TEST_ASSERT(!missing_node.ok());
  TEST_ASSERT(!missing_node);
  TEST_ASSERT(missing_node.code == rund::cluster::PlacementCode::NodeRequired);
  TEST_ASSERT(missing_node.error() == "node_required");
  TEST_ASSERT(missing_node.exit_code() == 1);

  const rund::cluster::PlacementResult missing_shard =
      rund::cluster::place_shard(
          rund::cluster::PlacementRequest{rund::cluster::ShardRef{}, candidates,
                                          rund::cluster::PlacementEpoch{}});
  TEST_ASSERT(!missing_shard.ok());
  TEST_ASSERT(!missing_shard);
  TEST_ASSERT(missing_shard.code ==
              rund::cluster::PlacementCode::ShardRequired);
  TEST_ASSERT(missing_shard.error() == "shard_required");
  return 0;
}
