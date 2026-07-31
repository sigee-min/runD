#include <cluster/placement/selection.hpp>

namespace rund::cluster {

namespace {

PlacementResult reject(const PlacementCode code) {
  return PlacementResult{.code = code};
}

PlacementResult place(const ShardRef shard, const NodeId target,
                      const PlacementEpoch epoch) {
  return PlacementResult{
      .code = PlacementCode::Placed,
      .placement = ShardPlacement{shard, target, epoch},
  };
}

} // namespace

PlacementResult place_shard(const PlacementRequest &request) {
  if (!static_cast<bool>(request.shard)) {
    return reject(PlacementCode::ShardRequired);
  }
  for (const NodeId candidate : request.candidates) {
    if (static_cast<bool>(candidate)) {
      return place(request.shard, candidate, request.epoch);
    }
  }
  return reject(PlacementCode::NodeRequired);
}

} // namespace rund::cluster
