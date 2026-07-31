#pragma once

#include <cluster/placement/epoch.hpp>
#include <cluster/run/identity.hpp>

#include <cstdint>
#include <span>
#include <string_view>

namespace rund::cluster {

struct ShardPlacement {
  ShardRef shard{};
  NodeId node{};
  PlacementEpoch epoch{};
};

struct PlacementRequest {
  ShardRef shard{};
  std::span<const NodeId> candidates{};
  PlacementEpoch epoch{};
};

enum class PlacementCode : std::uint8_t {
  NotPlaced = 0u,
  ShardRequired = 1u,
  NodeRequired = 2u,
  Placed = 3u,
};

struct PlacementResult {
  PlacementCode code = PlacementCode::NotPlaced;
  ShardPlacement placement{};

  [[nodiscard]] constexpr bool ok() const noexcept {
    return code == PlacementCode::Placed;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok();
  }
  [[nodiscard]] constexpr std::string_view error() const noexcept {
    switch (code) {
    case PlacementCode::NotPlaced:
      return "not_placed";
    case PlacementCode::ShardRequired:
      return "shard_required";
    case PlacementCode::NodeRequired:
      return "node_required";
    case PlacementCode::Placed:
      return {};
    }
    return "placement_code_invalid";
  }
  [[nodiscard]] constexpr int exit_code() const noexcept {
    return ok() ? 0 : 1;
  }
};

[[nodiscard]] PlacementResult place_shard(const PlacementRequest &request);

} // namespace rund::cluster
