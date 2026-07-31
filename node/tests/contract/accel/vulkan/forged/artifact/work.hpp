#pragma once

#include "../../local.hpp"

#include <array>

namespace node_accel_contract::vulkan {

inline constexpr std::size_t kForgedArtifactTileCount = 4u;

struct ForgedArtifactWork {
  std::array<rund::kernel::u32, kForgedArtifactTileCount> lhs{3u, 5u, 8u, 13u};
  std::array<rund::kernel::u32, kForgedArtifactTileCount> rhs{2u, 4u, 6u, 10u};
  std::array<TileValue, kForgedArtifactTileCount> output{};
  std::array<rund::kernel::u64, kForgedArtifactTileCount> sequence_tiles{
      0u, 1u, 2u, 3u};
  rund::kernel::u32 scale = 7u;
};

}  // namespace node_accel_contract::vulkan
