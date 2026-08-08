#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include "tile.hpp"

#include <kernel/program/phase.hpp>

namespace node_accel_contract::vulkan {

[[nodiscard]] inline rund::AccelPolicy RequiredPolicy() {
  rund::AccelPolicy policy{};
  policy.preferred[0] = rund::AccelApi::Vulkan;
  policy.preferred_count = 1u;
  policy.allow_fake = false;
  return policy;
}

[[nodiscard]] inline rund::kernel::TilePhaseDescription Phase() {
  return rund::kernel::TilePhaseDescription{
      .phase_id = 46u,
      .tile_count = 4u,
  };
}

[[nodiscard]] inline rund::kernel::TilePhaseDescription StagedPhase() {
  return rund::kernel::TilePhaseDescription{
      .phase_id = 47u,
      .tile_count = kStagedTileCount,
  };
}

} // namespace node_accel_contract::vulkan
