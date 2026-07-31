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
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{.scratch_bytes_per_tile =
                                                         0u,
                                                     .scratch_alignment = 1u,
                                                     .output_shards = 4u,
                                                     .queue_slots = 4u,
                                                     .task_slots = 4u},
  };
}

[[nodiscard]] inline rund::kernel::TilePhaseDescription StagedPhase() {
  return rund::kernel::TilePhaseDescription{
      .phase_id = 47u,
      .tile_count = kStagedTileCount,
      .capacity =
          rund::kernel::TilePhaseCapacityRequirement{
              .scratch_bytes_per_tile = 0u,
              .scratch_alignment = 1u,
              .output_shards = kStagedTileCount,
              .queue_slots = kStagedTileCount,
              .task_slots = kStagedTileCount},
  };
}

[[nodiscard]] inline rund::kernel::TilePhasePreparedCapacity Capacity() {
  return rund::kernel::TilePhasePreparedCapacity{
      .scratch_bytes = 0u,
      .output_shards = 4u,
      .queue_slots = 4u,
      .task_slots = 4u,
  };
}

[[nodiscard]] inline rund::kernel::TilePhasePreparedCapacity StagedCapacity() {
  return rund::kernel::TilePhasePreparedCapacity{
      .scratch_bytes = 0u,
      .output_shards = kStagedTileCount,
      .queue_slots = kStagedTileCount,
      .task_slots = kStagedTileCount,
  };
}

} // namespace node_accel_contract::vulkan
