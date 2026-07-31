#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include "../local.hpp"

#include <kernel/program/compute/dsl.hpp>

namespace node_accel_contract::vulkan_stats {

struct TileValue {
  rund::kernel::u32 value = 0u;
};

[[nodiscard]] inline rund::AccelPolicy Policy() {
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
          rund::kernel::TilePhaseCapacityRequirement{
              .scratch_bytes_per_tile = 0u,
              .scratch_alignment = 1u,
              .output_shards = 4u,
              .queue_slots = 4u,
              .task_slots = 4u,
          },
  };
}

} // namespace node_accel_contract::vulkan_stats
