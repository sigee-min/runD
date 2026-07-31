#pragma once

#include <kernel/program/phase.hpp>

#include <cstddef>

namespace node_accel_contract::backend {

[[nodiscard]] inline rund::kernel::TilePhaseDescription Phase() {
  return rund::kernel::TilePhaseDescription{
      .phase_id = 44u,
      .tile_count = 4u,
      .capacity = rund::kernel::TilePhaseCapacityRequirement{
          .scratch_bytes_per_tile = 0u,
          .scratch_alignment = 1u,
          .output_shards = 4u,
          .queue_slots = 4u,
          .task_slots = 4u,
      },
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

}  // namespace node_accel_contract::backend
