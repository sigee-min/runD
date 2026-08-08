#pragma once

#include <kernel/program/phase.hpp>

namespace node_accel_contract::backend {

[[nodiscard]] inline rund::kernel::TilePhaseDescription Phase() {
  return rund::kernel::TilePhaseDescription{
      .phase_id = 44u,
      .tile_count = 4u,
  };
}

}  // namespace node_accel_contract::backend
