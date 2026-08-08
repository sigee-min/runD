#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class TilePhaseOrder : u32 {
  None = 0u,
  TileIndexAscending = 1u,
};

struct TilePhaseDescription {
  u64 phase_id = 0u;
  u64 tile_count = 0u;
  TilePhaseOrder order = TilePhaseOrder::TileIndexAscending;
};

struct TilePhaseAdmissionResult {
  bool ok = false;
  const char *reason = "tile_phase_invalid";
};

[[nodiscard]] constexpr TilePhaseAdmissionResult
ValidateTilePhaseDescription(const TilePhaseDescription &phase) noexcept {
  if (phase.phase_id == 0u) {
    return TilePhaseAdmissionResult{.reason = "tile_phase_id_invalid"};
  }
  if (phase.tile_count == 0u) {
    return TilePhaseAdmissionResult{.reason = "tile_phase_tile_count_invalid"};
  }
  if (phase.order != TilePhaseOrder::TileIndexAscending) {
    return TilePhaseAdmissionResult{.reason = "tile_phase_order_invalid"};
  }
  return TilePhaseAdmissionResult{
      .ok = true,
      .reason = "ok",
  };
}

} // namespace rund::kernel
