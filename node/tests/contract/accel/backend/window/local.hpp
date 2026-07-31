#pragma once

#include <accel/device.hpp>

#include "../local.hpp"

namespace node_accel_contract::backend::window {

[[nodiscard]] inline bool CanExercise(const rund::AccelDevice &pick) {
  return pick.check.ok && pick.caps.ok && pick.backend &&
         pick.caps.max_window_tiles != 0u &&
         pick.caps.max_window_tiles <= 4096u;
}

[[nodiscard]] inline rund::kernel::ComputeLimit
Limit(const rund::AccelDevice &pick) {
  return rund::kernel::ComputeLimit{
      .staging_bytes = pick.caps.staging_bytes,
      .max_window_tiles = pick.caps.max_window_tiles,
  };
}

[[nodiscard]] inline rund::kernel::ComputeDispatchWindow
Dispatch(const rund::kernel::u64 tile_count) {
  return rund::kernel::ComputeDispatchWindow{
      .begin_sequence = 0u,
      .tile_count = tile_count,
  };
}

} // namespace node_accel_contract::backend::window
