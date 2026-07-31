#pragma once

#include "storage.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline DispatchWindowStorage OneDispatchWindow(
    const rund::kernel::ComputePlan& plan) noexcept {
  if (plan.dispatch_window_tiles < plan.tile_count) {
    return RejectDispatchWindows("compute_plan_invalid");
  }
  DispatchWindowStorage storage{};
  storage.inline_windows[0] = rund::kernel::ComputeDispatchWindow{
      .begin_sequence = 0u,
      .tile_count = plan.tile_count,
  };
  storage.window_count = 1u;
  return storage;
}

[[nodiscard]] inline DispatchWindowStorage FullRangeDispatchWindow(
    const rund::kernel::ComputePlan& plan) noexcept {
  if (plan.tile_count == 0u) {
    return RejectDispatchWindows("compute_plan_invalid");
  }
  DispatchWindowStorage storage{};
  storage.inline_windows[0] = rund::kernel::ComputeDispatchWindow{
      .begin_sequence = 0u,
      .tile_count = plan.tile_count,
  };
  storage.window_count = 1u;
  return storage;
}

}  // namespace rund::node::accel::detail
