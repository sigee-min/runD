#pragma once

#include "fixed.hpp"

#include <kernel/core/checked.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

[[nodiscard]] inline DispatchWindowStorage
DispatchWindows(const rund::kernel::ComputePlan &plan) {
  if (plan.dispatch_count == 0u || plan.dispatch_window_tiles == 0u ||
      plan.tile_count == 0u) {
    return RejectDispatchWindows("compute_plan_invalid");
  }
  if (plan.dispatch_count == 1u) {
    return OneDispatchWindow(plan);
  }
  DispatchWindowStorage storage{};
  if (plan.dispatch_count >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return RejectDispatchWindows("compute_plan_invalid");
  }
  storage.overflow_windows.reserve(
      static_cast<std::size_t>(plan.dispatch_count));
  std::uint64_t begin = 0u;
  for (std::uint64_t index = 0u; index < plan.dispatch_count; ++index) {
    if (begin >= plan.tile_count) {
      return RejectDispatchWindows("compute_plan_invalid");
    }
    const std::uint64_t remaining = plan.tile_count - begin;
    const std::uint64_t tile_count = remaining < plan.dispatch_window_tiles
                                         ? remaining
                                         : plan.dispatch_window_tiles;
    std::uint64_t end = 0u;
    if (tile_count == 0u ||
        !rund::kernel::checked::add(begin, tile_count, end)) {
      return RejectDispatchWindows("compute_plan_invalid");
    }
    storage.overflow_windows.push_back(rund::kernel::ComputeDispatchWindow{
        .begin_sequence = begin,
        .tile_count = tile_count,
    });
    ++storage.window_count;
    begin = end;
  }
  if (begin != plan.tile_count || storage.window_count != plan.dispatch_count) {
    return RejectDispatchWindows("compute_plan_invalid");
  }
  return storage;
}

[[nodiscard]] inline DispatchWindowStorage
ResidentDispatchWindows(const rund::kernel::ComputePlan &plan,
                        const bool resident_identity, const bool real_backend) {
  if (resident_identity && real_backend) {
    return FullRangeDispatchWindow(plan);
  }
  return DispatchWindows(plan);
}

} // namespace rund::node::accel::detail
