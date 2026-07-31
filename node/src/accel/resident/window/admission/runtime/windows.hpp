#pragma once

#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/plan.hpp>

#include "../../../../backend/number.hpp"
#include "../generated/u32.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] inline bool RuntimeWindowsMatchPlan(
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings) noexcept {
  if (!rund::kernel::ComputePlanShapeValid(plan)) {
    return false;
  }
  const bool resident_identity = bindings.has_resident_output() &&
                                 bindings.sequence_tiles == nullptr &&
                                 bindings.sequence_tile_count == 0u;
  const bool resident_full_range = resident_identity && window_count == 1u &&
                                   windows != nullptr &&
                                   windows[0].begin_sequence == 0u &&
                                   windows[0].tile_count == plan.tile_count;
  if (resident_full_range &&
      !ResidentFullRangeWindowFitsGeneratedU32(plan, windows[0], bindings)) {
    return false;
  }
  if (!resident_full_range && (window_count != plan.dispatch_count ||
                               (window_count != 0u && windows == nullptr))) {
    return false;
  }

  rund::kernel::u64 total_tiles = 0u;
  rund::kernel::u64 expected_begin = 0u;
  for (rund::kernel::u64 window_index = 0u; window_index < window_count;
       ++window_index) {
    const rund::kernel::ComputeDispatchWindow window = windows[window_index];
    rund::kernel::u64 expected_tile_count = window.tile_count;
    if (!resident_full_range) {
      if (expected_begin >= plan.tile_count) {
        return false;
      }
      const rund::kernel::u64 remaining = plan.tile_count - expected_begin;
      expected_tile_count = remaining < plan.dispatch_window_tiles
                                ? remaining
                                : plan.dispatch_window_tiles;
    }
    if (window.begin_sequence != expected_begin || window.tile_count == 0u ||
        (!resident_full_range && window.tile_count != expected_tile_count) ||
        !rund::kernel::checked::add(window.begin_sequence, window.tile_count) ||
        window.begin_sequence + window.tile_count > plan.tile_count ||
        !rund::kernel::checked::add(total_tiles, window.tile_count)) {
      return false;
    }
    if (!resident_identity) {
      for (rund::kernel::u64 offset = 0u; offset < window.tile_count;
           ++offset) {
        rund::kernel::u64 tile = 0u;
        if (!bindings.sequence_tile_at(window.begin_sequence + offset, tile) ||
            tile >= plan.tile_count) {
          return false;
        }
      }
    }
    total_tiles += window.tile_count;
    expected_begin = total_tiles;
  }
  return total_tiles == plan.tile_count;
}

} // namespace rund::node::accel::detail
