#include "local.hpp"

namespace rund::node::accel::detail {

[[nodiscard]] bool CpuWindowsMatchPlan(
    const rund::kernel::ComputePlan& plan,
    const rund::kernel::ComputeDispatchWindow* const windows,
    const rund::kernel::u64 window_count) noexcept {
  if (windows == nullptr || window_count != plan.dispatch_count) {
    return false;
  }
  rund::kernel::u64 expected_begin = 0u;
  for (rund::kernel::u64 index = 0u; index < window_count; ++index) {
    const rund::kernel::u64 remaining = plan.tile_count - expected_begin;
    const rund::kernel::u64 expected_tiles =
        remaining < plan.dispatch_window_tiles ? remaining
                                               : plan.dispatch_window_tiles;
    if (windows[index].begin_sequence != expected_begin ||
        windows[index].tile_count != expected_tiles) {
      return false;
    }
    expected_begin += expected_tiles;
  }
  return expected_begin == plan.tile_count;
}

}  // namespace rund::node::accel::detail
