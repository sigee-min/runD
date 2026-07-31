#pragma once

#include "model.hpp"

#include <accel/check.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr rund::AccelCheck
SegmentedReduceStatus(const rund::kernel::u32 status) noexcept {
  if ((status & kSegmentInvalid) != 0u) {
    return {false, "compute_segmented_reduce_segment_invalid"};
  }
  if ((status & kSegmentSumOverflow) != 0u) {
    return {false, "compute_segmented_reduce_sum_overflow"};
  }
  if ((status & kSegmentCountOverflow) != 0u) {
    return {false, "compute_segmented_reduce_count_overflow"};
  }
  return status == 0u
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_segmented_reduce_invalid"};
}

} // namespace rund::node::accel::detail
