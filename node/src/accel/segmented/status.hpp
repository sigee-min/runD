#pragma once

#include <accel/check.hpp>

#include <cstdint>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr rund::AccelCheck
SegmentedScanStatus(const std::uint32_t status) noexcept {
  if (status == 1u) {
    return {false, "compute_segmented_scan_sum_overflow"};
  }
  if (status == 2u) {
    return {false, "compute_segmented_scan_segment_invalid"};
  }
  return status == 0u
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, "compute_segmented_scan_invalid"};
}

} // namespace rund::node::accel::detail
