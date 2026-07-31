#pragma once

#include <kernel/program/skeleton/shape.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::kernel::skeleton_detail {

template <std::size_t Rank>
[[nodiscard]] constexpr const char* ValidateView(
    const void* const data,
    const Index<Rank>& shape,
    const std::array<std::ptrdiff_t, Rank>& stride,
    const u64 element_count,
    const std::size_t alignment_bytes) noexcept {
  u64 units = 0u;
  if (!ShapeProduct(shape, units)) {
    return "view_shape_overflow";
  }
  if (data == nullptr) {
    return units == 0u ? nullptr : "view_null_data";
  }
  if (alignment_bytes != 0u &&
      reinterpret_cast<std::uintptr_t>(data) % alignment_bytes != 0u) {
    return "view_alignment_failed";
  }
  if (units == 0u) {
    return nullptr;
  }
  std::ptrdiff_t min_offset = 0;
  std::ptrdiff_t max_offset = 0;
  for (std::size_t axis = 0u; axis < Rank; ++axis) {
    const u64 extent = shape[axis];
    if (extent == 0u) {
      continue;
    }
    const u64 steps = extent - 1u;
    if (steps > static_cast<u64>(std::numeric_limits<std::ptrdiff_t>::max())) {
      return "view_stride_range_overflow";
    }
    const std::ptrdiff_t signed_steps = static_cast<std::ptrdiff_t>(steps);
    const std::ptrdiff_t axis_stride = stride[axis];
    if (axis_stride == std::numeric_limits<std::ptrdiff_t>::min()) {
      return "view_stride_range_overflow";
    }
    const std::ptrdiff_t abs_stride = axis_stride < 0 ? -axis_stride : axis_stride;
    if (abs_stride != 0 &&
        signed_steps > std::numeric_limits<std::ptrdiff_t>::max() / abs_stride) {
      return "view_stride_range_overflow";
    }
    const std::ptrdiff_t delta = signed_steps * axis_stride;
    if (delta < 0) {
      if (min_offset < std::numeric_limits<std::ptrdiff_t>::min() - delta) {
        return "view_stride_range_overflow";
      }
      min_offset += delta;
    } else {
      if (max_offset > std::numeric_limits<std::ptrdiff_t>::max() - delta) {
        return "view_stride_range_overflow";
      }
      max_offset += delta;
    }
  }
  if (min_offset < 0) {
    return "view_negative_offset";
  }
  if (static_cast<u64>(max_offset) >= element_count) {
    return "view_backing_span_too_small";
  }
  return nullptr;
}

} // namespace rund::kernel::skeleton_detail
