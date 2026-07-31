#pragma once

#include <kernel/program/skeleton/shape.hpp>

#include <array>
#include <cstddef>
#include <limits>

namespace rund::kernel {

namespace skeleton_detail {

template <std::size_t Rank>
[[nodiscard]] constexpr ViewAccessPattern ClassifyViewAccess(
    const Index<Rank>& shape,
    const std::array<std::ptrdiff_t, Rank>& stride) noexcept {
  u64 units = 0u;
  if (!ShapeProduct(shape, units)) {
    return ViewAccessPattern::Unsupported;
  }
  if (units == 0u) {
    return ViewAccessPattern::Contiguous;
  }
  bool any_zero_stride = false;
  bool all_zero_stride = true;
  for (std::size_t axis = 0u; axis < Rank; ++axis) {
    const bool zero_stride = stride[axis] == 0;
    any_zero_stride = any_zero_stride || zero_stride;
    all_zero_stride = all_zero_stride && zero_stride;
  }
  if (all_zero_stride) {
    return ViewAccessPattern::BroadcastZeroStride;
  }
  std::array<std::ptrdiff_t, Rank> row_major{};
  std::ptrdiff_t expected_stride = 1;
  for (std::size_t offset = 0u; offset < Rank; ++offset) {
    const std::size_t axis = Rank - 1u - offset;
    row_major[axis] = expected_stride;
    if (shape[axis] >
        static_cast<u64>(std::numeric_limits<std::ptrdiff_t>::max() / expected_stride)) {
      return any_zero_stride ? ViewAccessPattern::BroadcastZeroStride
                             : ViewAccessPattern::StridedAffine;
    }
    expected_stride *= static_cast<std::ptrdiff_t>(shape[axis]);
  }
  bool contiguous = true;
  for (std::size_t axis = 0u; axis < Rank; ++axis) {
    contiguous = contiguous && stride[axis] == row_major[axis];
  }
  if (contiguous) {
    return ViewAccessPattern::Contiguous;
  }
  return any_zero_stride ? ViewAccessPattern::BroadcastZeroStride
                         : ViewAccessPattern::StridedAffine;
}

} // namespace skeleton_detail

template <typename T = void, std::size_t Rank>
  requires(Rank > 0u)
[[nodiscard]] constexpr std::array<std::ptrdiff_t, Rank> RowMajorStrides(
    const Index<Rank>& shape) noexcept {
  std::array<std::ptrdiff_t, Rank> strides{};
  std::ptrdiff_t stride = 1;
  for (std::size_t offset = 0u; offset < Rank; ++offset) {
    const std::size_t axis = Rank - 1u - offset;
    strides[axis] = stride;
    stride *= static_cast<std::ptrdiff_t>(shape[axis]);
  }
  return strides;
}

} // namespace rund::kernel
