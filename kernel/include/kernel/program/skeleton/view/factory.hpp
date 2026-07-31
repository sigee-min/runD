#pragma once

#include <kernel/program/skeleton/view/access.hpp>
#include <kernel/program/skeleton/view/type.hpp>
#include <kernel/program/skeleton/view/validation.hpp>

#include <array>
#include <cstddef>
#include <limits>

namespace rund::kernel {

template <typename T, std::size_t Rank>
  requires(Rank > 0u)
[[nodiscard]] constexpr View<T, Rank> view(
    T* const data,
    const Index<Rank>& shape,
    const std::size_t alignment_bytes = 0u) noexcept {
  u64 units = 0u;
  const bool product_ok = skeleton_detail::ShapeProduct(shape, units);
  const bool row_major_fits =
      product_ok && units <= static_cast<u64>(std::numeric_limits<std::ptrdiff_t>::max());
  const std::array<std::ptrdiff_t, Rank> strides =
      row_major_fits ? RowMajorStrides<T>(shape) : std::array<std::ptrdiff_t, Rank>{};
  const char* const reason =
      !product_ok ? "view_shape_overflow"
                  : (!row_major_fits
                         ? "view_stride_range_overflow"
                         : skeleton_detail::ValidateView(data,
                                                         shape,
                                                         strides,
                                                         units,
                                                         alignment_bytes));
  return View<T, Rank>{
      .data = data,
      .shape = shape,
      .stride = strides,
      .element_count = product_ok ? units : 0u,
      .alignment_bytes = alignment_bytes,
      .access = reason == nullptr ? ViewAccessPattern::Contiguous
                                  : ViewAccessPattern::Unsupported,
      .valid = product_ok && reason == nullptr,
      .reason = reason == nullptr ? "pass" : reason,
  };
}

template <typename T, std::size_t Rank>
  requires(Rank > 0u)
[[nodiscard]] constexpr View<T, Rank> view(
    T* const data,
    const Index<Rank>& shape,
    const std::array<std::ptrdiff_t, Rank>& stride,
    const u64 element_count,
    const std::size_t alignment_bytes = 0u) noexcept {
  const char* const reason =
      skeleton_detail::ValidateView(data, shape, stride, element_count, alignment_bytes);
  return View<T, Rank>{
      .data = data,
      .shape = shape,
      .stride = stride,
      .element_count = element_count,
      .alignment_bytes = alignment_bytes,
      .access = reason == nullptr ? skeleton_detail::ClassifyViewAccess(shape, stride)
                                  : ViewAccessPattern::Unsupported,
      .valid = reason == nullptr,
      .reason = reason == nullptr ? "pass" : reason,
  };
}

} // namespace rund::kernel
