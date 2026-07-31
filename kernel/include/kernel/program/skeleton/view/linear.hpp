#pragma once

#include <kernel/program/skeleton/view/type.hpp>

#include <cstddef>

namespace rund::kernel {

template <typename T, std::size_t Rank>
  requires(Rank > 0u)
struct LinearContiguousView {
  T* data = nullptr;
  u64 unit_count = 0u;
  bool valid = false;
  const char* reason = "linear_view_not_validated";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return valid; }
};

template <typename T, std::size_t Rank>
  requires(Rank > 0u)
[[nodiscard]] constexpr LinearContiguousView<T, Rank> TryLinearContiguousView(
    const View<T, Rank>& target,
    const Space<Rank>& index_space) noexcept {
  if (!index_space.valid) {
    return LinearContiguousView<T, Rank>{.reason = index_space.reason};
  }
  u64 units = 0u;
  if (!skeleton_detail::ShapeProduct(index_space.extent, units)) {
    return LinearContiguousView<T, Rank>{.reason = "skeleton_index_space_overflow"};
  }
  if (!target.valid) {
    return LinearContiguousView<T, Rank>{.reason = target.reason};
  }
  if (target.data == nullptr) {
    return LinearContiguousView<T, Rank>{.reason = "linear_view_null_data"};
  }
  if (target.access != ViewAccessPattern::Contiguous) {
    return LinearContiguousView<T, Rank>{.reason = "linear_view_not_contiguous"};
  }
  if (target.element_count < units) {
    return LinearContiguousView<T, Rank>{.reason = "linear_view_backing_span_too_small"};
  }
  for (std::size_t axis = 0u; axis < Rank; ++axis) {
    if (target.shape[axis] != index_space.extent[axis]) {
      return LinearContiguousView<T, Rank>{.reason = "linear_view_shape_mismatch"};
    }
  }
  return LinearContiguousView<T, Rank>{
      .data = target.data,
      .unit_count = units,
      .valid = true,
      .reason = "pass",
  };
}

} // namespace rund::kernel
