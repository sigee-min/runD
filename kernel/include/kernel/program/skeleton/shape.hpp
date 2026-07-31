#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/skeleton/model.hpp>

#include <type_traits>

namespace rund::kernel::skeleton_detail {

template <typename T>
[[nodiscard]] constexpr bool Negative(const T value) noexcept {
  if constexpr (std::is_signed_v<T>) {
    return value < 0;
  } else {
    return false;
  }
}

template <std::size_t Rank>
[[nodiscard]] constexpr bool ShapeProduct(const Index<Rank> &shape,
                                          u64 &out) noexcept {
  u64 units = 1u;
  for (const u64 extent : shape) {
    if (extent == 0u) {
      out = 0u;
      return true;
    }
    if (!checked::mul(units, extent, units)) {
      return false;
    }
  }
  out = units;
  return true;
}

} // namespace rund::kernel::skeleton_detail
