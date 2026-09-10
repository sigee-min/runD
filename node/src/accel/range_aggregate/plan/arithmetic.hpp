#pragma once

#include "../model/plan.hpp"

#include <cstdint>

namespace rund::node::accel::detail {
namespace range_plan_detail {

inline constexpr rund::kernel::u128 kU128Maximum =
    ~static_cast<rund::kernel::u128>(0u);

[[nodiscard]] constexpr bool Add(const rund::kernel::u128 left,
                                 const rund::kernel::u128 right,
                                 rund::kernel::u128 &out) noexcept {
  if (left > kU128Maximum - right) {
    return false;
  }
  out = left + right;
  return true;
}

[[nodiscard]] constexpr bool Multiply(const rund::kernel::u128 left,
                                      const rund::kernel::u128 right,
                                      rund::kernel::u128 &out) noexcept {
  if (right != 0u && left > kU128Maximum / right) {
    return false;
  }
  out = left * right;
  return true;
}

[[nodiscard]] constexpr bool
Accumulate(rund::kernel::u128 &target,
           const rund::kernel::u128 value) noexcept {
  return Add(target, value, target);
}

[[nodiscard]] constexpr bool
AccumulateProduct(rund::kernel::u128 &target, const rund::kernel::u128 left,
                  const rund::kernel::u128 right) noexcept {
  rund::kernel::u128 product = 0u;
  return Multiply(left, right, product) && Accumulate(target, product);
}

[[nodiscard]] constexpr bool
AccumulateBytes(rund::kernel::u128 &target, const rund::kernel::u128 elements,
                const rund::kernel::u32 element_bytes) noexcept {
  return AccumulateProduct(target, elements, element_bytes);
}

[[nodiscard]] constexpr rund::kernel::u64
Groups(const rund::kernel::u64 count, const rund::kernel::u32 width) noexcept {
  return width == 0u ? 0u
                     : count / width +
                           static_cast<rund::kernel::u64>(count % width != 0u);
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
