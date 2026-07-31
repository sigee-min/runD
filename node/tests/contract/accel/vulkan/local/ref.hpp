#pragma once

#include "tile.hpp"
#include "../../cpu/fixed.hpp"

#include <math32/fixed/scalar.hpp>
#include <math64/fixed/scalar.hpp>

#include <bit>

namespace node_accel_contract::vulkan {

[[nodiscard]] constexpr rund::kernel::i32 Predicate32(
    const bool value) noexcept {
  return value ? rund::kernel::i32{1} : rund::kernel::i32{0};
}

[[nodiscard]] constexpr rund::kernel::i64 Predicate64(
    const bool value) noexcept {
  return value ? rund::kernel::i64{1} : rund::kernel::i64{0};
}

[[nodiscard]] constexpr rund::kernel::i32 Select32(
    const rund::kernel::i32 condition,
    const rund::kernel::i32 when_true,
    const rund::kernel::i32 when_false) noexcept {
  return rund::math32::detail::ScalarSelect(condition != 0,
                                            when_true,
                                            when_false);
}

[[nodiscard]] constexpr rund::kernel::i64 Select64(
    const rund::kernel::i64 condition,
    const rund::kernel::i64 when_true,
    const rund::kernel::i64 when_false) noexcept {
  return condition != 0 ? when_true : when_false;
}

[[nodiscard]] constexpr rund::kernel::i32 ExpandedReference32(
    const rund::kernel::i32 lhs,
    const rund::kernel::i32 rhs,
    const rund::kernel::i32 scale,
    const rund::kernel::i32 bias,
    const rund::kernel::i32 lo,
    const rund::kernel::i32 hi,
    const rund::kernel::i32 marker) noexcept {
  const rund::kernel::i32 mixed = cpu::fixed::QuantizeMulAdd(
      lhs, scale, static_cast<rund::kernel::i32>(bias - rhs));
  const rund::kernel::i32 bounded =
      rund::math32::detail::ScalarClamp(mixed, lo, hi);
  const rund::kernel::i32 extreme = rund::math32::detail::ScalarMin(
      rund::math32::detail::ScalarMax(lhs, rhs), hi);
  return Select32(Predicate32(std::bit_cast<rund::kernel::u32>(lhs) ==
                              std::bit_cast<rund::kernel::u32>(marker)),
                  Predicate32(rhs <= lhs),
                  Select32(Predicate32(lhs < rhs), bounded, extreme));
}

[[nodiscard]] constexpr rund::kernel::i64 ExpandedReference64(
    const rund::kernel::i64 lhs,
    const rund::kernel::i64 rhs,
    const rund::kernel::i64 scale,
    const rund::kernel::i64 bias,
    const rund::kernel::i64 lo,
    const rund::kernel::i64 hi,
    const rund::kernel::i64 marker) noexcept {
  const rund::kernel::i64 mixed = cpu::fixed::QuantizeSum<rund::kernel::i64>(
      cpu::fixed::QuantizeProduct(lhs, scale),
      static_cast<rund::kernel::i128>(bias) - rhs);
  const rund::kernel::i64 bounded =
      rund::math64::detail::ScalarClamp(mixed, lo, hi);
  const rund::kernel::i64 extreme = rund::math64::detail::ScalarMin(
      rund::math64::detail::ScalarMax(lhs, rhs), hi);
  return Select64(Predicate64(std::bit_cast<rund::kernel::u64>(lhs) ==
                              std::bit_cast<rund::kernel::u64>(marker)),
                  Predicate64(rhs <= lhs),
                  Select64(Predicate64(lhs < rhs), bounded, extreme));
}

}  // namespace node_accel_contract::vulkan
