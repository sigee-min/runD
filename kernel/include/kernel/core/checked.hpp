#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel::checked {

[[nodiscard]] constexpr bool add(const u64 left, const u64 right) noexcept {
  return left <= (~u64{0u} - right);
}

[[nodiscard]] constexpr bool add(const u64 left, const u64 right,
                                 u64 &value) noexcept {
  if (!add(left, right)) {
    return false;
  }
  value = left + right;
  return true;
}

[[nodiscard]] constexpr bool sub(const u64 value, const u64 base,
                                 u64 &difference) noexcept {
  if (value < base) {
    return false;
  }
  difference = value - base;
  return true;
}

[[nodiscard]] constexpr bool mul(const u64 left, const u64 right) noexcept {
  return right == 0u || left <= (~u64{0u} / right);
}

[[nodiscard]] constexpr bool mul(const u64 left, const u64 right,
                                 u64 &value) noexcept {
  if (!mul(left, right)) {
    return false;
  }
  value = left * right;
  return true;
}

[[nodiscard]] constexpr bool mul(const u64 first, const u64 second,
                                 const u64 third, u64 &value) noexcept {
  u64 pair = 0u;
  if (!mul(first, second, pair) || !mul(pair, third)) {
    return false;
  }
  value = pair * third;
  return true;
}

[[nodiscard]] constexpr u64 ceil(const u64 value, const u64 divisor) noexcept {
  return divisor == 0u
             ? 0u
             : value / divisor + static_cast<u64>(value % divisor != 0u);
}

} // namespace rund::kernel::checked
