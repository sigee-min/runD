#pragma once

#include <math32/core/model.hpp>
#include <math32/fixed/constants.hpp>

#include <bit>
#include <cstddef>
#include <limits>

namespace rund::math32::detail {

[[nodiscard]] constexpr i32 ScalarClamp(const i64 value) {
  if (value > static_cast<i64>(FixedMax)) return FixedMax;
  if (value < static_cast<i64>(FixedMin)) return FixedMin;
  return static_cast<i32>(value);
}

[[nodiscard]] constexpr u64 ScalarSatAdd(const u64 lhs, const u64 rhs) {
  const u64 max_value = std::numeric_limits<u64>::max();
  return lhs > max_value - rhs ? max_value : lhs + rhs;
}
[[nodiscard]] constexpr u64 ScalarSatSub(const u64 value, const u64 amount) {
  return value - (amount > value ? value : amount);
}
[[nodiscard]] constexpr u64 ScalarSatMul(const u64 lhs, const u64 rhs) {
  if (lhs != 0u && rhs > std::numeric_limits<u64>::max() / lhs) return std::numeric_limits<u64>::max();
  return lhs * rhs;
}
[[nodiscard]] constexpr u32 ScalarSatU32(const std::size_t value) {
  return value > static_cast<std::size_t>(std::numeric_limits<u32>::max())
             ? std::numeric_limits<u32>::max()
             : static_cast<u32>(value);
}
[[nodiscard]] constexpr u32 ScalarSatMilliRatio(const u64 numerator, const u64 denominator) {
  if (denominator == 0u) return 0u;
  const u64 scaled = numerator > std::numeric_limits<u64>::max() / 1000u
                         ? std::numeric_limits<u64>::max()
                         : numerator * 1000u;
  const u64 ratio = scaled / denominator;
  return ratio > std::numeric_limits<u32>::max() ? std::numeric_limits<u32>::max() : static_cast<u32>(ratio);
}

[[nodiscard]] constexpr u32 ScalarAbsMagnitude(const i32 value) {
  const u32 bits = std::bit_cast<u32>(value);
  return (bits & 0x80000000u) == 0u ? bits : (~bits + 1u);
}
[[nodiscard]] constexpr i32 ScalarAbs(const i32 value) {
  return value == FixedMin ? FixedMax : static_cast<i32>(ScalarAbsMagnitude(value));
}
[[nodiscard]] constexpr i32 ScalarSign(const i32 value) {
  return value > 0 ? 1 : (value < 0 ? -1 : 0);
}
[[nodiscard]] constexpr i32 ScalarClampSignedMagnitude(const bool negative, const u128 magnitude) {
  if (negative) {
    const u128 limit = u128{1} << 31u;
    if (magnitude >= limit) return FixedMin;
    return static_cast<i32>(-static_cast<i64>(magnitude));
  }
  if (magnitude > static_cast<u128>(static_cast<u32>(FixedMax))) return FixedMax;
  return static_cast<i32>(static_cast<u32>(magnitude));
}
[[nodiscard]] constexpr i32 ScalarAddWrap(const i32 lhs, const i32 rhs) {
  return std::bit_cast<i32>(std::bit_cast<u32>(lhs) + std::bit_cast<u32>(rhs));
}
[[nodiscard]] constexpr u32 ScalarAddWrapUnsigned(const u32 lhs, const u32 rhs) {
  return static_cast<u32>(lhs + rhs);
}
[[nodiscard]] constexpr i32 ScalarSubWrap(const i32 lhs, const i32 rhs) {
  return std::bit_cast<i32>(std::bit_cast<u32>(lhs) - std::bit_cast<u32>(rhs));
}
[[nodiscard]] constexpr i32 ScalarMulLow(const i32 lhs, const i32 rhs) {
  const u64 product = static_cast<u64>(std::bit_cast<u32>(lhs)) * static_cast<u64>(std::bit_cast<u32>(rhs));
  return std::bit_cast<i32>(static_cast<u32>(product));
}
[[nodiscard]] constexpr i32 ScalarMulHigh(const i32 lhs, const i32 rhs) {
  const u64 product = static_cast<u64>(std::bit_cast<u32>(lhs)) * static_cast<u64>(std::bit_cast<u32>(rhs));
  return std::bit_cast<i32>(static_cast<u32>(product >> 32u));
}
[[nodiscard]] constexpr i64 ScalarWidenMul(const i32 lhs, const i32 rhs) {
  return static_cast<i64>(lhs) * static_cast<i64>(rhs);
}
[[nodiscard]] constexpr i32 ScalarAddSat(const i32 lhs, const i32 rhs) {
  return ScalarClamp(static_cast<i64>(lhs) + static_cast<i64>(rhs));
}
[[nodiscard]] constexpr u32 ScalarAddSatUnsigned(const u32 lhs, const u32 rhs) {
  const u32 sum = lhs + rhs;
  return sum < lhs ? std::numeric_limits<u32>::max() : sum;
}
[[nodiscard]] constexpr i32 ScalarSubSat(const i32 lhs, const i32 rhs) {
  return ScalarClamp(static_cast<i64>(lhs) - static_cast<i64>(rhs));
}
[[nodiscard]] constexpr i32 ScalarMin(const i32 lhs, const i32 rhs) { return lhs < rhs ? lhs : rhs; }
[[nodiscard]] constexpr i32 ScalarMax(const i32 lhs, const i32 rhs) { return lhs > rhs ? lhs : rhs; }
[[nodiscard]] constexpr i32 ScalarClamp(const i32 value, const i32 lower, const i32 upper) {
  return ScalarMin(ScalarMax(value, lower), upper);
}
[[nodiscard]] constexpr i32 ScalarSelect(const bool condition, const i32 when_true, const i32 when_false) {
  return condition ? when_true : when_false;
}
[[nodiscard]] constexpr i32 ScalarNegPositiveFixed(const i32 value) {
  const i32 negated = static_cast<i32>(-static_cast<i64>(value));
  return value == FixedMax ? FixedMin : negated;
}
[[nodiscard]] constexpr i32 ScalarMulFixed(const i32 lhs, const i32 rhs) {
  const i64 product = static_cast<i64>(lhs) * static_cast<i64>(rhs);
  return ScalarClamp(product / static_cast<i64>(FixedScale));
}
[[nodiscard]] constexpr i32 ScalarMulFixedScaled(const i32 value, const u64 scaled_coefficient) {
  const i128 product = static_cast<i128>(value) * static_cast<i128>(scaled_coefficient);
  return ScalarClamp(static_cast<i64>(product / static_cast<i128>(FixedScale)));
}
[[nodiscard]] constexpr u64 ScalarMulUnsignedFixed(const u64 lhs, const u64 rhs) {
  const u64 product = static_cast<u64>(static_cast<u32>(lhs)) * static_cast<u64>(static_cast<u32>(rhs));
  return product >> 31u;
}
[[nodiscard]] constexpr i32 ScalarMulAddFixed(const i32 lhs, const i32 rhs, const i32 addend) {
  const i64 product = static_cast<i64>(lhs) * static_cast<i64>(rhs);
  const i64 term = product / static_cast<i64>(FixedScale);
  return ScalarClamp(term + static_cast<i64>(addend));
}
[[nodiscard]] constexpr i32 ScalarDivFixed(const i32 lhs, const i32 rhs) {
  if (rhs == 0) return lhs > 0 ? FixedMax : (lhs < 0 ? FixedMin : 0);
  const i64 numerator = static_cast<i64>(lhs) * static_cast<i64>(FixedScale);
  return ScalarClamp(numerator / static_cast<i64>(rhs));
}
[[nodiscard]] constexpr i32 ScalarRecip(const i32 value) {
  if (value == 0) return FixedMax;
  const i64 numerator = static_cast<i64>(FixedScale) * static_cast<i64>(FixedScale);
  return ScalarClamp(numerator / static_cast<i64>(value));
}
[[nodiscard]] constexpr u128 ScalarIntegerSqrt128(u128 value) {
  u128 result = 0;
  u128 bit = u128{1} << 126u;
  while (bit > value) bit >>= 2u;
  while (bit != 0u) {
    if (value >= result + bit) {
      value -= result + bit;
      result = (result >> 1u) + bit;
    } else {
      result >>= 1u;
    }
    bit >>= 2u;
  }
  return result;
}
[[nodiscard]] constexpr i32 ScalarSqrt(const i32 value) {
  if (value <= 0) return 0;
  const u128 radicand = static_cast<u128>(static_cast<u32>(value)) * static_cast<u128>(FixedScale);
  return ScalarClamp(static_cast<i64>(ScalarIntegerSqrt128(radicand)));
}
[[nodiscard]] constexpr i32 ScalarRsqrt(const i32 value) {
  if (value <= 0) return FixedMax;
  const i32 root = ScalarSqrt(value);
  return root == 0 ? FixedMax : ScalarRecip(root);
}

}  // namespace rund::math32::detail
