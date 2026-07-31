#pragma once

#include <math64/core/model.hpp>
#include <math64/fixed/constants.hpp>

#include <bit>
#include <cstddef>
#include <limits>

namespace rund::math64::detail {

[[nodiscard]] constexpr i64 ScalarClamp(const i128 value) {
  if (value > static_cast<i128>(FixedMax)) return FixedMax;
  if (value < static_cast<i128>(FixedMin)) return FixedMin;
  return static_cast<i64>(value);
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

[[nodiscard]] constexpr u64 ScalarAbsMagnitude(const i64 value) {
  const u64 bits = std::bit_cast<u64>(value);
  return (bits & 0x8000000000000000ull) == 0ull ? bits : (~bits + 1ull);
}
[[nodiscard]] constexpr i64 ScalarAbs(const i64 value) {
  return value == FixedMin ? FixedMax : static_cast<i64>(ScalarAbsMagnitude(value));
}
[[nodiscard]] constexpr i64 ScalarSign(const i64 value) { return value > 0 ? 1 : (value < 0 ? -1 : 0); }
[[nodiscard]] constexpr i64 ScalarClampSignedMagnitude(const bool negative, const u128 magnitude) {
  if (negative) {
    const u128 limit = u128{1} << 63u;
    if (magnitude >= limit) return FixedMin;
    return static_cast<i64>(-static_cast<i128>(magnitude));
  }
  if (magnitude > static_cast<u128>(static_cast<u64>(FixedMax))) return FixedMax;
  return static_cast<i64>(static_cast<u64>(magnitude));
}
[[nodiscard]] constexpr i64 ScalarAddWrap(const i64 lhs, const i64 rhs) {
  return std::bit_cast<i64>(std::bit_cast<u64>(lhs) + std::bit_cast<u64>(rhs));
}
[[nodiscard]] constexpr u64 ScalarAddWrapUnsigned(const u64 lhs, const u64 rhs) { return static_cast<u64>(lhs + rhs); }
[[nodiscard]] constexpr i64 ScalarSubWrap(const i64 lhs, const i64 rhs) {
  return std::bit_cast<i64>(std::bit_cast<u64>(lhs) - std::bit_cast<u64>(rhs));
}
[[nodiscard]] constexpr i64 ScalarMulLow(const i64 lhs, const i64 rhs) {
  const u64 product = std::bit_cast<u64>(lhs) * std::bit_cast<u64>(rhs);
  return std::bit_cast<i64>(product);
}
[[nodiscard]] constexpr i64 ScalarMulHigh(const i64 lhs, const i64 rhs) {
  const u128 product = static_cast<u128>(std::bit_cast<u64>(lhs)) * static_cast<u128>(std::bit_cast<u64>(rhs));
  return std::bit_cast<i64>(static_cast<u64>(product >> 64u));
}
[[nodiscard]] constexpr i128 ScalarWidenMul(const i64 lhs, const i64 rhs) {
  return static_cast<i128>(lhs) * static_cast<i128>(rhs);
}
[[nodiscard]] constexpr i64 ScalarAddSat(const i64 lhs, const i64 rhs) { return ScalarClamp(static_cast<i128>(lhs) + rhs); }
[[nodiscard]] constexpr u64 ScalarAddSatUnsigned(const u64 lhs, const u64 rhs) { return ScalarSatAdd(lhs, rhs); }
[[nodiscard]] constexpr i64 ScalarSubSat(const i64 lhs, const i64 rhs) { return ScalarClamp(static_cast<i128>(lhs) - rhs); }
[[nodiscard]] constexpr i64 ScalarMin(const i64 lhs, const i64 rhs) { return lhs < rhs ? lhs : rhs; }
[[nodiscard]] constexpr i64 ScalarMax(const i64 lhs, const i64 rhs) { return lhs > rhs ? lhs : rhs; }
[[nodiscard]] constexpr i64 ScalarClamp(const i64 value, const i64 lower, const i64 upper) {
  return ScalarMin(ScalarMax(value, lower), upper);
}
[[nodiscard]] constexpr i64 ScalarNegPositiveFixed(const i64 value) {
  const i64 negated = static_cast<i64>(-static_cast<i128>(value));
  return value == FixedMax ? FixedMin : negated;
}
[[nodiscard]] constexpr i64 ScalarMulFixed(const i64 lhs, const i64 rhs) {
  return ScalarClamp((static_cast<i128>(lhs) * rhs) / static_cast<i128>(FixedScale));
}
[[nodiscard]] constexpr i64 ScalarMulFixedScaled(const i64 value, const u64 scaled_coefficient) {
  return ScalarClamp((static_cast<i128>(value) * scaled_coefficient) / static_cast<i128>(FixedScale));
}
[[nodiscard]] constexpr u128 ScalarMulUnsignedFixed(const u128 lhs, const u128 rhs) {
  return (lhs * rhs) / static_cast<u128>(FixedScale);
}
[[nodiscard]] constexpr i64 ScalarMulAddFixed(const i64 lhs, const i64 rhs, const i64 addend) {
  return ScalarClamp((static_cast<i128>(lhs) * rhs) / static_cast<i128>(FixedScale) + addend);
}
[[nodiscard]] constexpr i64 ScalarDivFixed(const i64 lhs, const i64 rhs) {
  if (rhs == 0) return lhs > 0 ? FixedMax : (lhs < 0 ? FixedMin : 0);
  return ScalarClamp((static_cast<i128>(lhs) * static_cast<i128>(FixedScale)) / static_cast<i128>(rhs));
}
[[nodiscard]] constexpr i64 ScalarRecip(const i64 value) {
  if (value == 0) return FixedMax;
  const i128 scale = static_cast<i128>(FixedScale);
  return ScalarClamp((scale * scale) / static_cast<i128>(value));
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
[[nodiscard]] constexpr i64 ScalarSqrt(const i64 value) {
  if (value <= 0) return 0;
  const u128 radicand = static_cast<u128>(static_cast<u64>(value)) * static_cast<u128>(FixedScale);
  return ScalarClamp(static_cast<i128>(ScalarIntegerSqrt128(radicand)));
}
[[nodiscard]] constexpr i64 ScalarRsqrt(const i64 value) {
  if (value <= 0) return FixedMax;
  const i64 root = ScalarSqrt(value);
  return root == 0 ? FixedMax : ScalarRecip(root);
}

}  // namespace rund::math64::detail
