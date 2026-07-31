#pragma once

#include <math64/fixed/sqrt.hpp>
#include <math64/simd/mask.hpp>

#include <limits>

namespace rund::math64::quant {

struct RequantResult {
  simd::I64x value{};
  simd::Mask64x valid_shift{};
  [[nodiscard]] bool ok() const noexcept { return simd::All(valid_shift); }
};

[[nodiscard]] inline simd::I64x ClampI8(const simd::I64x value) noexcept {
  return Clamp(value, simd::SplatI64(std::numeric_limits<i8>::min()), simd::SplatI64(std::numeric_limits<i8>::max()));
}
[[nodiscard]] inline simd::I64x ClampI16(const simd::I64x value) noexcept {
  return Clamp(value, simd::SplatI64(std::numeric_limits<i16>::min()), simd::SplatI64(std::numeric_limits<i16>::max()));
}
[[nodiscard]] inline simd::I64x ClampU8(const simd::I64x value) noexcept {
  return Clamp(value, simd::SplatI64(0), simd::SplatI64(std::numeric_limits<u8>::max()));
}
}  // namespace rund::math64::quant
