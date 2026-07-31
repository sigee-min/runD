#pragma once

#include <math32/fixed/sqrt.hpp>
#include <math32/simd/mask.hpp>

#include <limits>

namespace rund::math32::quant {

struct RequantResult {
  simd::I32x value{};
  simd::Mask32x valid_shift{};
  [[nodiscard]] bool ok() const noexcept { return simd::All(valid_shift); }
};

[[nodiscard]] inline simd::I32x ClampI8(const simd::I32x value) noexcept {
  return Clamp(value, simd::SplatI32(std::numeric_limits<i8>::min()), simd::SplatI32(std::numeric_limits<i8>::max()));
}
[[nodiscard]] inline simd::I32x ClampI16(const simd::I32x value) noexcept {
  return Clamp(value, simd::SplatI32(std::numeric_limits<i16>::min()), simd::SplatI32(std::numeric_limits<i16>::max()));
}
[[nodiscard]] inline simd::I32x ClampU8(const simd::I32x value) noexcept {
  return Clamp(value, simd::SplatI32(0), simd::SplatI32(std::numeric_limits<u8>::max()));
}
}  // namespace rund::math32::quant
