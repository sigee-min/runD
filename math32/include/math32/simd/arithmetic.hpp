#pragma once

#include <math32/simd/compare.hpp>
#include <math32/simd/select.hpp>
#include <math32/simd/model.hpp>

#include <bit>

namespace rund::math32::simd {

[[nodiscard]] inline U32x Add(const U32x lhs, const U32x rhs) noexcept { return lhs + rhs; }
[[nodiscard]] inline U32x Sub(const U32x lhs, const U32x rhs) noexcept { return lhs - rhs; }
[[nodiscard]] inline U32x MulLow(const U32x lhs, const U32x rhs) noexcept { return lhs * rhs; }

[[nodiscard]] inline I32x Add(const I32x lhs, const I32x rhs) noexcept {
  return std::bit_cast<I32x>(Add(std::bit_cast<U32x>(lhs), std::bit_cast<U32x>(rhs)));
}

[[nodiscard]] inline I32x Sub(const I32x lhs, const I32x rhs) noexcept {
  return std::bit_cast<I32x>(Sub(std::bit_cast<U32x>(lhs), std::bit_cast<U32x>(rhs)));
}

[[nodiscard]] inline I32x MulLow(const I32x lhs, const I32x rhs) noexcept {
  return std::bit_cast<I32x>(MulLow(std::bit_cast<U32x>(lhs), std::bit_cast<U32x>(rhs)));
}

[[nodiscard]] inline I32x Min(const I32x lhs, const I32x rhs) noexcept { return Select(Lt(lhs, rhs), lhs, rhs); }
[[nodiscard]] inline I32x Max(const I32x lhs, const I32x rhs) noexcept { return Select(Gt(lhs, rhs), lhs, rhs); }
[[nodiscard]] inline U32x Min(const U32x lhs, const U32x rhs) noexcept { return Select(Lt(lhs, rhs), lhs, rhs); }
[[nodiscard]] inline U32x Max(const U32x lhs, const U32x rhs) noexcept { return Select(Gt(lhs, rhs), lhs, rhs); }

}  // namespace rund::math32::simd
