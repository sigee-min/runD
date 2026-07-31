#pragma once

#include <math64/simd/compare.hpp>
#include <math64/simd/select.hpp>
#include <math64/simd/model.hpp>

#include <bit>

namespace rund::math64::simd {

namespace detail {
using U32x4 = u32 __attribute__((vector_size(16)));
using U32x2 = u32 __attribute__((vector_size(8)));

[[nodiscard]] inline U32x2 LowWords(const U64x value) noexcept {
  const U32x4 words = std::bit_cast<U32x4>(value);
  return __builtin_shufflevector(words, words, 0, 2);
}

[[nodiscard]] inline U32x2 HighWords(const U64x value) noexcept {
  const U32x4 words = std::bit_cast<U32x4>(value);
  return __builtin_shufflevector(words, words, 1, 3);
}
}  // namespace detail

[[nodiscard]] inline U64x Add(const U64x lhs, const U64x rhs) noexcept { return lhs + rhs; }
[[nodiscard]] inline U64x Sub(const U64x lhs, const U64x rhs) noexcept { return lhs - rhs; }
[[nodiscard]] inline U64x MulLow(const U64x lhs, const U64x rhs) noexcept {
  const U64x lhs_lo = __builtin_convertvector(detail::LowWords(lhs), U64x);
  const U64x rhs_lo = __builtin_convertvector(detail::LowWords(rhs), U64x);
  const U64x lhs_hi = __builtin_convertvector(detail::HighWords(lhs), U64x);
  const U64x rhs_hi = __builtin_convertvector(detail::HighWords(rhs), U64x);
  const U64x low = lhs_lo * rhs_lo;
  const U64x cross = (lhs_hi * rhs_lo) + (lhs_lo * rhs_hi);
  return low + (cross << 32u);
}

[[nodiscard]] inline I64x Add(const I64x lhs, const I64x rhs) noexcept {
  return std::bit_cast<I64x>(Add(std::bit_cast<U64x>(lhs), std::bit_cast<U64x>(rhs)));
}

[[nodiscard]] inline I64x Sub(const I64x lhs, const I64x rhs) noexcept {
  return std::bit_cast<I64x>(Sub(std::bit_cast<U64x>(lhs), std::bit_cast<U64x>(rhs)));
}

[[nodiscard]] inline I64x MulLow(const I64x lhs, const I64x rhs) noexcept {
  return std::bit_cast<I64x>(MulLow(std::bit_cast<U64x>(lhs), std::bit_cast<U64x>(rhs)));
}

[[nodiscard]] inline I64x Min(const I64x lhs, const I64x rhs) noexcept { return Select(Lt(lhs, rhs), lhs, rhs); }
[[nodiscard]] inline I64x Max(const I64x lhs, const I64x rhs) noexcept { return Select(Gt(lhs, rhs), lhs, rhs); }
[[nodiscard]] inline U64x Min(const U64x lhs, const U64x rhs) noexcept { return Select(Lt(lhs, rhs), lhs, rhs); }
[[nodiscard]] inline U64x Max(const U64x lhs, const U64x rhs) noexcept { return Select(Gt(lhs, rhs), lhs, rhs); }

}  // namespace rund::math64::simd
