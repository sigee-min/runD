#pragma once

#include <math64/simd/model.hpp>

#include <bit>

namespace rund::math64::simd {

[[nodiscard]] inline U64x Select(const Mask64x mask,
                                 const U64x when_true,
                                 const U64x when_false) noexcept {
  return (mask & when_true) | (~mask & when_false);
}

[[nodiscard]] inline I64x Select(const Mask64x mask,
                                 const I64x when_true,
                                 const I64x when_false) noexcept {
  const U64x selected = Select(mask, std::bit_cast<U64x>(when_true), std::bit_cast<U64x>(when_false));
  return std::bit_cast<I64x>(selected);
}

}  // namespace rund::math64::simd
