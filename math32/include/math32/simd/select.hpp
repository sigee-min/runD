#pragma once

#include <math32/simd/model.hpp>

#include <bit>

namespace rund::math32::simd {

[[nodiscard]] inline U32x Select(const Mask32x mask,
                                 const U32x when_true,
                                 const U32x when_false) noexcept {
  return (mask & when_true) | (~mask & when_false);
}

[[nodiscard]] inline I32x Select(const Mask32x mask,
                                 const I32x when_true,
                                 const I32x when_false) noexcept {
  const U32x selected = Select(mask, std::bit_cast<U32x>(when_true), std::bit_cast<U32x>(when_false));
  return std::bit_cast<I32x>(selected);
}

}  // namespace rund::math32::simd
