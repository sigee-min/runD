#pragma once

#include <math32/simd/model.hpp>

namespace rund::math32::simd {

inline I32x LoadI32(const i32 *const source) noexcept {
  I32x value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}

inline U32x LoadU32(const u32* const source) noexcept {
  U32x value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}

inline void Store(i32* const target, const I32x value) noexcept {
  __builtin_memcpy(target, &value, sizeof(value));
}

inline void Store(u32* const target, const U32x value) noexcept {
  __builtin_memcpy(target, &value, sizeof(value));
}

}  // namespace rund::math32::simd
