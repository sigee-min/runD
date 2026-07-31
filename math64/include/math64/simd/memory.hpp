#pragma once

#include <math64/simd/model.hpp>

namespace rund::math64::simd {

inline I64x LoadI64(const i64 *const source) noexcept {
  I64x value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}

inline U64x LoadU64(const u64* const source) noexcept {
  U64x value{};
  __builtin_memcpy(&value, source, sizeof(value));
  return value;
}

inline void Store(i64* const target, const I64x value) noexcept {
  __builtin_memcpy(target, &value, sizeof(value));
}

inline void Store(u64* const target, const U64x value) noexcept {
  __builtin_memcpy(target, &value, sizeof(value));
}

}  // namespace rund::math64::simd
