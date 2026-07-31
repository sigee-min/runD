#pragma once

#include <math32/simd/arithmetic.hpp>
#include <math32/simd/model.hpp>

namespace rund::math32::simd {

inline i32 FirstLane(const I32x value) noexcept { return value[0]; }
inline u32 FirstLane(const U32x value) noexcept { return value[0]; }

inline i32 ReduceAdd(const I32x value) noexcept {
  const I32x folded2 = Add(value, __builtin_shufflevector(value, value, 2, 3, 0, 1));
  const I32x folded4 = Add(folded2, __builtin_shufflevector(folded2, folded2, 1, 0, 3, 2));
  return FirstLane(folded4);
}

inline u32 ReduceAdd(const U32x value) noexcept {
  const U32x folded2 = Add(value, __builtin_shufflevector(value, value, 2, 3, 0, 1));
  const U32x folded4 = Add(folded2, __builtin_shufflevector(folded2, folded2, 1, 0, 3, 2));
  return FirstLane(folded4);
}

inline i32 ReduceMin(const I32x value) noexcept {
  const I32x folded2 = Min(value, __builtin_shufflevector(value, value, 2, 3, 0, 1));
  const I32x folded4 = Min(folded2, __builtin_shufflevector(folded2, folded2, 1, 0, 3, 2));
  return FirstLane(folded4);
}

inline i32 ReduceMax(const I32x value) noexcept {
  const I32x folded2 = Max(value, __builtin_shufflevector(value, value, 2, 3, 0, 1));
  const I32x folded4 = Max(folded2, __builtin_shufflevector(folded2, folded2, 1, 0, 3, 2));
  return FirstLane(folded4);
}

inline u32 ReduceAnd(const U32x value) noexcept {
  const U32x folded2 = value & __builtin_shufflevector(value, value, 2, 3, 0, 1);
  const U32x folded4 = folded2 & __builtin_shufflevector(folded2, folded2, 1, 0, 3, 2);
  return FirstLane(folded4);
}

inline u32 ReduceOr(const U32x value) noexcept {
  const U32x folded2 = value | __builtin_shufflevector(value, value, 2, 3, 0, 1);
  const U32x folded4 = folded2 | __builtin_shufflevector(folded2, folded2, 1, 0, 3, 2);
  return FirstLane(folded4);
}

}  // namespace rund::math32::simd
