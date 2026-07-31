#pragma once

#include <math64/simd/arithmetic.hpp>
#include <math64/simd/model.hpp>

namespace rund::math64::simd {

inline i64 FirstLane(const I64x value) noexcept { return value[0]; }
inline u64 FirstLane(const U64x value) noexcept { return value[0]; }

inline i64 ReduceAdd(const I64x value) noexcept {
  const I64x folded = Add(value, __builtin_shufflevector(value, value, 1, 0));
  return FirstLane(folded);
}

inline u64 ReduceAdd(const U64x value) noexcept {
  const U64x folded = Add(value, __builtin_shufflevector(value, value, 1, 0));
  return FirstLane(folded);
}

inline i64 ReduceMin(const I64x value) noexcept {
  const I64x folded = Min(value, __builtin_shufflevector(value, value, 1, 0));
  return FirstLane(folded);
}

inline i64 ReduceMax(const I64x value) noexcept {
  const I64x folded = Max(value, __builtin_shufflevector(value, value, 1, 0));
  return FirstLane(folded);
}

inline u64 ReduceAnd(const U64x value) noexcept {
  const U64x folded = value & __builtin_shufflevector(value, value, 1, 0);
  return FirstLane(folded);
}

inline u64 ReduceOr(const U64x value) noexcept {
  const U64x folded = value | __builtin_shufflevector(value, value, 1, 0);
  return FirstLane(folded);
}

}  // namespace rund::math64::simd
