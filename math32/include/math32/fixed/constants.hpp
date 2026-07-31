#pragma once

#include <math32/core/model.hpp>
#include <math32/simd/model.hpp>

#include <limits>

namespace rund::math32 {

inline constexpr i32 FixedMin = std::numeric_limits<i32>::min();
inline constexpr i32 FixedMax = std::numeric_limits<i32>::max();
inline constexpr u64 FixedScale = u64{1} << 31u;
inline constexpr i32 FixedHalf = static_cast<i32>(FixedScale / 2u);
inline constexpr i32 FixedQuarter = static_cast<i32>(FixedScale / 4u);
inline constexpr i32 FixedSqrtHalf = 1518500250;
inline constexpr i32 FixedInvSqrt2 = FixedSqrtHalf;
inline constexpr u32 FixedSqrt2Scaled = 3037000500u;
inline constexpr i32 FixedTanTurnEighth = 889516852;

struct SinCos {
  i32 sin = 0;
  i32 cos = FixedMax;
};

struct SinCosx {
  simd::I32x sin{};
  simd::I32x cos{};
};

}  // namespace rund::math32
