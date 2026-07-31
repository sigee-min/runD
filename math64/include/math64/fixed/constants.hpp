#pragma once

#include <math64/core/model.hpp>
#include <math64/simd/model.hpp>

#include <limits>

namespace rund::math64 {

inline constexpr i64 FixedMin = std::numeric_limits<i64>::min();
inline constexpr i64 FixedMax = std::numeric_limits<i64>::max();
inline constexpr u64 FixedScale = u64{1} << 63u;
inline constexpr i64 FixedHalf = static_cast<i64>(FixedScale / 2u);
inline constexpr i64 FixedQuarter = static_cast<i64>(FixedScale / 4u);
inline constexpr i64 FixedSqrtHalf = 6521908912666391106ll;
inline constexpr i64 FixedInvSqrt2 = FixedSqrtHalf;
inline constexpr u64 FixedSqrt2Scaled = 13043817825332782212ull;
inline constexpr i64 FixedTanTurnEighth = 3820445788478006404ll;

struct SinCos {
  i64 sin = 0;
  i64 cos = FixedMax;
};

struct SinCosx {
  simd::I64x sin{};
  simd::I64x cos{};
};

}  // namespace rund::math64
