#pragma once

#include <math32/fixed/sqrt.hpp>

namespace rund::math32 {
inline constexpr u32 TurnEighth = u32{1} << 29u;
inline constexpr u32 TurnOctant = TurnEighth;
inline constexpr u32 TurnQuarter = TurnEighth * 2u;
inline constexpr u32 TurnHalf = TurnQuarter * 2u;
inline constexpr u32 TurnThreeQuarter = TurnQuarter * 3u;
namespace detail {
inline constexpr i32 TurnSinCoef[6] = {1686629713, -173399667, 5348082, -78547, 673, -4};
inline constexpr i32 TurnCosCoef[6] = {FixedMax, -662337939, 34046945, -700062, 7711, -53};
inline constexpr i32 TurnAtanCoef[12] = {
    1367130551, -455708559, 273424749, -195303391, 151887082, -124270335,
    105459068, -91860930, 81654160, -73704920, 67352940, -62155800};
}  // namespace detail
}  // namespace rund::math32
