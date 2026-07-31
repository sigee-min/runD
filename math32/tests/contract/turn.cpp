#include <math32/math32.hpp>
#include "test/assert.hpp"

#include <array>

namespace {
void ExpectI32x(const rund::math32::simd::I32x value,
                const std::array<rund::math32::i32, rund::math32::simd::LaneCount>& expected) {
  alignas(16) rund::math32::i32 lanes[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(lanes, value);
  for (std::size_t index = 0u; index < rund::math32::simd::LaneCount; ++index) TEST_ASSERT(lanes[index] == expected[index]);
}
void ExpectU32x(const rund::math32::simd::U32x value,
                const std::array<rund::math32::u32, rund::math32::simd::LaneCount>& expected) {
  alignas(16) rund::math32::u32 lanes[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(lanes, value);
  for (std::size_t index = 0u; index < rund::math32::simd::LaneCount; ++index) TEST_ASSERT(lanes[index] == expected[index]);
}
}  // namespace

int RunMath32TurnContract() {
  using namespace rund::math32;
  ExpectI32x(TurnSin(simd::U32x{0u, TurnQuarter, TurnHalf, TurnThreeQuarter}),
             {0, FixedMax, 0, FixedMin});
  ExpectI32x(TurnCos(simd::U32x{0u, TurnQuarter, TurnHalf, TurnThreeQuarter}),
             {FixedMax, 0, FixedMin, 0});
  ExpectU32x(TurnAtan2(simd::I32x{0, 1, 0, -1}, simd::I32x{1, 0, -1, 0}),
             {0u, TurnQuarter, TurnHalf, TurnThreeQuarter});
  return 0;
}
