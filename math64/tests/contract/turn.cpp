#include <math64/math64.hpp>
#include "test/assert.hpp"

#include <array>

namespace {
void ExpectI64x(const rund::math64::simd::I64x value,
                const std::array<rund::math64::i64, rund::math64::simd::LaneCount>& expected) {
  alignas(16) rund::math64::i64 lanes[rund::math64::simd::LaneCount]{};
  rund::math64::simd::Store(lanes, value);
  for (std::size_t index = 0u; index < rund::math64::simd::LaneCount; ++index) {
    TEST_ASSERT(lanes[index] == expected[index]);
  }
}

void ExpectU64x(const rund::math64::simd::U64x value,
                const std::array<rund::math64::u64, rund::math64::simd::LaneCount>& expected) {
  alignas(16) rund::math64::u64 lanes[rund::math64::simd::LaneCount]{};
  rund::math64::simd::Store(lanes, value);
  for (std::size_t index = 0u; index < rund::math64::simd::LaneCount; ++index) {
    TEST_ASSERT(lanes[index] == expected[index]);
  }
}

void ExpectMask64x(const rund::math64::simd::Mask64x value, const bool lane0, const bool lane1) {
  alignas(16) rund::math64::u64 lanes[rund::math64::simd::LaneCount]{};
  rund::math64::simd::Store(lanes, value);
  TEST_ASSERT((lanes[0] == rund::math64::simd::MaskTrueLane) == lane0);
  TEST_ASSERT((lanes[1] == rund::math64::simd::MaskTrueLane) == lane1);
}
}  // namespace

int RunMath64TurnContract() {
  using namespace rund::math64;
  ExpectI64x(TurnSin(simd::U64x{0u, TurnQuarter}), {0, FixedMax});
  ExpectI64x(TurnCos(simd::U64x{TurnHalf, TurnThreeQuarter}), {FixedMin, 0});
  ExpectMask64x(TurnRatioLe(simd::U64x{0u, FixedScale},
                            simd::U64x{FixedScale, FixedScale},
                            simd::I64x{0, FixedHalf}),
                true,
                false);
  ExpectU64x(TurnRatio(simd::U64x{0u, FixedScale / 2u}, simd::U64x{FixedScale, FixedScale}),
             {0u, static_cast<u64>(FixedHalf)});
  ExpectU64x(TurnAtan2(simd::I64x{0, 1}, simd::I64x{1, 0}), {0u, TurnQuarter});
  ExpectU64x(TurnAtan2(simd::I64x{0, -1}, simd::I64x{-1, 0}), {TurnHalf, TurnThreeQuarter});
  return 0;
}
