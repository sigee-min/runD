#include <math32/math32.hpp>
#include "test/assert.hpp"

#include <array>

namespace {
void ExpectI32x(const rund::math32::simd::I32x value,
                const std::array<rund::math32::i32, rund::math32::simd::LaneCount>& expected) {
  alignas(16) rund::math32::i32 lanes[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(lanes, value);
  for (std::size_t index = 0u; index < rund::math32::simd::LaneCount; ++index) {
    TEST_ASSERT(lanes[index] == expected[index]);
  }
}
void ExpectU32x(const rund::math32::simd::U32x value,
                const std::array<rund::math32::u32, rund::math32::simd::LaneCount>& expected) {
  alignas(16) rund::math32::u32 lanes[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(lanes, value);
  for (std::size_t index = 0u; index < rund::math32::simd::LaneCount; ++index) {
    TEST_ASSERT(lanes[index] == expected[index]);
  }
}
}  // namespace

int RunMath32StatContract() {
  using namespace rund::math32;
  std::array<i32, simd::LaneCount> values{FixedHalf, FixedHalf, FixedMax, 0};
  const auto mean = Mean(soa::I32View(values));
  TEST_ASSERT(mean.ok() && mean.processed == simd::LaneCount);
  ExpectI32x(mean.sum, {FixedHalf, FixedHalf, FixedMax, 0});
  const auto variance = Variance(soa::I32View(values), simd::I32x{FixedHalf, FixedHalf, FixedHalf, FixedHalf});
  TEST_ASSERT(variance.ok() && variance.processed == simd::LaneCount);
  const auto rms = Rms(soa::I32View(values));
  TEST_ASSERT(rms.ok() && rms.processed == simd::LaneCount);
  std::array<i32, simd::LaneCount + 1u> tail_values{1, 2, 3, 4, 8};
  const auto tail_mean = Mean(soa::I32View(tail_values));
  TEST_ASSERT(tail_mean.ok() && tail_mean.processed == tail_values.size());
  ExpectI32x(tail_mean.sum, {9, 2, 3, 4});
  const auto tail_variance = Variance(soa::I32View(tail_values), simd::SplatI32(0));
  TEST_ASSERT(tail_variance.ok() && tail_variance.processed == tail_values.size());
  const auto tail_rms = Rms(soa::I32View(tail_values));
  TEST_ASSERT(tail_rms.ok() && tail_rms.processed == tail_values.size());
  std::array<i32, simd::LaneCount + 1u> variance_tail_values{FixedQuarter, 0, 0, 0, FixedQuarter};
  const auto variance_tail = Variance(soa::I32View(variance_tail_values), simd::SplatI32(FixedQuarter));
  const u32 quarter_square = static_cast<u32>(FixedScale / 16u);
  TEST_ASSERT(variance_tail.ok() && variance_tail.processed == variance_tail_values.size());
  ExpectU32x(variance_tail.squares, {0u, quarter_square, quarter_square, quarter_square});
  ExpectI32x(MulUnit(simd::I32x{FixedMax, FixedHalf, 0, FixedQuarter},
                     simd::I32x{FixedMax, FixedHalf, FixedMax, FixedHalf}),
             {FixedMax, detail::ScalarClamp((static_cast<detail::i128>(FixedHalf) * FixedHalf) / FixedMax),
              0, detail::ScalarClamp((static_cast<detail::i128>(FixedQuarter) * FixedHalf) / FixedMax)});
  ExpectI32x(Lerp(simd::I32x{0, FixedHalf, FixedMax, 0},
                  simd::I32x{FixedMax, 0, 0, FixedMax},
                  simd::I32x{FixedMax, FixedHalf, 0, FixedQuarter}),
             {detail::ScalarMulFixed(FixedMax, FixedMax),
              detail::ScalarAddSat(detail::ScalarMulFixed(FixedHalf, detail::ScalarSubSat(FixedMax, FixedHalf)),
                                   detail::ScalarMulFixed(0, FixedHalf)),
              detail::ScalarMulFixed(FixedMax, FixedMax),
              detail::ScalarAddSat(detail::ScalarMulFixed(0, detail::ScalarSubSat(FixedMax, FixedQuarter)),
                                   detail::ScalarMulFixed(FixedMax, FixedQuarter))});
  ExpectI32x(SmoothStep(simd::I32x{0, FixedHalf, FixedMax, -1}), {0, FixedHalf, 0, 0});
  ExpectI32x(Hermite(simd::I32x{0, 0, FixedHalf, 0},
                     simd::I32x{0, 0, 0, FixedHalf},
                     simd::I32x{FixedMax, FixedMax, 0, FixedMax},
                     simd::I32x{0, 0, 0, 0},
                     simd::I32x{0, FixedMax, 0, 0}),
             {0, 0, FixedHalf - 1, 0});
  return 0;
}
