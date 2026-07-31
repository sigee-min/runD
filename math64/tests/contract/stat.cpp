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
}  // namespace

int RunMath64StatContract() {
  using namespace rund::math64;
  std::array<i64, simd::LaneCount> values{FixedHalf, FixedQuarter};
  const auto mean = Mean(soa::I64View(values));
  TEST_ASSERT(mean.ok() && mean.processed == simd::LaneCount);
  ExpectI64x(mean.sum, {FixedHalf, FixedQuarter});
  std::array<i64, simd::LaneCount + 1u> tail_values{1, 2, 8};
  const auto tail_mean = Mean(soa::I64View(tail_values));
  TEST_ASSERT(tail_mean.ok() && tail_mean.processed == tail_values.size());
  ExpectI64x(tail_mean.sum, {9, 2});
  const auto empty_mean = Mean(soa::I64View{});
  TEST_ASSERT(!empty_mean.ok() && empty_mean.empty_input && empty_mean.processed == 0u);

  const auto variance = Variance(soa::I64View(values), simd::I64x{FixedHalf, 0});
  TEST_ASSERT(variance.ok() && variance.processed == simd::LaneCount);
  ExpectU64x(variance.squares, {0u, FixedScale / 16u});

  const auto rms = Rms(soa::I64View(values));
  TEST_ASSERT(rms.ok() && rms.processed == simd::LaneCount);
  ExpectU64x(rms.squares, {FixedScale / 4u, FixedScale / 16u});
  const auto tail_variance = Variance(soa::I64View(tail_values), simd::SplatI64(0));
  TEST_ASSERT(tail_variance.ok() && tail_variance.processed == tail_values.size());
  const auto tail_rms = Rms(soa::I64View(tail_values));
  TEST_ASSERT(tail_rms.ok() && tail_rms.processed == tail_values.size());
  std::array<i64, simd::LaneCount + 1u> variance_tail_values{FixedQuarter, 0, FixedQuarter};
  const auto variance_tail = Variance(soa::I64View(variance_tail_values), simd::SplatI64(FixedQuarter));
  TEST_ASSERT(variance_tail.ok() && variance_tail.processed == variance_tail_values.size());
  ExpectU64x(variance_tail.squares, {0u, FixedScale / 16u});

  ExpectI64x(MulUnit(simd::I64x{FixedMax, FixedHalf}, simd::I64x{FixedMax, FixedHalf}),
             {FixedMax, static_cast<i64>(FixedScale / 4u)});
  ExpectI64x(Lerp(simd::I64x{0, FixedHalf}, simd::I64x{FixedHalf, 0}, simd::I64x{0, FixedMax}), {0, 0});
  ExpectI64x(ClampLerp(simd::I64x{FixedQuarter, FixedHalf},
                       simd::I64x{FixedHalf, 0},
                       simd::I64x{-1, FixedMax}),
             {FixedQuarter - 1, 0});
  ExpectI64x(SmoothStep(simd::I64x{0, FixedHalf}), {0, FixedHalf});
  ExpectI64x(Hermite(simd::I64x{FixedQuarter, FixedHalf},
                     simd::I64x{0, 0},
                     simd::I64x{0, 0},
                     simd::I64x{0, 0},
                     simd::I64x{0, 0}),
             {FixedQuarter - 1, FixedHalf - 1});
  return 0;
}
