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
}  // namespace

int RunMath32RlContract() {
  using namespace rund::math32;
  const simd::I32x reward{1, 2, 3, 4};
  const simd::I32x gamma{FixedHalf, FixedHalf, FixedQuarter, FixedMax};
  const simd::I32x next{2, 4, 8, 16};
  ExpectI32x(rl::Bellman(reward, gamma, next),
             {detail::ScalarAddSat(1, detail::ScalarMulFixed(FixedHalf, 2)),
              detail::ScalarAddSat(2, detail::ScalarMulFixed(FixedHalf, 4)),
              detail::ScalarAddSat(3, detail::ScalarMulFixed(FixedQuarter, 8)),
              detail::ScalarAddSat(4, detail::ScalarMulFixed(FixedMax, 16))});
  ExpectI32x(rl::TdError(reward, gamma, next, simd::I32x{1, 1, 1, 1}),
             {detail::ScalarSubSat(detail::ScalarAddSat(1, detail::ScalarMulFixed(FixedHalf, 2)), 1),
              detail::ScalarSubSat(detail::ScalarAddSat(2, detail::ScalarMulFixed(FixedHalf, 4)), 1),
              detail::ScalarSubSat(detail::ScalarAddSat(3, detail::ScalarMulFixed(FixedQuarter, 8)), 1),
              detail::ScalarSubSat(detail::ScalarAddSat(4, detail::ScalarMulFixed(FixedMax, 16)), 1)});
  ExpectI32x(rl::QUpdate(simd::I32x{1, 2, 3, 4}, gamma, next),
             {detail::ScalarAddSat(1, detail::ScalarMulFixed(FixedHalf, 2)),
              detail::ScalarAddSat(2, detail::ScalarMulFixed(FixedHalf, 4)),
              detail::ScalarAddSat(3, detail::ScalarMulFixed(FixedQuarter, 8)),
              detail::ScalarAddSat(4, detail::ScalarMulFixed(FixedMax, 16))});
  return 0;
}
