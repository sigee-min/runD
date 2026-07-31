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

int RunMath32NonlinearContract() {
  using namespace rund::math32;
  ExpectI32x(Exp2(simd::I32x{0, FixedMin, -FixedHalf, -FixedQuarter}),
             {detail::ScalarExp2(0), detail::ScalarExp2(FixedMin), detail::ScalarExp2(-FixedHalf),
              detail::ScalarExp2(-FixedQuarter)});
  ExpectI32x(Log2(simd::I32x{FixedMax, FixedHalf, FixedQuarter, FixedMax - 1}),
             {detail::ScalarLog2(FixedMax), detail::ScalarLog2(FixedHalf), detail::ScalarLog2(FixedQuarter),
              detail::ScalarLog2(FixedMax - 1)});
  return 0;
}
