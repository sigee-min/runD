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
}  // namespace

int RunMath64NonlinearContract() {
  using namespace rund::math64;
  ExpectI64x(Exp2(simd::I64x{0, FixedMin}), {detail::ScalarExp2(0), detail::ScalarExp2(FixedMin)});
  ExpectI64x(Exp2(simd::I64x{-FixedHalf, -FixedQuarter}),
             {detail::ScalarExp2(-FixedHalf), detail::ScalarExp2(-FixedQuarter)});
  ExpectI64x(Log2(simd::I64x{FixedMax, FixedHalf}), {detail::ScalarLog2(FixedMax), detail::ScalarLog2(FixedHalf)});
  ExpectI64x(Log2(simd::I64x{FixedQuarter, FixedMax - 1}),
             {detail::ScalarLog2(FixedQuarter), detail::ScalarLog2(FixedMax - 1)});
  return 0;
}
