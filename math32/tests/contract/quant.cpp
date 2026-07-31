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

int RunMath32QuantContract() {
  using namespace rund::math32;
  ExpectI32x(quant::ClampI8(simd::I32x{1000, -200, 7, 0}), {127, -128, 7, 0});
  ExpectI32x(quant::ClampI16(simd::I32x{40000, -40000, 7, 0}), {32767, -32768, 7, 0});
  ExpectI32x(quant::ClampU8(simd::I32x{-1, 0, 255, 1000}), {0, 0, 255, 255});
  const auto r = quant::RequantI32ToI8(simd::I32x{100, -100, 10000, 1},
                                       simd::I32x{2, 2, 100, 1},
                                       simd::U32x{1u, 1u, 0u, 63u},
                                       simd::I32x{0, 0, 0, 0});
  ExpectI32x(r.value, {100, -100, 127, 0});
  TEST_ASSERT(!r.ok());
  return 0;
}
