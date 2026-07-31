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

int RunMath64QuantContract() {
  using namespace rund::math64;
  ExpectI64x(quant::ClampI8(simd::I64x{1000, -200}), {127, -128});
  ExpectI64x(quant::ClampI16(simd::I64x{40000, -40000}), {32767, -32768});
  ExpectI64x(quant::ClampU8(simd::I64x{-1, 1000}), {0, 255});
  const auto r = quant::RequantI64ToI8(simd::I64x{100, 10000},
                                       simd::I64x{2, 100},
                                       simd::U64x{1u, 0u},
                                       simd::I64x{0, 0});
  ExpectI64x(r.value, {100, 127});
  TEST_ASSERT(r.ok());
  const auto invalid_shift = quant::RequantI64ToU8(simd::I64x{1, -1},
                                                   simd::I64x{1, 1},
                                                   simd::U64x{127u, 0u},
                                                   simd::I64x{0, 0});
  ExpectI64x(invalid_shift.value, {0, 0});
  TEST_ASSERT(!invalid_shift.ok());
  return 0;
}
