#include <math64/math64.hpp>
#include "test/assert.hpp"

int RunMath64PublicContract() {
  static_assert(sizeof(rund::math64::i64) == 8u);
  static_assert(rund::math64::FixedScale == (rund::math64::u64{1} << 63u));
  static_assert(rund::math64::simd::LaneCount == 2u);
  TEST_ASSERT(rund::math64::simd::Any(rund::math64::simd::MaskTrue()));
  return 0;
}
