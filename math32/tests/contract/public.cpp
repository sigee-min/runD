#include <math32/math32.hpp>
#include "test/assert.hpp"

#include <type_traits>

int RunMath32PublicContract() {
  static_assert(sizeof(rund::math32::i32) == 4u);
  static_assert(rund::math32::FixedScale == (rund::math32::u64{1} << 31u));
  static_assert(!std::is_same_v<rund::math32::i64, rund::math32::i32>);
  static_assert(rund::math32::simd::LaneCount == 4u);
  TEST_ASSERT(rund::math32::simd::Any(rund::math32::simd::MaskTrue()));
  return 0;
}
