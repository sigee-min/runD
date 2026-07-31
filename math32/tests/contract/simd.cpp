#include <math32/math32.hpp>
#include "test/assert.hpp"

#include <cstddef>
#include <type_traits>

namespace {

void ExpectI32x(const rund::math32::simd::I32x value,
                const rund::math32::i32 a,
                const rund::math32::i32 b,
                const rund::math32::i32 c,
                const rund::math32::i32 d) {
  alignas(16) rund::math32::i32 lanes[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(lanes, value);
  TEST_ASSERT(lanes[0] == a);
  TEST_ASSERT(lanes[1] == b);
  TEST_ASSERT(lanes[2] == c);
  TEST_ASSERT(lanes[3] == d);
}

void ExpectMask(const rund::math32::simd::Mask32x value,
                const bool a,
                const bool b,
                const bool c,
                const bool d) {
  alignas(16) rund::math32::u32 lanes[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(lanes, value);
  TEST_ASSERT((lanes[0] == rund::math32::simd::MaskTrueLane) == a);
  TEST_ASSERT((lanes[1] == rund::math32::simd::MaskTrueLane) == b);
  TEST_ASSERT((lanes[2] == rund::math32::simd::MaskTrueLane) == c);
  TEST_ASSERT((lanes[3] == rund::math32::simd::MaskTrueLane) == d);
}

void ExpectU32x(const rund::math32::simd::U32x value,
                const rund::math32::u32 a,
                const rund::math32::u32 b,
                const rund::math32::u32 c,
                const rund::math32::u32 d) {
  alignas(16) rund::math32::u32 lanes[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(lanes, value);
  TEST_ASSERT(lanes[0] == a);
  TEST_ASSERT(lanes[1] == b);
  TEST_ASSERT(lanes[2] == c);
  TEST_ASSERT(lanes[3] == d);
}

}  // namespace

int RunMath32SimdContract() {
  using namespace rund::math32;
  using namespace rund::math32::simd;

  static_assert(LaneCount == 4u);
  static_assert(sizeof(I32x) == 16u);
  static_assert(sizeof(U32x) == 16u);
  static_assert(sizeof(Mask32x) == 16u);
  static_assert(std::is_same_v<Mask32x, U32x>);
  static_assert(alignof(I32x) >= alignof(i32));
  static_assert(alignof(U32x) >= alignof(u32));

  alignas(16) i32 signed_lanes[LaneCount]{1, -2, 3, -4};
  alignas(16) u32 unsigned_lanes[LaneCount]{0u, 1u, 0x80000000u, 0xffffffffu};
  const I32x signed_values = LoadI32(signed_lanes);
  const U32x unsigned_values = LoadU32(unsigned_lanes);
  ExpectI32x(signed_values, 1, -2, 3, -4);

  alignas(16) u32 stored_unsigned[LaneCount]{};
  Store(stored_unsigned, unsigned_values);
  TEST_ASSERT(stored_unsigned[0] == 0u);
  TEST_ASSERT(stored_unsigned[1] == 1u);
  TEST_ASSERT(stored_unsigned[2] == 0x80000000u);
  TEST_ASSERT(stored_unsigned[3] == 0xffffffffu);

  ExpectMask(MaskFromBools(true, false, true, false), true, false, true, false);
  ExpectMask(Not(MaskFromBools(true, false, true, false)), false, true, false, true);
  TEST_ASSERT(All(MaskTrue()));
  TEST_ASSERT(!All(MaskFromBools(true, true, false, true)));
  TEST_ASSERT(Any(MaskFromBools(false, false, true, false)));
  TEST_ASSERT(!Any(MaskFalse()));

  alignas(16) const i32 select_true[LaneCount]{10, 20, 30, 40};
  alignas(16) const i32 select_false[LaneCount]{-10, -20, -30, -40};
  const I32x selected = simd::Select(MaskFromBools(true, false, true, false),
                                    LoadI32(select_true),
                                    LoadI32(select_false));
  ExpectI32x(selected, 10, -20, 30, -40);

  alignas(16) const i32 lhs_lanes[LaneCount]{7, -8, 9, -10};
  alignas(16) const i32 rhs_lanes[LaneCount]{3, -8, -9, 11};
  const I32x lhs = LoadI32(lhs_lanes);
  const I32x rhs = LoadI32(rhs_lanes);
  ExpectI32x(Add(lhs, rhs), 10, -16, 0, 1);
  ExpectI32x(Sub(lhs, rhs), 4, 0, 18, -21);
  ExpectI32x(simd::MulLow(lhs, rhs), 21, 64, -81, -110);
  ExpectI32x(simd::Min(lhs, rhs), 3, -8, -9, -10);
  ExpectI32x(simd::Max(lhs, rhs), 7, -8, 9, 11);

  ExpectMask(Eq(lhs, rhs), false, true, false, false);
  ExpectMask(Ne(lhs, rhs), true, false, true, true);
  ExpectMask(Lt(lhs, rhs), false, false, false, true);
  ExpectMask(Le(lhs, rhs), false, true, false, true);
  ExpectMask(Gt(lhs, rhs), true, false, true, false);
  ExpectMask(Ge(lhs, rhs), true, true, true, false);

  alignas(16) const u32 lhs_unsigned_lanes[LaneCount]{0u, 1u, 0xfffffffeu, 0x80000000u};
  alignas(16) const u32 rhs_unsigned_lanes[LaneCount]{1u, 1u, 4u, 0x7fffffffu};
  const U32x lhs_unsigned = LoadU32(lhs_unsigned_lanes);
  const U32x rhs_unsigned = LoadU32(rhs_unsigned_lanes);
  ExpectU32x(Add(lhs_unsigned, rhs_unsigned), 1u, 2u, 2u, 0xffffffffu);
  ExpectU32x(Sub(lhs_unsigned, rhs_unsigned), 0xffffffffu, 0u, 0xfffffffau, 1u);
  ExpectU32x(simd::MulLow(lhs_unsigned, rhs_unsigned), 0u, 1u, 0xfffffff8u, 0x80000000u);
  ExpectU32x(simd::Min(lhs_unsigned, rhs_unsigned), 0u, 1u, 4u, 0x7fffffffu);
  ExpectU32x(simd::Max(lhs_unsigned, rhs_unsigned), 1u, 1u, 0xfffffffeu, 0x80000000u);
  ExpectMask(Eq(lhs_unsigned, rhs_unsigned), false, true, false, false);
  ExpectMask(Ne(lhs_unsigned, rhs_unsigned), true, false, true, true);
  ExpectMask(Lt(lhs_unsigned, rhs_unsigned), true, false, false, false);
  ExpectMask(Le(lhs_unsigned, rhs_unsigned), true, true, false, false);
  ExpectMask(Gt(lhs_unsigned, rhs_unsigned), false, false, true, true);
  ExpectMask(Ge(lhs_unsigned, rhs_unsigned), false, true, true, true);

  TEST_ASSERT(ReduceAdd(lhs) == -2);
  TEST_ASSERT(ReduceAdd(lhs_unsigned) == 0x7fffffffu);
  TEST_ASSERT(ReduceMin(lhs) == -10);
  TEST_ASSERT(ReduceMax(lhs) == 9);
  alignas(16) const u32 and_lanes[LaneCount]{0xfffffff0u, 0xffffff0fu, 0xfffff0ffu, 0xffff0fffu};
  alignas(16) const u32 or_lanes[LaneCount]{0x1u, 0x20u, 0x300u, 0x4000u};
  TEST_ASSERT(ReduceAnd(LoadU32(and_lanes)) == 0xffff0000u);
  TEST_ASSERT(ReduceOr(LoadU32(or_lanes)) == 0x4321u);

  return 0;
}
