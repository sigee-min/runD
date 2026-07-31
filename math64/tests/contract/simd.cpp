#include <math64/math64.hpp>
#include "test/assert.hpp"

#include <cstddef>
#include <limits>
#include <type_traits>

namespace {

void ExpectI64x(const rund::math64::simd::I64x value,
                const rund::math64::i64 a,
                const rund::math64::i64 b) {
  alignas(16) rund::math64::i64 lanes[rund::math64::simd::LaneCount]{};
  rund::math64::simd::Store(lanes, value);
  TEST_ASSERT(lanes[0] == a);
  TEST_ASSERT(lanes[1] == b);
}

void ExpectU64x(const rund::math64::simd::U64x value,
                const rund::math64::u64 a,
                const rund::math64::u64 b) {
  alignas(16) rund::math64::u64 lanes[rund::math64::simd::LaneCount]{};
  rund::math64::simd::Store(lanes, value);
  TEST_ASSERT(lanes[0] == a);
  TEST_ASSERT(lanes[1] == b);
}

void ExpectMask(const rund::math64::simd::Mask64x value, const bool a, const bool b) {
  alignas(16) rund::math64::u64 lanes[rund::math64::simd::LaneCount]{};
  rund::math64::simd::Store(lanes, value);
  TEST_ASSERT((lanes[0] == rund::math64::simd::MaskTrueLane) == a);
  TEST_ASSERT((lanes[1] == rund::math64::simd::MaskTrueLane) == b);
}

}  // namespace

int RunMath64SimdContract() {
  using namespace rund::math64;
  using namespace rund::math64::simd;

  static_assert(LaneCount == 2u);
  static_assert(sizeof(I64x) == 16u);
  static_assert(sizeof(U64x) == 16u);
  static_assert(sizeof(Mask64x) == 16u);
  static_assert(std::is_same_v<Mask64x, U64x>);
  static_assert(!std::is_same_v<I64x, U64x>);
  static_assert(alignof(I64x) >= alignof(i64));
  static_assert(alignof(U64x) >= alignof(u64));

  alignas(16) i64 signed_lanes[LaneCount]{1, -2};
  alignas(16) u64 unsigned_lanes[LaneCount]{0u, 0xffffffffffffffffull};
  const I64x signed_values = LoadI64(signed_lanes);
  const U64x unsigned_values = LoadU64(unsigned_lanes);
  ExpectI64x(signed_values, 1, -2);

  alignas(16) u64 stored_unsigned[LaneCount]{};
  Store(stored_unsigned, unsigned_values);
  TEST_ASSERT(stored_unsigned[0] == 0u);
  TEST_ASSERT(stored_unsigned[1] == 0xffffffffffffffffull);

  i64 scalar_aligned_signed[3]{111, -7, 12};
  u64 scalar_aligned_unsigned[3]{111u, 0x123456789abcdef0ull, 0xffff0000ffff0000ull};
  ExpectI64x(LoadI64(scalar_aligned_signed + 1), -7, 12);
  ExpectU64x(LoadU64(scalar_aligned_unsigned + 1), 0x123456789abcdef0ull, 0xffff0000ffff0000ull);
  Store(scalar_aligned_signed + 1, I64x{-9, 13});
  Store(scalar_aligned_unsigned + 1, U64x{0x10u, 0x20u});
  TEST_ASSERT(scalar_aligned_signed[0] == 111);
  TEST_ASSERT(scalar_aligned_signed[1] == -9);
  TEST_ASSERT(scalar_aligned_signed[2] == 13);
  TEST_ASSERT(scalar_aligned_unsigned[0] == 111u);
  TEST_ASSERT(scalar_aligned_unsigned[1] == 0x10u);
  TEST_ASSERT(scalar_aligned_unsigned[2] == 0x20u);

  ExpectMask(MaskFromBools(true, false), true, false);
  ExpectMask(Not(MaskFromBools(true, false)), false, true);
  TEST_ASSERT(All(MaskTrue()));
  TEST_ASSERT(!All(MaskFromBools(true, false)));
  TEST_ASSERT(Any(MaskFromBools(false, true)));
  TEST_ASSERT(!Any(MaskFalse()));

  alignas(16) const i64 select_true[LaneCount]{10, 20};
  alignas(16) const i64 select_false[LaneCount]{-10, -20};
  const I64x selected = simd::Select(MaskFromBools(true, false), LoadI64(select_true), LoadI64(select_false));
  ExpectI64x(selected, 10, -20);

  alignas(16) const i64 lhs_lanes[LaneCount]{7, -8};
  alignas(16) const i64 rhs_lanes[LaneCount]{3, -8};
  const I64x lhs = LoadI64(lhs_lanes);
  const I64x rhs = LoadI64(rhs_lanes);
  ExpectI64x(Add(lhs, rhs), 10, -16);
  ExpectI64x(Sub(lhs, rhs), 4, 0);
  ExpectI64x(simd::MulLow(lhs, rhs), 21, 64);
  ExpectI64x(simd::Min(lhs, rhs), 3, -8);
  ExpectI64x(simd::Max(lhs, rhs), 7, -8);

  ExpectMask(Eq(lhs, rhs), false, true);
  ExpectMask(Ne(lhs, rhs), true, false);
  ExpectMask(Lt(lhs, rhs), false, false);
  ExpectMask(Le(lhs, rhs), false, true);
  ExpectMask(Gt(lhs, rhs), true, false);
  ExpectMask(Ge(lhs, rhs), true, true);

  alignas(16) const u64 lhs_unsigned_lanes[LaneCount]{0u, 0xfffffffffffffffeull};
  alignas(16) const u64 rhs_unsigned_lanes[LaneCount]{1u, 4u};
  const U64x lhs_unsigned = LoadU64(lhs_unsigned_lanes);
  const U64x rhs_unsigned = LoadU64(rhs_unsigned_lanes);
  ExpectU64x(Add(lhs_unsigned, rhs_unsigned), 1u, 2u);
  ExpectU64x(Sub(lhs_unsigned, rhs_unsigned), 0xffffffffffffffffull, 0xfffffffffffffffaull);
  ExpectU64x(simd::MulLow(lhs_unsigned, rhs_unsigned), 0u, 0xfffffffffffffff8ull);
  ExpectU64x(simd::Min(lhs_unsigned, rhs_unsigned), 0u, 4u);
  ExpectU64x(simd::Max(lhs_unsigned, rhs_unsigned), 1u, 0xfffffffffffffffeull);
  ExpectMask(Eq(lhs_unsigned, rhs_unsigned), false, false);
  ExpectMask(Ne(lhs_unsigned, rhs_unsigned), true, true);
  ExpectMask(Lt(lhs_unsigned, rhs_unsigned), true, false);
  ExpectMask(Le(lhs_unsigned, rhs_unsigned), true, false);
  ExpectMask(Gt(lhs_unsigned, rhs_unsigned), false, true);
  ExpectMask(Ge(lhs_unsigned, rhs_unsigned), false, true);

  const I64x wrap_lhs = I64x{std::numeric_limits<i64>::max(), std::numeric_limits<i64>::min()};
  const I64x wrap_rhs = I64x{1, 1};
  ExpectI64x(Add(wrap_lhs, wrap_rhs), std::numeric_limits<i64>::min(), std::numeric_limits<i64>::min() + 1);
  ExpectI64x(Sub(I64x{std::numeric_limits<i64>::min(), 0}, I64x{1, 1}),
             std::numeric_limits<i64>::max(),
             -1);
  ExpectI64x(simd::MulLow(I64x{std::numeric_limits<i64>::min(), 0x4000000000000000ll}, I64x{2, 4}), 0, 0);
  TEST_ASSERT(ReduceAdd(I64x{std::numeric_limits<i64>::max(), 1}) == std::numeric_limits<i64>::min());

  TEST_ASSERT(ReduceAdd(lhs) == -1);
  TEST_ASSERT(ReduceAdd(lhs_unsigned) == 0xfffffffffffffffeull);
  TEST_ASSERT(ReduceMin(lhs) == -8);
  TEST_ASSERT(ReduceMax(lhs) == 7);
  alignas(16) const u64 and_lanes[LaneCount]{0xfffffffffffffff0ull, 0xffffffffffff0fffull};
  alignas(16) const u64 or_lanes[LaneCount]{0x1u, 0x200u};
  TEST_ASSERT(ReduceAnd(LoadU64(and_lanes)) == 0xffffffffffff0ff0ull);
  TEST_ASSERT(ReduceOr(LoadU64(or_lanes)) == 0x201u);

  return 0;
}
