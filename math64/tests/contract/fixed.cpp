#include <math64/math64.hpp>
#include "test/assert.hpp"

#include <array>
#include <limits>

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

int RunMath64FixedContract() {
  using namespace rund::math64;
  const simd::I64x lhs{std::numeric_limits<i64>::max(), FixedMin};
  const simd::I64x rhs{1, -1};
  ExpectI64x(AddWrap(lhs, rhs), {detail::ScalarAddWrap(std::numeric_limits<i64>::max(), 1), detail::ScalarAddWrap(FixedMin, -1)});
  ExpectU64x(AddWrapUnsigned(simd::U64x{std::numeric_limits<u64>::max(), 100u}, simd::U64x{1u, std::numeric_limits<u64>::max()}),
             {0u, 99u});
  ExpectI64x(AddSat(lhs, rhs), {FixedMax, FixedMin});
  ExpectU64x(AddSatUnsigned(simd::U64x{std::numeric_limits<u64>::max(), 100u}, simd::U64x{1u, std::numeric_limits<u64>::max()}),
             {std::numeric_limits<u64>::max(), std::numeric_limits<u64>::max()});
  ExpectI64x(SubSat(simd::I64x{FixedMin, FixedMax}, simd::I64x{1, -1}), {FixedMin, FixedMax});
  ExpectI64x(SubWrap(simd::I64x{FixedMin, 0}, simd::I64x{1, 1}), {FixedMax, -1});
  ExpectI64x(MulLow(simd::I64x{FixedMax, FixedMin}, simd::I64x{2, 2}),
             {detail::ScalarMulLow(FixedMax, 2), detail::ScalarMulLow(FixedMin, 2)});
  ExpectI64x(MulHigh(simd::I64x{FixedMax, FixedMin}, simd::I64x{2, 2}),
             {detail::ScalarMulHigh(FixedMax, 2), detail::ScalarMulHigh(FixedMin, 2)});
  ExpectI64x(Min(simd::I64x{3, -2}, simd::I64x{2, -3}), {2, -3});
  ExpectI64x(Max(simd::I64x{3, -2}, simd::I64x{2, -3}), {3, -2});
  ExpectI64x(Clamp(simd::I64x{-10, 10}, simd::I64x{-5, -5}, simd::I64x{5, 5}), {-5, 5});
  ExpectI64x(Select(simd::Mask64x{simd::MaskTrueLane, simd::MaskFalseLane}, simd::I64x{1, 2}, simd::I64x{10, 20}), {1, 20});
  ExpectI64x(NegPositiveFixed(simd::I64x{FixedMax, -FixedQuarter}), {FixedMin, FixedQuarter});
  ExpectU64x(AbsMagnitude(simd::I64x{FixedMin, -7}), {0x8000000000000000ull, 7u});
  ExpectI64x(Abs(simd::I64x{FixedMin, -7}), {FixedMax, 7});
  ExpectI64x(Sign(simd::I64x{-7, 0}), {-1, 0});
  ExpectI64x(MulFixed(simd::I64x{FixedHalf, -1}, simd::I64x{FixedHalf, FixedHalf}),
             {detail::ScalarMulFixed(FixedHalf, FixedHalf), detail::ScalarMulFixed(-1, FixedHalf)});
  ExpectI64x(MulFixedScaled(simd::I64x{FixedHalf, -1}, simd::U64x{static_cast<u64>(FixedHalf), static_cast<u64>(FixedHalf)}),
             {detail::ScalarMulFixedScaled(FixedHalf, static_cast<u64>(FixedHalf)),
              detail::ScalarMulFixedScaled(-1, static_cast<u64>(FixedHalf))});
  ExpectU64x(MulUnsignedFixed(simd::U64x{static_cast<u64>(FixedHalf), 7u}, simd::U64x{static_cast<u64>(FixedHalf), 11u}),
             {static_cast<u64>(detail::ScalarMulUnsignedFixed(static_cast<u64>(FixedHalf), static_cast<u64>(FixedHalf))),
              static_cast<u64>(detail::ScalarMulUnsignedFixed(7u, 11u))});
  ExpectI64x(MulAddFixed(simd::I64x{FixedHalf, -1}, simd::I64x{FixedHalf, FixedHalf}, simd::I64x{1, 7}),
             {detail::ScalarMulAddFixed(FixedHalf, FixedHalf, 1), detail::ScalarMulAddFixed(-1, FixedHalf, 7)});
  ExpectI64x(DivFixed(simd::I64x{FixedHalf, -1}, simd::I64x{FixedHalf, 0}),
             {detail::ScalarDivFixed(FixedHalf, FixedHalf), FixedMin});
  ExpectI64x(Recip(simd::I64x{FixedMax, 0}), {detail::ScalarRecip(FixedMax), FixedMax});
  ExpectI64x(Sqrt(simd::I64x{FixedQuarter, -1}), {detail::ScalarSqrt(FixedQuarter), 0});
  ExpectI64x(Rsqrt(simd::I64x{FixedQuarter, 0}), {detail::ScalarRsqrt(FixedQuarter), FixedMax});
  return 0;
}
