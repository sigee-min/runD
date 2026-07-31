#include <math32/math32.hpp>
#include "test/assert.hpp"

#include <array>
#include <limits>

namespace {
void ExpectI32x(const rund::math32::simd::I32x value,
                const std::array<rund::math32::i32, rund::math32::simd::LaneCount>& expected) {
  alignas(16) rund::math32::i32 lanes[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(lanes, value);
  for (std::size_t index = 0u; index < rund::math32::simd::LaneCount; ++index) {
    TEST_ASSERT(lanes[index] == expected[index]);
  }
}

void ExpectU32x(const rund::math32::simd::U32x value,
                const std::array<rund::math32::u32, rund::math32::simd::LaneCount>& expected) {
  alignas(16) rund::math32::u32 lanes[rund::math32::simd::LaneCount]{};
  rund::math32::simd::Store(lanes, value);
  for (std::size_t index = 0u; index < rund::math32::simd::LaneCount; ++index) {
    TEST_ASSERT(lanes[index] == expected[index]);
  }
}
}  // namespace

int RunMath32FixedContract() {
  using namespace rund::math32;
  const simd::I32x lhs{std::numeric_limits<i32>::max(), FixedMin, FixedHalf, -FixedQuarter};
  const simd::I32x rhs{1, -1, FixedHalf, FixedQuarter};
  ExpectI32x(AddWrap(lhs, rhs),
             {detail::ScalarAddWrap(std::numeric_limits<i32>::max(), 1),
              detail::ScalarAddWrap(FixedMin, -1),
              detail::ScalarAddWrap(FixedHalf, FixedHalf),
              detail::ScalarAddWrap(-FixedQuarter, FixedQuarter)});
  ExpectU32x(AddWrapUnsigned(simd::U32x{std::numeric_limits<u32>::max(), 1u, 100u, 0u},
                             simd::U32x{1u, 2u, std::numeric_limits<u32>::max(), 0u}),
             {0u, 3u, 99u, 0u});
  ExpectI32x(AddSat(lhs, rhs), {FixedMax, FixedMin, FixedMax, 0});
  ExpectU32x(AddSatUnsigned(simd::U32x{std::numeric_limits<u32>::max(), 1u, 100u, 0u},
                            simd::U32x{1u, 2u, std::numeric_limits<u32>::max(), 0u}),
             {std::numeric_limits<u32>::max(), 3u, std::numeric_limits<u32>::max(), 0u});
  ExpectI32x(SubSat(simd::I32x{FixedMin, FixedMax, 5, -5}, simd::I32x{1, -1, 7, -9}),
             {FixedMin, FixedMax, -2, 4});
  ExpectI32x(SubWrap(simd::I32x{FixedMin, 0, 5, -5}, simd::I32x{1, 1, 7, -9}),
             {FixedMax, -1, -2, 4});
  ExpectI32x(MulLow(simd::I32x{FixedMax, FixedMin, 12345, -7}, simd::I32x{2, 2, -3, -9}),
             {detail::ScalarMulLow(FixedMax, 2),
              detail::ScalarMulLow(FixedMin, 2),
              detail::ScalarMulLow(12345, -3),
              detail::ScalarMulLow(-7, -9)});
  ExpectI32x(MulHigh(simd::I32x{FixedMax, FixedMin, 12345, -7}, simd::I32x{2, 2, -3, -9}),
             {detail::ScalarMulHigh(FixedMax, 2),
              detail::ScalarMulHigh(FixedMin, 2),
              detail::ScalarMulHigh(12345, -3),
              detail::ScalarMulHigh(-7, -9)});
  ExpectI32x(Min(simd::I32x{3, -2, 7, -8}, simd::I32x{2, -3, 8, -7}), {2, -3, 7, -8});
  ExpectI32x(Max(simd::I32x{3, -2, 7, -8}, simd::I32x{2, -3, 8, -7}), {3, -2, 8, -7});
  ExpectI32x(Clamp(simd::I32x{-10, -2, 4, 10}, simd::I32x{-5, -5, -5, -5}, simd::I32x{5, 5, 5, 5}),
             {-5, -2, 4, 5});
  ExpectI32x(Select(simd::Mask32x{simd::MaskTrueLane, simd::MaskFalseLane, simd::MaskTrueLane, simd::MaskFalseLane},
                    simd::I32x{1, 2, 3, 4},
                    simd::I32x{10, 20, 30, 40}),
             {1, 20, 3, 40});
  ExpectI32x(NegPositiveFixed(simd::I32x{FixedMax, FixedHalf, -FixedQuarter, 0}), {FixedMin, -FixedHalf, FixedQuarter, 0});
  ExpectU32x(AbsMagnitude(simd::I32x{FixedMin, -7, 0, FixedMax}), {0x80000000u, 7u, 0u, static_cast<u32>(FixedMax)});
  ExpectI32x(Abs(simd::I32x{FixedMin, -7, 0, FixedMax}), {FixedMax, 7, 0, FixedMax});
  ExpectI32x(Sign(simd::I32x{FixedMin, -7, 0, FixedMax}), {-1, -1, 0, 1});
  ExpectI32x(MulFixed(simd::I32x{FixedHalf, FixedQuarter, -FixedHalf, -1},
                      simd::I32x{FixedHalf, -FixedHalf, FixedHalf, FixedHalf}),
             {detail::ScalarMulFixed(FixedHalf, FixedHalf),
              detail::ScalarMulFixed(FixedQuarter, -FixedHalf),
              detail::ScalarMulFixed(-FixedHalf, FixedHalf),
              detail::ScalarMulFixed(-1, FixedHalf)});
  ExpectI32x(MulFixedScaled(simd::I32x{FixedHalf, -FixedHalf, FixedQuarter, -1},
                            simd::U32x{static_cast<u32>(FixedHalf),
                                       static_cast<u32>(FixedHalf),
                                       static_cast<u32>(FixedMax),
                                       static_cast<u32>(FixedHalf)}),
             {detail::ScalarMulFixedScaled(FixedHalf, static_cast<u32>(FixedHalf)),
              detail::ScalarMulFixedScaled(-FixedHalf, static_cast<u32>(FixedHalf)),
              detail::ScalarMulFixedScaled(FixedQuarter, static_cast<u32>(FixedMax)),
              detail::ScalarMulFixedScaled(-1, static_cast<u32>(FixedHalf))});
  ExpectU32x(MulUnsignedFixed(simd::U32x{static_cast<u32>(FixedHalf), static_cast<u32>(FixedQuarter), 0u, 7u},
                              simd::U32x{static_cast<u32>(FixedHalf), static_cast<u32>(FixedMax), 9u, 11u}),
             {static_cast<u32>(detail::ScalarMulUnsignedFixed(static_cast<u32>(FixedHalf), static_cast<u32>(FixedHalf))),
              static_cast<u32>(detail::ScalarMulUnsignedFixed(static_cast<u32>(FixedQuarter), static_cast<u32>(FixedMax))),
              0u,
              static_cast<u32>(detail::ScalarMulUnsignedFixed(7u, 11u))});
  ExpectI32x(MulAddFixed(simd::I32x{FixedHalf, FixedQuarter, -FixedHalf, -1},
                         simd::I32x{FixedHalf, -FixedHalf, FixedHalf, FixedHalf},
                         simd::I32x{1, -1, 7, -7}),
             {detail::ScalarMulAddFixed(FixedHalf, FixedHalf, 1),
              detail::ScalarMulAddFixed(FixedQuarter, -FixedHalf, -1),
              detail::ScalarMulAddFixed(-FixedHalf, FixedHalf, 7),
              detail::ScalarMulAddFixed(-1, FixedHalf, -7)});
  ExpectI32x(DivFixed(simd::I32x{FixedHalf, FixedQuarter, 1, -1}, simd::I32x{FixedHalf, FixedHalf, 0, 0}),
             {detail::ScalarDivFixed(FixedHalf, FixedHalf), detail::ScalarDivFixed(FixedQuarter, FixedHalf), FixedMax, FixedMin});
  ExpectI32x(Recip(simd::I32x{FixedMax, FixedHalf, -FixedHalf, 0}),
             {detail::ScalarRecip(FixedMax), detail::ScalarRecip(FixedHalf), detail::ScalarRecip(-FixedHalf), FixedMax});
  ExpectI32x(Sqrt(simd::I32x{FixedQuarter, FixedMax, 0, -1}),
             {detail::ScalarSqrt(FixedQuarter), detail::ScalarSqrt(FixedMax), 0, 0});
  ExpectI32x(Rsqrt(simd::I32x{FixedQuarter, FixedMax, 0, -1}),
             {detail::ScalarRsqrt(FixedQuarter), detail::ScalarRsqrt(FixedMax), FixedMax, FixedMax});
  return 0;
}
