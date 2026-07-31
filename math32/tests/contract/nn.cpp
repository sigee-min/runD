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
void StoreI32x(const rund::math32::simd::I32x value,
               std::array<rund::math32::i32, rund::math32::simd::LaneCount>& out) {
  rund::math32::simd::Store(out.data(), value);
}
}  // namespace

int RunMath32NnContract() {
  using namespace rund::math32;
  ExpectI32x(nn::Relu(simd::I32x{-1, 0, 7, FixedMin}), {0, 0, 7, 0});
  ExpectI32x(nn::SigmoidApprox(simd::I32x{0, 0, 0, 0}), {FixedHalf, FixedHalf, FixedHalf, FixedHalf});

  std::array<i8, simd::LaneCount> a{2, -3, 0, 6};
  std::array<i8, simd::LaneCount> b{4, 5, 7, 1};
  const auto dot = nn::DotI8I8To32(soa::I8View(a), soa::I8View(b));
  TEST_ASSERT(dot.ok() && dot.processed == simd::LaneCount);
  ExpectI32x(dot.sum, {8, -15, 0, 6});
  std::array<i8, simd::LaneCount + 1u> tail_a{2, -3, 0, 6, 5};
  std::array<i8, simd::LaneCount + 1u> tail_b{4, 5, 7, 1, 9};
  const auto tail_dot = nn::DotI8I8To32(soa::I8View(tail_a), soa::I8View(tail_b));
  TEST_ASSERT(tail_dot.ok() && tail_dot.processed == tail_a.size());
  ExpectI32x(tail_dot.sum, {53, -15, 0, 6});

  std::array<i32, simd::LaneCount> input{FixedHalf, FixedQuarter, 0, FixedHalf};
  std::array<i32, simd::LaneCount> weight{FixedMax, FixedMax, FixedMax, FixedHalf};
  std::array<i32, simd::LaneCount> out{};
  const auto norm = nn::RmsNorm(soa::I32View(input), soa::I32View(weight), soa::I32MutView(out), simd::SplatI32(1));
  TEST_ASSERT(norm.ok() && norm.processed == simd::LaneCount);
  std::array<i32, simd::LaneCount> row_input{FixedQuarter, FixedQuarter, 0, 0};
  std::array<i32, simd::LaneCount> row_weight{FixedHalf, FixedHalf, FixedHalf, FixedHalf};
  std::array<i32, simd::LaneCount> row_out{};
  const auto row_norm = nn::RmsNorm(soa::I32View(row_input), soa::I32View(row_weight), soa::I32MutView(row_out), simd::SplatI32(0));
  TEST_ASSERT(row_norm.ok() && row_norm.processed == row_input.size());
  std::array<i32, simd::LaneCount> expected_norm{};
  StoreI32x(DivFixed(MulFixed(simd::LoadI32(row_input.data()), simd::LoadI32(row_weight.data())),
                     Sqrt(simd::SplatI32(static_cast<i32>(FixedScale / 32u)))),
            expected_norm);
  TEST_ASSERT(row_out[0] == expected_norm[0] && row_out[1] == expected_norm[1]);
  std::array<i32, simd::LaneCount + 1u> tail_input{FixedHalf, FixedQuarter, 0, FixedHalf, FixedQuarter};
  std::array<i32, simd::LaneCount + 1u> tail_weight{FixedMax, FixedMax, FixedMax, FixedHalf, FixedMax};
  std::array<i32, simd::LaneCount + 1u> tail_out{-7, -7, -7, -7, -7};
  const auto tail_norm = nn::RmsNorm(soa::I32View(tail_input), soa::I32View(tail_weight), soa::I32MutView(tail_out), simd::SplatI32(1));
  TEST_ASSERT(tail_norm.ok() && tail_norm.processed == tail_input.size());
  TEST_ASSERT(tail_out.back() != -7);

  const auto rope = nn::RopePair(simd::I32x{FixedHalf, FixedHalf, 0, FixedQuarter},
                                 simd::I32x{0, FixedHalf, FixedHalf, FixedQuarter},
                                 simd::I32x{0, FixedHalf, FixedHalf, FixedHalf},
                                 simd::I32x{FixedMax, FixedHalf, FixedHalf, FixedHalf});
  ExpectI32x(rope.even,
             {detail::ScalarMulFixed(FixedHalf, FixedMax),
              detail::ScalarSubSat(detail::ScalarMulFixed(FixedHalf, FixedHalf),
                                   detail::ScalarMulFixed(FixedHalf, FixedHalf)),
              detail::ScalarSubSat(0, detail::ScalarMulFixed(FixedHalf, FixedHalf)),
              detail::ScalarSubSat(detail::ScalarMulFixed(FixedQuarter, FixedHalf),
                                   detail::ScalarMulFixed(FixedQuarter, FixedHalf))});
  std::array<i32, simd::LaneCount + 1u> even{FixedHalf, 0, 0, 0, FixedQuarter};
  std::array<i32, simd::LaneCount + 1u> odd{0, 0, 0, 0, FixedHalf};
  std::array<i32, simd::LaneCount + 1u> sin{0, 0, 0, 0, 0};
  std::array<i32, simd::LaneCount + 1u> cos{FixedMax, FixedMax, FixedMax, FixedMax, FixedMax};
  std::array<i32, simd::LaneCount + 1u> out_even{};
  std::array<i32, simd::LaneCount + 1u> out_odd{};
  const auto tail_rope = nn::RopeRow(soa::I32View(even), soa::I32View(odd), soa::I32View(sin), soa::I32View(cos),
                                     soa::I32MutView(out_even), soa::I32MutView(out_odd));
  TEST_ASSERT(tail_rope.ok() && tail_rope.processed == even.size());
  TEST_ASSERT(out_even.back() == detail::ScalarMulFixed(FixedQuarter, FixedMax));
  return 0;
}
