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

void StoreI64x(const rund::math64::simd::I64x value,
               std::array<rund::math64::i64, rund::math64::simd::LaneCount>& out) {
  rund::math64::simd::Store(out.data(), value);
}
}  // namespace

int RunMath64NnContract() {
  using namespace rund::math64;
  ExpectI64x(nn::Relu(simd::I64x{-1, FixedMin}), {0, 0});
  ExpectI64x(nn::SigmoidApprox(simd::I64x{0, 0}), {FixedHalf, FixedHalf});

  std::array<i8, simd::LaneCount> a{2, -3};
  std::array<i8, simd::LaneCount> b{4, 5};
  const auto dot = nn::DotI8I8To64(soa::I8View(a), soa::I8View(b));
  TEST_ASSERT(dot.ok() && dot.processed == simd::LaneCount);
  ExpectI64x(dot.sum, {8, -15});
  std::array<i8, simd::LaneCount + 1u> tail_a{2, -3, 5};
  std::array<i8, simd::LaneCount + 1u> tail_b{4, 5, 9};
  const auto tail_dot = nn::DotI8I8To64(soa::I8View(tail_a), soa::I8View(tail_b));
  TEST_ASSERT(tail_dot.ok() && tail_dot.processed == tail_a.size());
  ExpectI64x(tail_dot.sum, {53, -15});
  const auto mismatch_dot = nn::DotI8I8To64(soa::I8View(tail_a.data(), 2u), soa::I8View(tail_b.data(), 3u));
  TEST_ASSERT(!mismatch_dot.ok() && mismatch_dot.processed == 0u);

  std::array<i64, simd::LaneCount> row_input{FixedQuarter, FixedQuarter};
  std::array<i64, simd::LaneCount> row_weight{FixedHalf, FixedHalf};
  std::array<i64, simd::LaneCount> row_out{};
  const auto row_norm = nn::RmsNorm(soa::I64View(row_input), soa::I64View(row_weight), soa::I64MutView(row_out), simd::SplatI64(0));
  TEST_ASSERT(row_norm.ok() && row_norm.processed == row_input.size());
  std::array<i64, simd::LaneCount> expected_norm{};
  StoreI64x(DivFixed(MulFixed(simd::LoadI64(row_input.data()), simd::LoadI64(row_weight.data())),
                     Sqrt(simd::SplatI64(static_cast<i64>(FixedScale / 16u)))),
            expected_norm);
  TEST_ASSERT(row_out[0] == expected_norm[0] && row_out[1] == expected_norm[1]);
  const auto invalid_epsilon = nn::RmsNorm(soa::I64View(row_input), soa::I64View(row_weight), soa::I64MutView(row_out), simd::SplatI64(-1));
  TEST_ASSERT(!invalid_epsilon.ok() && !invalid_epsilon.valid_epsilon);

  std::array<i64, simd::LaneCount + 1u> even{FixedHalf, 0, FixedQuarter};
  std::array<i64, simd::LaneCount + 1u> odd{0, 0, FixedHalf};
  std::array<i64, simd::LaneCount + 1u> sin{0, 0, 0};
  std::array<i64, simd::LaneCount + 1u> cos{FixedMax, FixedMax, FixedMax};
  std::array<i64, simd::LaneCount + 1u> out_even{};
  std::array<i64, simd::LaneCount + 1u> out_odd{};
  const auto tail_rope = nn::RopeRow(soa::I64View(even), soa::I64View(odd), soa::I64View(sin), soa::I64View(cos),
                                     soa::I64MutView(out_even), soa::I64MutView(out_odd));
  TEST_ASSERT(tail_rope.ok() && tail_rope.processed == even.size());
  TEST_ASSERT(out_even.back() == detail::ScalarMulFixed(FixedQuarter, FixedMax));
  std::array<i64, simd::LaneCount + 2u> overlap_even{FixedHalf, 0, FixedQuarter, 0};
  const auto overlap_rope = nn::RopeRow(soa::I64View(overlap_even.data(), 3u), soa::I64View(odd), soa::I64View(sin), soa::I64View(cos),
                                        soa::I64MutView(overlap_even.data() + 1u, 3u), soa::I64MutView(out_odd.data(), 3u));
  TEST_ASSERT(!overlap_rope.ok() && !overlap_rope.overlap_ok);
  return 0;
}
