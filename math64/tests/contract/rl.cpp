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

int RunMath64RlContract() {
  using namespace rund::math64;
  const simd::I64x reward{1, 2};
  const simd::I64x gamma{FixedHalf, FixedQuarter};
  const simd::I64x next{2, 8};
  ExpectI64x(rl::Bellman(reward, gamma, next),
             {detail::ScalarAddSat(1, detail::ScalarMulFixed(FixedHalf, 2)),
              detail::ScalarAddSat(2, detail::ScalarMulFixed(FixedQuarter, 8))});
  ExpectI64x(rl::TdError(reward, gamma, next, simd::I64x{1, 1}),
             {detail::ScalarSubSat(detail::ScalarAddSat(1, detail::ScalarMulFixed(FixedHalf, 2)), 1),
              detail::ScalarSubSat(detail::ScalarAddSat(2, detail::ScalarMulFixed(FixedQuarter, 8)), 1)});
  ExpectI64x(rl::QUpdate(simd::I64x{1, 2}, gamma, next),
             {detail::ScalarAddSat(1, detail::ScalarMulFixed(FixedHalf, 2)),
              detail::ScalarAddSat(2, detail::ScalarMulFixed(FixedQuarter, 8))});
  ExpectI64x(rl::ReturnStep(reward, gamma, next),
             {detail::ScalarAddSat(1, detail::ScalarMulFixed(FixedHalf, 2)),
              detail::ScalarAddSat(2, detail::ScalarMulFixed(FixedQuarter, 8))});
  ExpectI64x(rl::GaeStep(reward, gamma, next, simd::I64x{1, 1}, simd::SplatI64(FixedHalf), simd::I64x{2, 2}),
             {detail::ScalarAddSat(detail::ScalarSubSat(detail::ScalarAddSat(1, detail::ScalarMulFixed(FixedHalf, 2)), 1),
                                   detail::ScalarMulFixed(detail::ScalarMulFixed(FixedHalf, FixedHalf), 2)),
              detail::ScalarAddSat(detail::ScalarSubSat(detail::ScalarAddSat(2, detail::ScalarMulFixed(FixedQuarter, 8)), 1),
                                   detail::ScalarMulFixed(detail::ScalarMulFixed(FixedQuarter, FixedHalf), 2))});
  return 0;
}
