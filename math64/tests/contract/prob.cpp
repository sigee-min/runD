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

void ExpectNear(const rund::math64::i64 value, const rund::math64::i64 expected, const rund::math64::i64 tolerance) {
  const rund::math64::i64 diff = value > expected ? value - expected : expected - value;
  TEST_ASSERT(diff <= tolerance);
}
}  // namespace

int RunMath64ProbContract() {
  using namespace rund::math64;
  std::array<i64, simd::LaneCount> logits{0, FixedHalf};
  const auto max = prob::Max(soa::I64View(logits));
  TEST_ASSERT(max.ok() && max.processed == simd::LaneCount);
  ExpectI64x(max.value, {FixedHalf, FixedHalf});
  std::array<i64, simd::LaneCount + 1u> tail_logits{0, -FixedHalf, FixedQuarter};
  const auto tail_max = prob::Max(soa::I64View(tail_logits));
  TEST_ASSERT(tail_max.ok() && tail_max.processed == tail_logits.size());
  ExpectI64x(tail_max.value, {FixedQuarter, FixedQuarter});

  std::array<i64, simd::LaneCount> out{};
  TEST_ASSERT(prob::SoftmaxApprox(soa::I64View(logits), soa::I64MutView(out)).ok());
  TEST_ASSERT(prob::LogSoftmaxApprox(soa::I64View(logits), soa::I64MutView(out)).ok());
  TEST_ASSERT(prob::LogSumExpApprox(soa::I64View(logits)).ok());
  TEST_ASSERT(prob::CrossEntropyLogitsApprox(soa::I64View(logits), 1u).ok());
  TEST_ASSERT(!prob::CrossEntropyLogitsApprox(soa::I64View(logits), logits.size()).ok());
  std::array<i64, simd::LaneCount + 1u> tail_out{-7, -7, -7};
  const auto tail_softmax = prob::SoftmaxApprox(soa::I64View(tail_logits), soa::I64MutView(tail_out));
  TEST_ASSERT(tail_softmax.ok() && tail_softmax.processed == tail_logits.size());
  TEST_ASSERT(tail_out.back() != -7);
  const auto overlap_softmax = prob::SoftmaxApprox(soa::I64View(tail_logits.data(), 2u), soa::I64MutView(tail_logits.data() + 1u, 2u));
  TEST_ASSERT(!overlap_softmax.ok() && !overlap_softmax.overlap_ok);
  std::array<i64, simd::LaneCount> equal_logits{0, 0};
  std::array<i64, simd::LaneCount> equal_out{};
  TEST_ASSERT(prob::SoftmaxApprox(soa::I64View(equal_logits), soa::I64MutView(equal_out)).ok());
  for (const i64 value : equal_out) ExpectNear(value, FixedHalf, 32);
  ExpectI64x(prob::SoftplusApprox(simd::I64x{0, FixedHalf}), {FixedHalf - 1, FixedMax});
  ExpectI64x(prob::LogSigmoidApprox(simd::I64x{0, 0}), {-FixedHalf + 1, -FixedHalf + 1});
  return 0;
}
