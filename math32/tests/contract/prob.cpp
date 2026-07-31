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
void ExpectNear(const rund::math32::i32 value, const rund::math32::i32 expected, const rund::math32::i32 tolerance) {
  const rund::math32::i32 diff = value > expected ? value - expected : expected - value;
  TEST_ASSERT(diff <= tolerance);
}
}  // namespace

int RunMath32ProbContract() {
  using namespace rund::math32;
  std::array<i32, simd::LaneCount> logits{0, FixedHalf, -FixedHalf, FixedQuarter};
  const auto max = prob::Max(soa::I32View(logits));
  TEST_ASSERT(max.ok() && max.processed == simd::LaneCount);
  ExpectI32x(max.value, {FixedHalf, FixedHalf, FixedHalf, FixedHalf});
  std::array<i32, simd::LaneCount + 1u> tail_logits{0, -FixedHalf, -FixedHalf, FixedQuarter, FixedHalf};
  const auto tail_max = prob::Max(soa::I32View(tail_logits));
  TEST_ASSERT(tail_max.ok() && tail_max.processed == tail_logits.size());
  ExpectI32x(tail_max.value, {FixedHalf, FixedHalf, FixedHalf, FixedHalf});

  std::array<i32, simd::LaneCount> out{};
  TEST_ASSERT(prob::SoftmaxApprox(soa::I32View(logits), soa::I32MutView(out)).ok());
  TEST_ASSERT(prob::LogSoftmaxApprox(soa::I32View(logits), soa::I32MutView(out)).ok());
  TEST_ASSERT(prob::LogSumExpApprox(soa::I32View(logits)).ok());
  TEST_ASSERT(prob::CrossEntropyLogitsApprox(soa::I32View(logits), 1u).ok());
  std::array<i32, simd::LaneCount + 1u> tail_out{-7, -7, -7, -7, -7};
  const auto tail_softmax = prob::SoftmaxApprox(soa::I32View(tail_logits), soa::I32MutView(tail_out));
  TEST_ASSERT(tail_softmax.ok() && tail_softmax.processed == tail_logits.size());
  TEST_ASSERT(tail_out.back() != -7);
  tail_out.fill(-7);
  const auto tail_logsoftmax = prob::LogSoftmaxApprox(soa::I32View(tail_logits), soa::I32MutView(tail_out));
  TEST_ASSERT(tail_logsoftmax.ok() && tail_logsoftmax.processed == tail_logits.size());
  TEST_ASSERT(tail_out.back() != -7);
  const auto tail_logsum = prob::LogSumExpApprox(soa::I32View(tail_logits));
  TEST_ASSERT(tail_logsum.ok() && tail_logsum.processed == tail_logits.size());
  std::array<i32, simd::LaneCount> equal_logits{0, 0, 0, 0};
  std::array<i32, simd::LaneCount> equal_out{};
  TEST_ASSERT(prob::SoftmaxApprox(soa::I32View(equal_logits), soa::I32MutView(equal_out)).ok());
  for (const i32 value : equal_out) ExpectNear(value, FixedQuarter, 16);
  const auto equal_logsum = prob::LogSumExpApprox(soa::I32View(equal_logits));
  TEST_ASSERT(equal_logsum.ok() && equal_logsum.value[0] > 0 && equal_logsum.value[0] == equal_logsum.value[1]);
  ExpectI32x(prob::SoftplusApprox(simd::I32x{0, 1, -1, FixedHalf}),
             {FixedHalf - 1, FixedHalf - 5, FixedHalf - 6, FixedMax});
  ExpectI32x(prob::LogSigmoidApprox(simd::I32x{0, 0, 0, 0}), {-FixedHalf + 1, -FixedHalf + 1, -FixedHalf + 1, -FixedHalf + 1});
  return 0;
}
