#include "local.hpp"

namespace program_compute_contract {

int SegmentedReduceReference() {
  const std::array<rund::kernel::u32, 6u> u32_input{3u, 1u, 4u, 2u, 5u, 1u};
  const std::array<rund::kernel::u32, 6u> heads{1u, 0u, 1u, 0u, 0u, 1u};
  std::array<rund::kernel::u32, 6u> u32_output{};
  const rund::kernel::SegmentedReduceResult sum_u32 =
      rund::kernel::ReferenceSegmentedReduceSumU32(
          u32_input.data(), heads.data(), u32_output.data(), u32_input.size());
  TEST_ASSERT(sum_u32.ok);
  TEST_ASSERT(std::string_view{sum_u32.reason} == "ok");
  TEST_ASSERT(sum_u32.segment_count == 3u);
  TEST_ASSERT(sum_u32.final_segment_total == 1u);
  TEST_ASSERT((u32_output ==
               std::array<rund::kernel::u32, 6u>{4u, 11u, 1u, 0u, 0u, 0u}));

  const std::array<rund::kernel::u64, 6u> u64_input{3u, 1u, 4u, 2u, 5u, 1u};
  std::array<rund::kernel::u64, 6u> u64_output{};
  const rund::kernel::SegmentedReduceResult max_u64 =
      rund::kernel::ReferenceSegmentedReduceMaxU64(
          u64_input.data(), heads.data(), u64_output.data(), u64_input.size());
  TEST_ASSERT(max_u64.ok);
  TEST_ASSERT(max_u64.segment_count == 3u);
  TEST_ASSERT(max_u64.final_segment_total == 1u);
  TEST_ASSERT((u64_output ==
               std::array<rund::kernel::u64, 6u>{3u, 5u, 1u, 0u, 0u, 0u}));

  const std::array<rund::kernel::u32, 2u> overflow_input{
      std::numeric_limits<rund::kernel::u32>::max(), 1u};
  const std::array<rund::kernel::u32, 2u> overflow_heads{1u, 0u};
  std::array<rund::kernel::u32, 2u> overflow_output{};
  const rund::kernel::SegmentedReduceResult overflow =
      rund::kernel::ReferenceSegmentedReduceSumU32(
          overflow_input.data(), overflow_heads.data(), overflow_output.data(),
          overflow_input.size());
  TEST_ASSERT(!overflow.ok);
  TEST_ASSERT(std::string_view{overflow.reason} ==
              "compute_segmented_reduce_sum_overflow");

  const std::array<rund::kernel::u32, 4u> mixed_input{
      std::numeric_limits<rund::kernel::u32>::max(), 1u, 0u, 0u};
  const std::array<rund::kernel::u32, 4u> mixed_heads{1u, 0u, 0u, 2u};
  std::array<rund::kernel::u32, 4u> mixed_output{};
  const rund::kernel::SegmentedReduceResult mixed =
      rund::kernel::ReferenceSegmentedReduceSumU32(
          mixed_input.data(), mixed_heads.data(), mixed_output.data(),
          mixed_input.size());
  TEST_ASSERT(!mixed.ok);
  TEST_ASSERT(std::string_view{mixed.reason} ==
              "compute_segmented_reduce_segment_invalid");

  const std::array<rund::kernel::i32, 4u> signed_input{-2, 3, -4, -1};
  const std::array<rund::kernel::u32, 4u> signed_heads{1u, 0u, 1u, 0u};
  std::array<rund::kernel::i32, 4u> signed_output{};
  const rund::kernel::SegmentedReduceResult signed_sum =
      rund::kernel::ReferenceSignedSegmentedReduce(
          signed_input.data(), signed_heads.data(), signed_output.data(),
          signed_input.size(), rund::kernel::ReduceOp::Sum);
  TEST_ASSERT(signed_sum.ok);
  TEST_ASSERT((signed_output ==
               std::array<rund::kernel::i32, 4u>{1, -5, 0, 0}));
  const std::array<rund::kernel::i32, 4u> signed_mixed_input{
      std::numeric_limits<rund::kernel::i32>::max(), 1, 0, 0};
  const rund::kernel::SegmentedReduceResult signed_mixed =
      rund::kernel::ReferenceSignedSegmentedReduce(
          signed_mixed_input.data(), mixed_heads.data(), signed_output.data(),
          signed_mixed_input.size(), rund::kernel::ReduceOp::Sum);
  TEST_ASSERT(!signed_mixed.ok);
  TEST_ASSERT(std::string_view{signed_mixed.reason} ==
              "compute_segmented_reduce_segment_invalid");
  return 0;
}

}  // namespace program_compute_contract
