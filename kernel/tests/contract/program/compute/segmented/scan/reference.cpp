#include "local.hpp"

namespace program_compute_contract {

int SegmentedScanReference() {
  const std::array<rund::kernel::u32, 6u> u32_input{3u, 1u, 4u, 2u, 5u, 1u};
  const std::array<rund::kernel::u32, 6u> heads{1u, 0u, 1u, 0u, 0u, 1u};
  std::array<rund::kernel::u32, 6u> u32_output{};
  const rund::kernel::SegmentedScanResult exclusive_u32 =
      rund::kernel::ReferenceExclusiveSegmentedScanU32(
          u32_input.data(), heads.data(), u32_output.data(), u32_input.size());
  TEST_ASSERT(exclusive_u32.ok);
  TEST_ASSERT(std::string_view{exclusive_u32.reason} == "ok");
  TEST_ASSERT(exclusive_u32.segment_count == 3u);
  TEST_ASSERT(exclusive_u32.final_segment_total == 1u);
  TEST_ASSERT((u32_output ==
               std::array<rund::kernel::u32, 6u>{0u, 3u, 0u, 4u, 6u, 0u}));

  const std::array<rund::kernel::u64, 6u> u64_input{3u, 1u, 4u, 2u, 5u, 1u};
  std::array<rund::kernel::u64, 6u> u64_output{};
  const rund::kernel::SegmentedScanResult inclusive_u64 =
      rund::kernel::ReferenceInclusiveSegmentedScanU64(
          u64_input.data(), heads.data(), u64_output.data(), u64_input.size());
  TEST_ASSERT(inclusive_u64.ok);
  TEST_ASSERT(inclusive_u64.segment_count == 3u);
  TEST_ASSERT(inclusive_u64.final_segment_total == 1u);
  TEST_ASSERT((u64_output ==
               std::array<rund::kernel::u64, 6u>{3u, 4u, 4u, 6u, 11u, 1u}));

  const std::array<rund::kernel::u32, 3u> invalid_input{1u, 2u, 3u};
  const std::array<rund::kernel::u32, 3u> missing_first_head{0u, 1u, 0u};
  const std::array<rund::kernel::u32, 3u> invalid_head{1u, 2u, 0u};
  std::array<rund::kernel::u32, 3u> invalid_output{};
  const rund::kernel::SegmentedScanResult missing_head =
      rund::kernel::ReferenceExclusiveSegmentedScanU32(
          invalid_input.data(), missing_first_head.data(),
          invalid_output.data(), invalid_input.size());
  const rund::kernel::SegmentedScanResult invalid_segment =
      rund::kernel::ReferenceExclusiveSegmentedScanU32(
          invalid_input.data(), invalid_head.data(), invalid_output.data(),
          invalid_input.size());
  TEST_ASSERT(!missing_head.ok);
  TEST_ASSERT(std::string_view{missing_head.reason} ==
              "compute_segmented_scan_segment_invalid");
  TEST_ASSERT(!invalid_segment.ok);
  TEST_ASSERT(std::string_view{invalid_segment.reason} ==
              "compute_segmented_scan_segment_invalid");

  const std::array<rund::kernel::u32, 2u> overflow_input{
      std::numeric_limits<rund::kernel::u32>::max(), 1u};
  const std::array<rund::kernel::u32, 2u> overflow_heads{1u, 0u};
  std::array<rund::kernel::u32, 2u> overflow_output{};
  const rund::kernel::SegmentedScanResult overflow =
      rund::kernel::ReferenceInclusiveSegmentedScanU32(
          overflow_input.data(), overflow_heads.data(), overflow_output.data(),
          overflow_input.size());
  TEST_ASSERT(!overflow.ok);
  TEST_ASSERT(std::string_view{overflow.reason} ==
              "compute_segmented_scan_sum_overflow");
  const rund::kernel::SegmentedScanResult exclusive_overflow =
      rund::kernel::ReferenceExclusiveSegmentedScanU32(
          overflow_input.data(), overflow_heads.data(), overflow_output.data(),
          overflow_input.size());
  TEST_ASSERT(!exclusive_overflow.ok);
  TEST_ASSERT(std::string_view{exclusive_overflow.reason} ==
              "compute_segmented_scan_sum_overflow");

  const std::array<rund::kernel::u32, 4u> mixed_input{
      std::numeric_limits<rund::kernel::u32>::max(), 1u, 0u, 0u};
  const std::array<rund::kernel::u32, 4u> mixed_heads{1u, 0u, 0u, 2u};
  std::array<rund::kernel::u32, 4u> mixed_output{};
  const rund::kernel::SegmentedScanResult mixed_inclusive =
      rund::kernel::ReferenceInclusiveSegmentedScanU32(
          mixed_input.data(), mixed_heads.data(), mixed_output.data(),
          mixed_input.size());
  const rund::kernel::SegmentedScanResult mixed_exclusive =
      rund::kernel::ReferenceExclusiveSegmentedScanU32(
          mixed_input.data(), mixed_heads.data(), mixed_output.data(),
          mixed_input.size());
  TEST_ASSERT(!mixed_inclusive.ok);
  TEST_ASSERT(std::string_view{mixed_inclusive.reason} ==
              "compute_segmented_scan_segment_invalid");
  TEST_ASSERT(!mixed_exclusive.ok);
  TEST_ASSERT(std::string_view{mixed_exclusive.reason} ==
              "compute_segmented_scan_segment_invalid");

  const std::array<rund::kernel::i32, 4u> signed_input{-2, 3, 4, -1};
  const std::array<rund::kernel::u32, 4u> signed_heads{1u, 0u, 1u, 0u};
  std::array<rund::kernel::i32, 4u> signed_output{};
  const rund::kernel::SegmentedScanResult signed_inclusive =
      rund::kernel::ReferenceSignedSegmentedScan(
          signed_input.data(), signed_heads.data(), signed_output.data(),
          signed_input.size(), true);
  TEST_ASSERT(signed_inclusive.ok);
  TEST_ASSERT((signed_output ==
               std::array<rund::kernel::i32, 4u>{-2, 1, 4, 3}));
  const std::array<rund::kernel::i32, 4u> signed_mixed_input{
      std::numeric_limits<rund::kernel::i32>::max(), 1, 0, 0};
  const rund::kernel::SegmentedScanResult signed_mixed =
      rund::kernel::ReferenceSignedSegmentedScan(
          signed_mixed_input.data(), mixed_heads.data(), signed_output.data(),
          signed_mixed_input.size(), true);
  TEST_ASSERT(!signed_mixed.ok);
  TEST_ASSERT(std::string_view{signed_mixed.reason} ==
              "compute_segmented_scan_segment_invalid");
  return 0;
}

} // namespace program_compute_contract
