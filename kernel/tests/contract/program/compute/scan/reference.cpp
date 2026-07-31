#include "local.hpp"

namespace program_compute_contract {

int ScanReference() {
  const std::array<rund::kernel::u32, 5u> u32_input{3u, 1u, 4u, 0u, 2u};
  std::array<rund::kernel::u32, 5u> u32_output{};
  rund::kernel::u64 u32_total = 0u;
  const rund::kernel::ScanResult exclusive_u32 =
      rund::kernel::ReferenceExclusiveScanU32(
          u32_input.data(), u32_output.data(), u32_input.size(), &u32_total);
  TEST_ASSERT(exclusive_u32.ok);
  TEST_ASSERT(std::string_view{exclusive_u32.reason} == "ok");
  TEST_ASSERT(u32_output[0] == 0u);
  TEST_ASSERT(u32_output[1] == 3u);
  TEST_ASSERT(u32_output[2] == 4u);
  TEST_ASSERT(u32_output[3] == 8u);
  TEST_ASSERT(u32_output[4] == 8u);
  TEST_ASSERT(u32_total == 10u);

  const std::array<rund::kernel::u64, 4u> u64_input{5u, 0u, 7u, 2u};
  std::array<rund::kernel::u64, 4u> u64_output{};
  rund::kernel::u64 u64_total = 0u;
  const rund::kernel::ScanResult exclusive_u64 =
      rund::kernel::ReferenceExclusiveScanU64(
          u64_input.data(), u64_output.data(), u64_input.size(), &u64_total);
  TEST_ASSERT(exclusive_u64.ok);
  TEST_ASSERT(u64_output[0] == 0u);
  TEST_ASSERT(u64_output[1] == 5u);
  TEST_ASSERT(u64_output[2] == 5u);
  TEST_ASSERT(u64_output[3] == 12u);
  TEST_ASSERT(u64_total == 14u);

  std::array<rund::kernel::u32, 5u> inclusive_u32_output{};
  u32_total = 0u;
  const rund::kernel::ScanResult inclusive_u32 =
      rund::kernel::ReferenceInclusiveScanU32(u32_input.data(),
                                              inclusive_u32_output.data(),
                                              u32_input.size(), &u32_total);
  TEST_ASSERT(inclusive_u32.ok);
  TEST_ASSERT(std::string_view{inclusive_u32.reason} == "ok");
  TEST_ASSERT(inclusive_u32_output[0] == 3u);
  TEST_ASSERT(inclusive_u32_output[1] == 4u);
  TEST_ASSERT(inclusive_u32_output[2] == 8u);
  TEST_ASSERT(inclusive_u32_output[3] == 8u);
  TEST_ASSERT(inclusive_u32_output[4] == 10u);
  TEST_ASSERT(u32_total == 10u);

  std::array<rund::kernel::u64, 4u> inclusive_u64_output{};
  u64_total = 0u;
  const rund::kernel::ScanResult inclusive_u64 =
      rund::kernel::ReferenceInclusiveScanU64(u64_input.data(),
                                              inclusive_u64_output.data(),
                                              u64_input.size(), &u64_total);
  TEST_ASSERT(inclusive_u64.ok);
  TEST_ASSERT(inclusive_u64_output[0] == 5u);
  TEST_ASSERT(inclusive_u64_output[1] == 5u);
  TEST_ASSERT(inclusive_u64_output[2] == 12u);
  TEST_ASSERT(inclusive_u64_output[3] == 14u);
  TEST_ASSERT(u64_total == 14u);

  const std::array<rund::kernel::u32, 2u> overflow_u32_input{
      std::numeric_limits<rund::kernel::u32>::max(), 1u};
  std::array<rund::kernel::u32, 2u> overflow_u32_output{};
  u32_total = 0u;
  const rund::kernel::ScanResult overflow_u32 =
      rund::kernel::ReferenceExclusiveScanU32(
          overflow_u32_input.data(), overflow_u32_output.data(),
          overflow_u32_input.size(), &u32_total);
  TEST_ASSERT(!overflow_u32.ok);
  TEST_ASSERT(std::string_view{overflow_u32.reason} ==
              "compute_scan_sum_overflow");

  const std::array<rund::kernel::u64, 2u> overflow_u64_input{
      std::numeric_limits<rund::kernel::u64>::max(), 1u};
  std::array<rund::kernel::u64, 2u> overflow_u64_output{};
  u64_total = 0u;
  const rund::kernel::ScanResult overflow_u64 =
      rund::kernel::ReferenceExclusiveScanU64(
          overflow_u64_input.data(), overflow_u64_output.data(),
          overflow_u64_input.size(), &u64_total);
  TEST_ASSERT(!overflow_u64.ok);
  TEST_ASSERT(std::string_view{overflow_u64.reason} ==
              "compute_scan_sum_overflow");

  u64_total = 0u;
  const rund::kernel::ScanResult missing =
      rund::kernel::ReferenceExclusiveScanU32(nullptr, nullptr, 4u, &u64_total);
  TEST_ASSERT(!missing.ok);
  TEST_ASSERT(std::string_view{missing.reason} ==
              "compute_scan_buffer_invalid");
  return 0;
}

} // namespace program_compute_contract
