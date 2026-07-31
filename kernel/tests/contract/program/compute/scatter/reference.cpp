#include "local.hpp"

namespace program_compute_contract {

int ScatterReference() {
  const std::array<rund::kernel::u32, 4u> u32_values{10u, 20u, 30u, 40u};
  const std::array<rund::kernel::u32, 4u> u32_indices{3u, 0u, 5u, 2u};
  std::array<rund::kernel::u32, 6u> u32_output{99u, 99u, 99u, 99u, 99u, 99u};
  const rund::kernel::ScatterResult u32_result =
      rund::kernel::ReferenceScatterU32(u32_values.data(), u32_indices.data(),
                                        u32_output.data(), u32_values.size(),
                                        u32_output.size());
  TEST_ASSERT(u32_result.ok);
  TEST_ASSERT(u32_output[0] == 20u);
  TEST_ASSERT(u32_output[1] == 99u);
  TEST_ASSERT(u32_output[2] == 40u);
  TEST_ASSERT(u32_output[3] == 10u);
  TEST_ASSERT(u32_output[4] == 99u);
  TEST_ASSERT(u32_output[5] == 30u);

  const std::array<rund::kernel::u64, 3u> u64_values{7u, 11u, 13u};
  const std::array<rund::kernel::u32, 3u> u64_indices{4u, 1u, 0u};
  std::array<rund::kernel::u64, 5u> u64_output{55u, 55u, 55u, 55u, 55u};
  const rund::kernel::ScatterResult u64_result =
      rund::kernel::ReferenceScatterU64(u64_values.data(), u64_indices.data(),
                                        u64_output.data(), u64_values.size(),
                                        u64_output.size());
  TEST_ASSERT(u64_result.ok);
  TEST_ASSERT(u64_output[0] == 13u);
  TEST_ASSERT(u64_output[1] == 11u);
  TEST_ASSERT(u64_output[4] == 7u);

  const std::array<rund::kernel::u32, 2u> range_values{1u, 2u};
  const std::array<rund::kernel::u32, 2u> range_indices{0u, 2u};
  std::array<rund::kernel::u32, 2u> range_output{};
  const rund::kernel::ScatterResult out_of_range =
      rund::kernel::ReferenceScatterU32(
          range_values.data(), range_indices.data(), range_output.data(),
          range_values.size(), range_output.size());
  TEST_ASSERT(!out_of_range.ok);
  TEST_ASSERT(std::string_view{out_of_range.reason} ==
              "compute_scatter_index_out_of_range");
  TEST_ASSERT(out_of_range.first_rejected_index == 1u);

  const std::array<rund::kernel::u32, 3u> duplicate_values{1u, 2u, 3u};
  const std::array<rund::kernel::u32, 3u> duplicate_indices{0u, 1u, 0u};
  std::array<rund::kernel::u32, 2u> duplicate_output{};
  const rund::kernel::ScatterResult duplicate =
      rund::kernel::ReferenceScatterU32(
          duplicate_values.data(), duplicate_indices.data(),
          duplicate_output.data(), duplicate_values.size(),
          duplicate_output.size());
  TEST_ASSERT(!duplicate.ok);
  TEST_ASSERT(std::string_view{duplicate.reason} ==
              "compute_scatter_duplicate_index");
  TEST_ASSERT(duplicate.first_rejected_index == 2u);

  const rund::kernel::ScatterResult missing =
      rund::kernel::ReferenceScatterU32(nullptr, nullptr, nullptr, 4u, 8u);
  TEST_ASSERT(!missing.ok);
  TEST_ASSERT(std::string_view{missing.reason} ==
              "compute_scatter_buffer_invalid");

  const rund::kernel::ScatterResult encoded = rund::kernel::ReferenceScatterU32(
      nullptr, nullptr, nullptr,
      rund::kernel::scatter_plan_detail::kMaxEncodedCount + 1u, 8u);
  TEST_ASSERT(!encoded.ok);
  TEST_ASSERT(std::string_view{encoded.reason} ==
              "compute_scatter_count_unsupported");
  return 0;
}

} // namespace program_compute_contract
