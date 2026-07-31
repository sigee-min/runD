#include "local.hpp"

namespace program_compute_contract {

int GatherReference() {
  const std::array<rund::kernel::u32, 6u> u32_values{10u, 20u, 30u,
                                                     40u, 50u, 60u};
  const std::array<rund::kernel::u32, 4u> u32_indices{3u, 0u, 3u, 5u};
  std::array<rund::kernel::u32, 4u> u32_output{};
  const rund::kernel::GatherResult u32_result =
      rund::kernel::ReferenceGatherU32(u32_values.data(), u32_indices.data(),
                                       u32_output.data(), u32_indices.size(),
                                       u32_values.size());
  TEST_ASSERT(u32_result.ok);
  TEST_ASSERT(u32_output[0] == 40u);
  TEST_ASSERT(u32_output[1] == 10u);
  TEST_ASSERT(u32_output[2] == 40u);
  TEST_ASSERT(u32_output[3] == 60u);

  const std::array<rund::kernel::u64, 5u> u64_values{7u, 11u, 13u, 17u, 19u};
  const std::array<rund::kernel::u32, 3u> u64_indices{4u, 1u, 2u};
  std::array<rund::kernel::u64, 3u> u64_output{};
  const rund::kernel::GatherResult u64_result =
      rund::kernel::ReferenceGatherU64(u64_values.data(), u64_indices.data(),
                                       u64_output.data(), u64_indices.size(),
                                       u64_values.size());
  TEST_ASSERT(u64_result.ok);
  TEST_ASSERT(u64_output[0] == 19u);
  TEST_ASSERT(u64_output[1] == 11u);
  TEST_ASSERT(u64_output[2] == 13u);

  const std::array<rund::kernel::u32, 2u> values{1u, 2u};
  const std::array<rund::kernel::u32, 2u> out_of_range_indices{0u, 2u};
  std::array<rund::kernel::u32, 2u> output{17u, 19u};
  const rund::kernel::GatherResult out_of_range =
      rund::kernel::ReferenceGatherU32(
          values.data(), out_of_range_indices.data(), output.data(),
          out_of_range_indices.size(), values.size());
  TEST_ASSERT(!out_of_range.ok);
  TEST_ASSERT(std::string_view{out_of_range.reason} ==
              "compute_gather_index_out_of_range");
  TEST_ASSERT(out_of_range.first_invalid_index == 1u);
  TEST_ASSERT((output == std::array<rund::kernel::u32, 2u>{17u, 19u}));

  const rund::kernel::GatherResult missing =
      rund::kernel::ReferenceGatherU32(nullptr, nullptr, nullptr, 4u, 8u);
  TEST_ASSERT(!missing.ok);
  TEST_ASSERT(std::string_view{missing.reason} ==
              "compute_gather_buffer_invalid");
  return 0;
}

} // namespace program_compute_contract
