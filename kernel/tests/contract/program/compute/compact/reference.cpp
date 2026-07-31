#include "local.hpp"

namespace program_compute_contract {

int CompactReference() {
  const std::array<rund::kernel::u32, 7u> flags{1u, 0u, 9u, 0u, 2u, 3u, 0u};
  std::array<rund::kernel::u32, 7u> output{};
  rund::kernel::u64 output_count = 0u;
  const rund::kernel::CompactResult ok =
      rund::kernel::ReferenceCompactIdsU32(flags.data(), flags.size(),
                                           output.size(), output.data(),
                                           &output_count);
  TEST_ASSERT(ok.ok);
  TEST_ASSERT(std::string_view{ok.reason} == "ok");
  TEST_ASSERT(output_count == 4u);
  TEST_ASSERT(output[0] == 0u);
  TEST_ASSERT(output[1] == 2u);
  TEST_ASSERT(output[2] == 4u);
  TEST_ASSERT(output[3] == 5u);

  const std::array<rund::kernel::u32, 4u> capacity_flags{1u, 0u, 1u, 1u};
  std::array<rund::kernel::u32, 2u> small_output{};
  output_count = 0u;
  const rund::kernel::CompactResult small =
      rund::kernel::ReferenceCompactIdsU32(
          capacity_flags.data(), capacity_flags.size(), small_output.size(),
          small_output.data(), &output_count);
  TEST_ASSERT(!small.ok);
  TEST_ASSERT(std::string_view{small.reason} ==
              "compute_compact_capacity_insufficient");
  TEST_ASSERT(output_count == 2u);

  const std::array<rund::kernel::u32, 1u> one_flag{1u};
  std::array<rund::kernel::u32, 1u> one_output{};
  output_count = 0u;
  const rund::kernel::CompactResult zero_count =
      rund::kernel::ReferenceCompactIdsU32(one_flag.data(), 0u,
                                           one_output.size(), one_output.data(),
                                           &output_count);
  TEST_ASSERT(!zero_count.ok);
  TEST_ASSERT(std::string_view{zero_count.reason} ==
              "compute_compact_count_zero");

  const rund::kernel::CompactResult zero_capacity =
      rund::kernel::ReferenceCompactIdsU32(one_flag.data(), one_flag.size(), 0u,
                                           one_output.data(), &output_count);
  TEST_ASSERT(!zero_capacity.ok);
  TEST_ASSERT(std::string_view{zero_capacity.reason} ==
              "compute_compact_capacity_zero");

  const rund::kernel::CompactResult missing_flags =
      rund::kernel::ReferenceCompactIdsU32(nullptr, one_flag.size(),
                                           one_output.size(), one_output.data(),
                                           &output_count);
  const rund::kernel::CompactResult missing_output =
      rund::kernel::ReferenceCompactIdsU32(one_flag.data(), one_flag.size(),
                                           one_output.size(), nullptr,
                                           &output_count);
  const rund::kernel::CompactResult missing_count =
      rund::kernel::ReferenceCompactIdsU32(one_flag.data(), one_flag.size(),
                                           one_output.size(), one_output.data(),
                                           nullptr);
  TEST_ASSERT(!missing_flags.ok);
  TEST_ASSERT(std::string_view{missing_flags.reason} ==
              "compute_compact_buffer_invalid");
  TEST_ASSERT(!missing_output.ok);
  TEST_ASSERT(std::string_view{missing_output.reason} ==
              "compute_compact_buffer_invalid");
  TEST_ASSERT(!missing_count.ok);
  TEST_ASSERT(std::string_view{missing_count.reason} ==
              "compute_compact_buffer_invalid");
  return 0;
}

} // namespace program_compute_contract
