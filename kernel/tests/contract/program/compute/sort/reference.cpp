#include "local.hpp"

namespace program_compute_contract {

int test_compute_sort_cpu_reference_is_stable_for_equal_u32_keys() {
  const std::array<rund::kernel::u32, 5u> keys{3u, 1u, 3u, 2u, 1u};
  const std::array<rund::kernel::u32, 5u> values{30u, 10u, 31u, 20u, 11u};
  std::array<rund::kernel::u32, 5u> out_keys{};
  std::array<rund::kernel::u32, 5u> out_values{};
  std::array<rund::kernel::u64, 5u> out_original_indices{};

  const rund::kernel::SortResult result =
      rund::kernel::ReferenceStableSortU32(keys.data(), values.data(),
                                          out_keys.data(), out_values.data(),
                                          out_original_indices.data(),
                                          keys.size());

  TEST_ASSERT(result.ok);
  TEST_ASSERT(std::string_view{result.reason} == "ok");
  TEST_ASSERT(out_keys[0] == 1u);
  TEST_ASSERT(out_keys[1] == 1u);
  TEST_ASSERT(out_keys[2] == 2u);
  TEST_ASSERT(out_keys[3] == 3u);
  TEST_ASSERT(out_keys[4] == 3u);
  TEST_ASSERT(out_values[0] == 10u);
  TEST_ASSERT(out_values[1] == 11u);
  TEST_ASSERT(out_values[2] == 20u);
  TEST_ASSERT(out_values[3] == 30u);
  TEST_ASSERT(out_values[4] == 31u);
  TEST_ASSERT(out_original_indices[0] == 1u);
  TEST_ASSERT(out_original_indices[1] == 4u);
  TEST_ASSERT(out_original_indices[2] == 3u);
  TEST_ASSERT(out_original_indices[3] == 0u);
  TEST_ASSERT(out_original_indices[4] == 2u);
  return 0;
}

int test_compute_sort_cpu_reference_sorts_u64_keys() {
  const std::array<rund::kernel::u64, 4u> keys{9u, 2u, 9u, 1u};
  const std::array<rund::kernel::u32, 4u> values{90u, 20u, 91u, 10u};
  std::array<rund::kernel::u64, 4u> out_keys{};
  std::array<rund::kernel::u32, 4u> out_values{};
  std::array<rund::kernel::u64, 4u> out_original_indices{};

  const rund::kernel::SortResult result =
      rund::kernel::ReferenceStableSortU64(keys.data(), values.data(),
                                          out_keys.data(), out_values.data(),
                                          out_original_indices.data(),
                                          keys.size());

  TEST_ASSERT(result.ok);
  TEST_ASSERT(out_keys[0] == 1u);
  TEST_ASSERT(out_keys[1] == 2u);
  TEST_ASSERT(out_keys[2] == 9u);
  TEST_ASSERT(out_keys[3] == 9u);
  TEST_ASSERT(out_values[0] == 10u);
  TEST_ASSERT(out_values[1] == 20u);
  TEST_ASSERT(out_values[2] == 90u);
  TEST_ASSERT(out_values[3] == 91u);
  TEST_ASSERT(out_original_indices[0] == 3u);
  TEST_ASSERT(out_original_indices[1] == 1u);
  TEST_ASSERT(out_original_indices[2] == 0u);
  TEST_ASSERT(out_original_indices[3] == 2u);
  return 0;
}

int test_compute_sort_cpu_reference_rejects_zero_count() {
  const std::array<rund::kernel::u32, 1u> keys{1u};
  const std::array<rund::kernel::u32, 1u> values{10u};
  std::array<rund::kernel::u32, 1u> out_keys{};
  std::array<rund::kernel::u32, 1u> out_values{};
  std::array<rund::kernel::u64, 1u> out_original_indices{};

  const rund::kernel::SortResult result =
      rund::kernel::ReferenceStableSortU32(keys.data(), values.data(),
                                          out_keys.data(), out_values.data(),
                                          out_original_indices.data(), 0u);

  TEST_ASSERT(!result.ok);
  TEST_ASSERT(std::string_view{result.reason} == "compute_sort_count_zero");
  return 0;
}

int test_compute_sort_cpu_reference_rejects_missing_buffers() {
  const std::array<rund::kernel::u32, 1u> values{10u};
  std::array<rund::kernel::u32, 1u> out_keys{};
  std::array<rund::kernel::u32, 1u> out_values{};
  std::array<rund::kernel::u64, 1u> out_original_indices{};

  const rund::kernel::SortResult missing_keys =
      rund::kernel::ReferenceStableSortU32(nullptr, values.data(),
                                          out_keys.data(), out_values.data(),
                                          out_original_indices.data(),
                                          values.size());
  const rund::kernel::SortResult missing_values =
      rund::kernel::ReferenceStableSortU32(out_keys.data(), nullptr,
                                          out_keys.data(), out_values.data(),
                                          out_original_indices.data(),
                                          values.size());
  const rund::kernel::SortResult missing_outputs =
      rund::kernel::ReferenceStableSortU32(out_keys.data(), values.data(),
                                          nullptr, out_values.data(),
                                          out_original_indices.data(),
                                          values.size());
  const rund::kernel::SortResult missing_output_values =
      rund::kernel::ReferenceStableSortU32(out_keys.data(), values.data(),
                                          out_keys.data(), nullptr,
                                          out_original_indices.data(),
                                          values.size());
  const rund::kernel::SortResult missing_output_indices =
      rund::kernel::ReferenceStableSortU32(out_keys.data(), values.data(),
                                          out_keys.data(), out_values.data(),
                                          nullptr, values.size());

  TEST_ASSERT(!missing_keys.ok);
  TEST_ASSERT(std::string_view{missing_keys.reason} ==
              "compute_sort_buffer_invalid");
  TEST_ASSERT(!missing_values.ok);
  TEST_ASSERT(std::string_view{missing_values.reason} ==
              "compute_sort_buffer_invalid");
  TEST_ASSERT(!missing_outputs.ok);
  TEST_ASSERT(std::string_view{missing_outputs.reason} ==
              "compute_sort_buffer_invalid");
  TEST_ASSERT(!missing_output_values.ok);
  TEST_ASSERT(std::string_view{missing_output_values.reason} ==
              "compute_sort_buffer_invalid");
  TEST_ASSERT(!missing_output_indices.ok);
  TEST_ASSERT(std::string_view{missing_output_indices.reason} ==
              "compute_sort_buffer_invalid");
  return 0;
}

}  // namespace program_compute_contract
