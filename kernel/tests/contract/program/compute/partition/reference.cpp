#include "test/assert.hpp"

#include <kernel/program/compute/partition/reference.hpp>

#include <array>
#include <string_view>

namespace program_compute_contract {
namespace {

int test_compute_partition_cpu_reference_preserves_stable_groups() {
  const std::array<rund::kernel::u32, 6u> flags{1u, 0u, 3u, 0u, 0u, 2u};
  const std::array<rund::kernel::u32, 6u> values{10u, 11u, 12u, 13u, 14u, 15u};
  std::array<rund::kernel::u32, 6u> output{};
  rund::kernel::u64 false_count = 0u;
  rund::kernel::u64 true_count = 0u;

  const rund::kernel::PartitionResult result =
      rund::kernel::ReferenceStablePartitionU32(flags.data(), values.data(),
                                                values.size(), output.data(),
                                                &false_count, &true_count);

  TEST_ASSERT(result.ok);
  TEST_ASSERT(std::string_view{result.reason} == "ok");
  TEST_ASSERT(false_count == 3u);
  TEST_ASSERT(true_count == 3u);
  TEST_ASSERT(output[0] == 11u);
  TEST_ASSERT(output[1] == 13u);
  TEST_ASSERT(output[2] == 14u);
  TEST_ASSERT(output[3] == 10u);
  TEST_ASSERT(output[4] == 12u);
  TEST_ASSERT(output[5] == 15u);
  return 0;
}

int test_compute_partition_cpu_reference_rejects_zero_count() {
  const std::array<rund::kernel::u32, 1u> flags{1u};
  const std::array<rund::kernel::u32, 1u> values{7u};
  std::array<rund::kernel::u32, 1u> output{};
  rund::kernel::u64 false_count = 0u;
  rund::kernel::u64 true_count = 0u;

  const rund::kernel::PartitionResult result =
      rund::kernel::ReferenceStablePartitionU32(flags.data(), values.data(), 0u,
                                                output.data(), &false_count,
                                                &true_count);

  TEST_ASSERT(!result.ok);
  TEST_ASSERT(std::string_view{result.reason} ==
              "compute_partition_count_zero");
  return 0;
}

int test_compute_partition_cpu_reference_handles_extreme_groups() {
  constexpr std::array<rund::kernel::u32, 4u> values{10u, 11u, 12u, 13u};
  constexpr std::array<rund::kernel::u32, 4u> all_false{0u, 0u, 0u, 0u};
  constexpr std::array<rund::kernel::u32, 4u> all_true{1u, 2u, 3u, 4u};
  std::array<rund::kernel::u32, 4u> output{};
  rund::kernel::u64 false_count = 0u;
  rund::kernel::u64 true_count = 0u;

  const rund::kernel::PartitionResult false_result =
      rund::kernel::ReferenceStablePartitionU32(all_false.data(), values.data(),
                                                values.size(), output.data(),
                                                &false_count, &true_count);
  TEST_ASSERT(false_result.ok);
  TEST_ASSERT(false_count == values.size());
  TEST_ASSERT(true_count == 0u);
  TEST_ASSERT(output == values);

  output.fill(0u);
  const rund::kernel::PartitionResult true_result =
      rund::kernel::ReferenceStablePartitionU32(all_true.data(), values.data(),
                                                values.size(), output.data(),
                                                &false_count, &true_count);
  TEST_ASSERT(true_result.ok);
  TEST_ASSERT(false_count == 0u);
  TEST_ASSERT(true_count == values.size());
  TEST_ASSERT(output == values);
  return 0;
}

int test_compute_partition_cpu_reference_rejects_missing_buffers() {
  const std::array<rund::kernel::u32, 1u> flags{1u};
  const std::array<rund::kernel::u32, 1u> values{7u};
  std::array<rund::kernel::u32, 1u> output{};
  rund::kernel::u64 false_count = 0u;
  rund::kernel::u64 true_count = 0u;

  const rund::kernel::PartitionResult missing_flags =
      rund::kernel::ReferenceStablePartitionU32(nullptr, values.data(),
                                                values.size(), output.data(),
                                                &false_count, &true_count);
  const rund::kernel::PartitionResult missing_values =
      rund::kernel::ReferenceStablePartitionU32(flags.data(), nullptr,
                                                values.size(), output.data(),
                                                &false_count, &true_count);
  const rund::kernel::PartitionResult missing_output =
      rund::kernel::ReferenceStablePartitionU32(flags.data(), values.data(),
                                                values.size(), nullptr,
                                                &false_count, &true_count);
  const rund::kernel::PartitionResult missing_false_count =
      rund::kernel::ReferenceStablePartitionU32(flags.data(), values.data(),
                                                values.size(), output.data(),
                                                nullptr, &true_count);
  const rund::kernel::PartitionResult missing_true_count =
      rund::kernel::ReferenceStablePartitionU32(flags.data(), values.data(),
                                                values.size(), output.data(),
                                                &false_count, nullptr);

  TEST_ASSERT(!missing_flags.ok);
  TEST_ASSERT(std::string_view{missing_flags.reason} ==
              "compute_partition_buffer_invalid");
  TEST_ASSERT(!missing_values.ok);
  TEST_ASSERT(std::string_view{missing_values.reason} ==
              "compute_partition_buffer_invalid");
  TEST_ASSERT(!missing_output.ok);
  TEST_ASSERT(std::string_view{missing_output.reason} ==
              "compute_partition_buffer_invalid");
  TEST_ASSERT(!missing_false_count.ok);
  TEST_ASSERT(std::string_view{missing_false_count.reason} ==
              "compute_partition_buffer_invalid");
  TEST_ASSERT(!missing_true_count.ok);
  TEST_ASSERT(std::string_view{missing_true_count.reason} ==
              "compute_partition_buffer_invalid");
  return 0;
}

} // namespace

int RunPartitionReferenceContract() {
  if (test_compute_partition_cpu_reference_preserves_stable_groups() != 0) {
    return 1;
  }
  if (test_compute_partition_cpu_reference_rejects_zero_count() != 0) {
    return 1;
  }
  if (test_compute_partition_cpu_reference_handles_extreme_groups() != 0) {
    return 1;
  }
  return test_compute_partition_cpu_reference_rejects_missing_buffers();
}

} // namespace program_compute_contract
