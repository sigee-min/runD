#include "test/assert.hpp"

#include "local.hpp"

#include <limits>
#include <string_view>

namespace program_compute_contract {
namespace {

int test_compute_partition_plan_rejects_zero_count() {
  rund::kernel::PartitionDesc desc = partition_contract::U32Partition();
  desc.element_count = 0u;

  const rund::kernel::PartitionPlan plan = rund::kernel::PlanPartition(desc);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "compute_partition_count_zero");
  return 0;
}

int test_compute_partition_plan_rejects_invalid_widths() {
  rund::kernel::PartitionDesc bad_flag = partition_contract::U32Partition();
  bad_flag.flag_bytes = 2u;
  rund::kernel::PartitionDesc bad_value = partition_contract::U32Partition();
  bad_value.value_bytes = 2u;

  const rund::kernel::PartitionPlan flag_plan =
      rund::kernel::PlanPartition(bad_flag);
  const rund::kernel::PartitionPlan value_plan =
      rund::kernel::PlanPartition(bad_value);

  TEST_ASSERT(!flag_plan.ok);
  TEST_ASSERT(std::string_view{flag_plan.reason} ==
              "compute_partition_invalid");
  TEST_ASSERT(!value_plan.ok);
  TEST_ASSERT(std::string_view{value_plan.reason} ==
              "compute_partition_invalid");
  return 0;
}

int test_compute_partition_plan_accepts_u64_flags() {
  rund::kernel::PartitionDesc desc = partition_contract::U32Partition();
  desc.flag_bytes = 8u;
  const rund::kernel::PartitionPlan plan = rund::kernel::PlanPartition(desc);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.flag_bytes == 8u);
  TEST_ASSERT(plan.value_bytes == 4u);
  TEST_ASSERT(plan.pass_count == 3u);
  return 0;
}

int test_compute_partition_plan_computes_u64_shape() {
  rund::kernel::PartitionDesc desc = partition_contract::U32Partition();
  desc.value_bytes = 8u;
  const rund::kernel::PartitionPlan plan = rund::kernel::PlanPartition(desc);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.flag_bytes == 4u);
  TEST_ASSERT(plan.value_bytes == 8u);
  TEST_ASSERT(plan.element_count == 8u);
  TEST_ASSERT(plan.pass_count == 3u);
  return 0;
}

int test_compute_partition_plan_rejects_temp_byte_overflow() {
  rund::kernel::PartitionDesc desc = partition_contract::U32Partition();
  desc.element_count = (std::numeric_limits<rund::kernel::u64>::max() /
                        sizeof(rund::kernel::u32)) +
                       1u;

  const rund::kernel::PartitionPlan plan = rund::kernel::PlanPartition(desc);

  TEST_ASSERT(!plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} ==
              "compute_partition_temp_overflow");
  return 0;
}

int test_compute_partition_plan_accepts_largest_temp_extent() {
  rund::kernel::PartitionDesc desc = partition_contract::U32Partition();
  desc.element_count =
      std::numeric_limits<rund::kernel::u64>::max() / sizeof(rund::kernel::u32);

  const rund::kernel::PartitionPlan plan = rund::kernel::PlanPartition(desc);

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(plan.scan_temp_bytes ==
              desc.element_count * sizeof(rund::kernel::u32));
  TEST_ASSERT(plan.temp_bytes == plan.scan_temp_bytes);
  return 0;
}

int test_compute_partition_plan_is_deterministic() {
  const rund::kernel::PartitionPlan first =
      rund::kernel::PlanPartition(partition_contract::U32Partition());
  const rund::kernel::PartitionPlan second =
      rund::kernel::PlanPartition(partition_contract::U32Partition());

  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(partition_contract::SamePlan(first, second));
  return 0;
}

int test_compute_partition_plan_computes_u32_shape() {
  const rund::kernel::PartitionPlan plan =
      rund::kernel::PlanPartition(partition_contract::U32Partition());

  TEST_ASSERT(plan.ok);
  TEST_ASSERT(std::string_view{plan.reason} == "ok");
  TEST_ASSERT(plan.element_count == 8u);
  TEST_ASSERT(plan.flag_bytes == 4u);
  TEST_ASSERT(plan.value_bytes == 4u);
  TEST_ASSERT(plan.scan_temp_bytes == 32u);
  TEST_ASSERT(plan.temp_bytes == 32u);
  TEST_ASSERT(plan.pass_count == 3u);
  TEST_ASSERT(rund::kernel::PartitionPlanMatchesDesc(
      partition_contract::U32Partition(), plan));
  rund::kernel::PartitionPlan forged = plan;
  ++forged.scan_temp_bytes;
  TEST_ASSERT(!rund::kernel::PartitionPlanMatchesDesc(
      partition_contract::U32Partition(), forged));
  return 0;
}

} // namespace

int RunPartitionPlanContract() {
  if (test_compute_partition_plan_rejects_zero_count() != 0) {
    return 1;
  }
  if (test_compute_partition_plan_rejects_invalid_widths() != 0) {
    return 1;
  }
  if (test_compute_partition_plan_accepts_u64_flags() != 0) {
    return 1;
  }
  if (test_compute_partition_plan_rejects_temp_byte_overflow() != 0) {
    return 1;
  }
  if (test_compute_partition_plan_accepts_largest_temp_extent() != 0) {
    return 1;
  }
  if (test_compute_partition_plan_is_deterministic() != 0) {
    return 1;
  }
  if (test_compute_partition_plan_computes_u32_shape() != 0) {
    return 1;
  }
  return test_compute_partition_plan_computes_u64_shape();
}

} // namespace program_compute_contract
