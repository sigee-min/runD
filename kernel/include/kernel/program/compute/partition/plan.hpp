#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/partition/model.hpp>

namespace rund::kernel {
namespace partition_plan_detail {

[[nodiscard]] constexpr PartitionPlan
Reject(const PartitionDesc &desc, const u64 flag_bytes, const u64 value_bytes,
       const u64 scan_temp_bytes, const u64 temp_bytes,
       const char *const reason) noexcept {
  return PartitionPlan{
      .element_count = desc.element_count,
      .flag_bytes = flag_bytes,
      .value_bytes = value_bytes,
      .scan_temp_bytes = scan_temp_bytes,
      .temp_bytes = temp_bytes,
      .reason = reason,
  };
}

} // namespace partition_plan_detail

[[nodiscard]] constexpr PartitionPlan
PlanPartition(const PartitionDesc &desc) noexcept {
  if ((desc.flag_bytes != 4u && desc.flag_bytes != 8u) ||
      (desc.value_bytes != 4u && desc.value_bytes != 8u)) {
    return partition_plan_detail::Reject(desc, desc.flag_bytes,
                                         desc.value_bytes, 0u, 0u,
                                         "compute_partition_invalid");
  }
  if (desc.element_count == 0u) {
    return partition_plan_detail::Reject(desc, desc.flag_bytes,
                                         desc.value_bytes, 0u, 0u,
                                         "compute_partition_count_zero");
  }

  constexpr u64 kOffsetBytes = sizeof(u32);
  if (!checked::mul(desc.element_count, kOffsetBytes)) {
    return partition_plan_detail::Reject(desc, desc.flag_bytes,
                                         desc.value_bytes, 0u, 0u,
                                         "compute_partition_temp_overflow");
  }
  const u64 scan_temp_bytes = desc.element_count * kOffsetBytes;

  return PartitionPlan{
      .element_count = desc.element_count,
      .flag_bytes = desc.flag_bytes,
      .value_bytes = desc.value_bytes,
      .scan_temp_bytes = scan_temp_bytes,
      .temp_bytes = scan_temp_bytes,
      .pass_count = 3u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
PartitionPlanMatchesDesc(const PartitionDesc &desc,
                         const PartitionPlan &plan) noexcept {
  const PartitionPlan expected = PlanPartition(desc);
  return expected.ok && plan.ok &&
         plan.element_count == expected.element_count &&
         plan.flag_bytes == expected.flag_bytes &&
         plan.value_bytes == expected.value_bytes &&
         plan.scan_temp_bytes == expected.scan_temp_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.pass_count == expected.pass_count;
}

} // namespace rund::kernel
