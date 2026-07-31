#pragma once

#include <kernel/program/compute/partition/model.hpp>
#include <kernel/program/compute/partition/identity.hpp>
#include <kernel/program/compute/partition/plan.hpp>
#include <kernel/program/compute/partition/reference.hpp>

#include <string_view>

namespace program_compute_contract::partition_contract {

[[nodiscard]] constexpr rund::kernel::PartitionDesc U32Partition() noexcept {
  return rund::kernel::PartitionDesc{
      .element_count = 8u,
      .flag_bytes = 4u,
      .value_bytes = 4u,
  };
}

[[nodiscard]] inline bool
SamePlan(const rund::kernel::PartitionPlan &lhs,
         const rund::kernel::PartitionPlan &rhs) noexcept {
  return lhs.element_count == rhs.element_count &&
         lhs.flag_bytes == rhs.flag_bytes &&
         lhs.value_bytes == rhs.value_bytes &&
         lhs.scan_temp_bytes == rhs.scan_temp_bytes &&
         lhs.temp_bytes == rhs.temp_bytes && lhs.pass_count == rhs.pass_count &&
         lhs.ok == rhs.ok &&
         std::string_view{lhs.reason} == std::string_view{rhs.reason};
}

} // namespace program_compute_contract::partition_contract
