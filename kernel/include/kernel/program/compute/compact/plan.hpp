#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/compact/model.hpp>

namespace rund::kernel {
namespace compact_plan_detail {

[[nodiscard]] constexpr CompactPlan
Reject(const CompactDesc &desc, const u64 flag_bytes, const u64 output_bytes,
       const u64 scan_temp_bytes, const u64 status_bytes, const u64 temp_bytes,
       const char *const reason) noexcept {
  return CompactPlan{
      .element_count = desc.element_count,
      .output_capacity = desc.output_capacity,
      .flag_bytes = flag_bytes,
      .output_bytes = output_bytes,
      .scan_temp_bytes = scan_temp_bytes,
      .status_bytes = status_bytes,
      .temp_bytes = temp_bytes,
      .reason = reason,
  };
}

} // namespace compact_plan_detail

[[nodiscard]] constexpr CompactPlan
PlanCompact(const CompactDesc &desc) noexcept {
  if (desc.flag_bytes != 4u || desc.output_bytes != 4u) {
    return compact_plan_detail::Reject(desc, desc.flag_bytes, desc.output_bytes,
                                       0u, 0u, 0u, "compute_compact_invalid");
  }
  if (desc.element_count == 0u) {
    return compact_plan_detail::Reject(desc, desc.flag_bytes, desc.output_bytes,
                                       0u, 0u, 0u,
                                       "compute_compact_count_zero");
  }
  if (desc.output_capacity == 0u) {
    return compact_plan_detail::Reject(desc, desc.flag_bytes, desc.output_bytes,
                                       0u, 0u, 0u,
                                       "compute_compact_capacity_zero");
  }

  const u64 status_bytes = desc.output_capacity < desc.element_count ? 4u : 0u;
  if (!checked::mul(desc.element_count, desc.flag_bytes)) {
    return compact_plan_detail::Reject(desc, desc.flag_bytes, desc.output_bytes,
                                       0u, status_bytes, 0u,
                                       "compute_compact_temp_overflow");
  }
  const u64 scan_temp_bytes =
      desc.element_count * static_cast<u64>(desc.flag_bytes);
  if (!checked::add(scan_temp_bytes, status_bytes)) {
    return compact_plan_detail::Reject(desc, desc.flag_bytes, desc.output_bytes,
                                       scan_temp_bytes, status_bytes, 0u,
                                       "compute_compact_temp_overflow");
  }

  return CompactPlan{
      .element_count = desc.element_count,
      .output_capacity = desc.output_capacity,
      .flag_bytes = desc.flag_bytes,
      .output_bytes = desc.output_bytes,
      .scan_temp_bytes = scan_temp_bytes,
      .status_bytes = status_bytes,
      .temp_bytes = scan_temp_bytes + status_bytes,
      .pass_count = 2u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
CompactPlanMatchesDesc(const CompactDesc &desc,
                       const CompactPlan &plan) noexcept {
  const CompactPlan expected = PlanCompact(desc);
  return expected.ok && plan.ok &&
         plan.element_count == expected.element_count &&
         plan.output_capacity == expected.output_capacity &&
         plan.flag_bytes == expected.flag_bytes &&
         plan.output_bytes == expected.output_bytes &&
         plan.scan_temp_bytes == expected.scan_temp_bytes &&
         plan.status_bytes == expected.status_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.pass_count == expected.pass_count;
}

} // namespace rund::kernel
