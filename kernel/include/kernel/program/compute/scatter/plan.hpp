#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scatter/model.hpp>

namespace rund::kernel {
namespace scatter_plan_detail {

inline constexpr u64 kIndexBytes = 4u;
inline constexpr u64 kStatusElementBytes = 4u;
inline constexpr u64 kMaxOutputCount = static_cast<u64>(~u32{0u});
inline constexpr u64 kMaxEncodedCount = kMaxOutputCount >> 1u;

[[nodiscard]] constexpr u64
ElementBytes(const ScatterElement element) noexcept {
  switch (element) {
  case ScatterElement::U32:
    return 4u;
  case ScatterElement::U64:
    return 8u;
  }
  return 0u;
}

[[nodiscard]] constexpr u64 ScratchSlots(const u64 count) noexcept {
  if (count == 0u || !checked::mul(count, 2u)) {
    return 0u;
  }
  const u64 required = count * 2u;
  u64 slots = 1u;
  while (slots < required) {
    if (!checked::mul(slots, 2u)) {
      return 0u;
    }
    slots *= 2u;
  }
  return slots;
}

[[nodiscard]] constexpr ScatterPlan
Reject(const ScatterDesc &desc, const u64 element_bytes, const u64 index_bytes,
       const u64 status_bytes, const u64 temp_bytes,
       const char *const reason) noexcept {
  return ScatterPlan{
      .element = desc.element,
      .element_count = desc.element_count,
      .output_count = desc.output_count,
      .element_bytes = element_bytes,
      .index_bytes = index_bytes,
      .status_bytes = status_bytes,
      .temp_bytes = temp_bytes,
      .reason = reason,
  };
}

} // namespace scatter_plan_detail

[[nodiscard]] constexpr ScatterPlan
PlanScatter(const ScatterDesc &desc) noexcept {
  const u64 element_bytes = scatter_plan_detail::ElementBytes(desc.element);
  if (element_bytes == 0u) {
    return scatter_plan_detail::Reject(desc, 0u, 0u, 0u, 0u,
                                       "compute_scatter_element_unsupported");
  }
  if (desc.element_count == 0u) {
    return scatter_plan_detail::Reject(desc, element_bytes,
                                       scatter_plan_detail::kIndexBytes, 0u, 0u,
                                       "compute_scatter_count_zero");
  }
  if (desc.element_count > scatter_plan_detail::kMaxEncodedCount) {
    return scatter_plan_detail::Reject(desc, element_bytes,
                                       scatter_plan_detail::kIndexBytes, 0u, 0u,
                                       "compute_scatter_count_unsupported");
  }
  if (desc.output_count == 0u) {
    return scatter_plan_detail::Reject(desc, element_bytes,
                                       scatter_plan_detail::kIndexBytes, 0u, 0u,
                                       "compute_scatter_output_zero");
  }
  if (desc.output_count > scatter_plan_detail::kMaxOutputCount) {
    return scatter_plan_detail::Reject(desc, element_bytes,
                                       scatter_plan_detail::kIndexBytes, 0u, 0u,
                                       "compute_scatter_output_unsupported");
  }
  const u64 scratch_slots =
      scatter_plan_detail::ScratchSlots(desc.element_count);
  const u64 status_bytes =
      (desc.output_count + 1u) * scatter_plan_detail::kStatusElementBytes;

  return ScatterPlan{
      .element = desc.element,
      .element_count = desc.element_count,
      .output_count = desc.output_count,
      .element_bytes = element_bytes,
      .index_bytes = scatter_plan_detail::kIndexBytes,
      .status_bytes = status_bytes,
      .temp_bytes = status_bytes,
      .scratch_slots = scratch_slots,
      .pass_count = 1u,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
ScatterPlanMatchesDesc(const ScatterDesc &desc,
                       const ScatterPlan &plan) noexcept {
  const ScatterPlan expected = PlanScatter(desc);
  return expected.ok && plan.ok && plan.element == expected.element &&
         plan.element_count == expected.element_count &&
         plan.output_count == expected.output_count &&
         plan.element_bytes == expected.element_bytes &&
         plan.index_bytes == expected.index_bytes &&
         plan.status_bytes == expected.status_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.scratch_slots == expected.scratch_slots &&
         plan.pass_count == expected.pass_count;
}

} // namespace rund::kernel
