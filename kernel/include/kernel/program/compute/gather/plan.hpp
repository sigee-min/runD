#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/gather/model.hpp>

namespace rund::kernel {
namespace gather_plan_detail {

inline constexpr u64 kIndexBytes = 4u;
// Failure atomicity is a two-pass contract. The preflight publishes a
// reason/first-ordinal pair; the second pass consumes one indirect dispatch
// record only after validation succeeds.
inline constexpr u64 kStatusBytes = 8u;
inline constexpr u64 kIndirectBytes = 16u;
inline constexpr u64 kMaxU32AddressableCount = static_cast<u64>(~u32{0u}) + 1u;

[[nodiscard]] constexpr u64 ElementBytes(const GatherElement element) noexcept {
  switch (element) {
  case GatherElement::U32:
    return 4u;
  case GatherElement::U64:
    return 8u;
  }
  return 0u;
}

[[nodiscard]] constexpr GatherPlan
Reject(const GatherDesc &desc, const u64 element_bytes, const u64 index_bytes,
       const u64 status_bytes, const u64 temp_bytes,
       const char *const reason) noexcept {
  return GatherPlan{
      .element = desc.element,
      .element_count = desc.element_count,
      .source_count = desc.source_count,
      .element_bytes = element_bytes,
      .index_bytes = index_bytes,
      .status_bytes = status_bytes,
      .temp_bytes = temp_bytes,
      .count_source = desc.count_source,
      .reason = reason,
  };
}

} // namespace gather_plan_detail

[[nodiscard]] constexpr GatherPlan PlanGather(const GatherDesc &desc) noexcept {
  const u64 element_bytes = gather_plan_detail::ElementBytes(desc.element);
  if (desc.count_source != ComputeCountSource::Descriptor &&
      ComputeCountBytes(desc.count_source) == 0u) {
    return gather_plan_detail::Reject(
        desc, 0u, gather_plan_detail::kIndexBytes,
        gather_plan_detail::kStatusBytes, 0u,
        "compute_gather_count_source_unsupported");
  }
  if (element_bytes == 0u) {
    return gather_plan_detail::Reject(desc, 0u, 0u, 0u, 0u,
                                      "compute_gather_element_unsupported");
  }
  if (desc.element_count == 0u) {
    return gather_plan_detail::Reject(
        desc, element_bytes, gather_plan_detail::kIndexBytes,
        gather_plan_detail::kStatusBytes, 0u, "compute_gather_count_zero");
  }
  if (desc.element_count > ~u32{0u}) {
    return gather_plan_detail::Reject(desc, element_bytes,
                                      gather_plan_detail::kIndexBytes,
                                      gather_plan_detail::kStatusBytes, 0u,
                                      "compute_gather_count_unsupported");
  }
  if (desc.source_count == 0u) {
    return gather_plan_detail::Reject(
        desc, element_bytes, gather_plan_detail::kIndexBytes,
        gather_plan_detail::kStatusBytes, 0u, "compute_gather_source_zero");
  }
  if (desc.source_count > gather_plan_detail::kMaxU32AddressableCount) {
    return gather_plan_detail::Reject(desc, element_bytes,
                                      gather_plan_detail::kIndexBytes,
                                      gather_plan_detail::kStatusBytes, 0u,
                                      "compute_gather_source_unsupported");
  }
  if (!checked::mul(desc.element_count, element_bytes) ||
      !checked::mul(desc.element_count, gather_plan_detail::kIndexBytes)) {
    return gather_plan_detail::Reject(
        desc, element_bytes, gather_plan_detail::kIndexBytes,
        gather_plan_detail::kStatusBytes, 0u, "compute_gather_temp_overflow");
  }

  return GatherPlan{
      .element = desc.element,
      .element_count = desc.element_count,
      .source_count = desc.source_count,
      .element_bytes = element_bytes,
      .index_bytes = gather_plan_detail::kIndexBytes,
      .status_bytes = gather_plan_detail::kStatusBytes,
      .temp_bytes =
          gather_plan_detail::kStatusBytes + gather_plan_detail::kIndirectBytes,
      .pass_count = 2u,
      .count_source = desc.count_source,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
GatherPlanMatchesDesc(const GatherDesc &desc, const GatherPlan &plan) noexcept {
  const GatherPlan expected = PlanGather(desc);
  return expected.ok && plan.ok && plan.element == expected.element &&
         plan.element_count == expected.element_count &&
         plan.source_count == expected.source_count &&
         plan.element_bytes == expected.element_bytes &&
         plan.index_bytes == expected.index_bytes &&
         plan.status_bytes == expected.status_bytes &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.pass_count == expected.pass_count &&
         plan.count_source == expected.count_source;
}

} // namespace rund::kernel
