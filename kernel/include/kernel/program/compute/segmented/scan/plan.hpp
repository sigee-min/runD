#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/segmented/scan/model.hpp>

namespace rund::kernel {
namespace segmented_scan_plan_detail {

[[nodiscard]] constexpr bool KnownOp(const SegmentedScanOp op) noexcept {
  return op == SegmentedScanOp::ExclusiveSum ||
         op == SegmentedScanOp::InclusiveSum;
}

[[nodiscard]] constexpr u64
ElementBytes(const SegmentedScanElement element) noexcept {
  switch (element) {
  case SegmentedScanElement::U32:
    return 4u;
  case SegmentedScanElement::U64:
    return 8u;
  }
  return 0u;
}

[[nodiscard]] constexpr SegmentedScanPlan
Reject(const SegmentedScanDesc &desc, const u64 element_bytes,
       const u64 head_bytes, const u64 block_count, const u64 temp_value_bytes,
       const u64 temp_head_bytes, const char *const reason) noexcept {
  return SegmentedScanPlan{
      .op = desc.op,
      .element = desc.element,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .head_bytes = head_bytes,
      .block_size = desc.block_size,
      .block_count = block_count,
      .temp_value_bytes = temp_value_bytes,
      .temp_head_bytes = temp_head_bytes,
      .reason = reason,
  };
}

} // namespace segmented_scan_plan_detail

[[nodiscard]] constexpr SegmentedScanPlan
PlanSegmentedScan(const SegmentedScanDesc &desc) noexcept {
  constexpr u64 head_bytes = 4u;
  if (!segmented_scan_plan_detail::KnownOp(desc.op)) {
    return segmented_scan_plan_detail::Reject(
        desc, 0u, 0u, 0u, 0u, 0u, "compute_segmented_scan_op_unsupported");
  }
  const u64 element_bytes =
      segmented_scan_plan_detail::ElementBytes(desc.element);
  if (element_bytes == 0u) {
    return segmented_scan_plan_detail::Reject(
        desc, 0u, head_bytes, 0u, 0u, 0u,
        "compute_segmented_scan_element_unsupported");
  }
  if (desc.element_count == 0u) {
    return segmented_scan_plan_detail::Reject(
        desc, element_bytes, head_bytes, 0u, 0u, 0u,
        "compute_segmented_scan_count_zero");
  }
  if (desc.block_size == 0u) {
    return segmented_scan_plan_detail::Reject(
        desc, element_bytes, head_bytes, 0u, 0u, 0u,
        "compute_segmented_scan_block_invalid");
  }

  const u64 block_count = checked::ceil(desc.element_count, desc.block_size);
  if (!checked::mul(desc.element_count, element_bytes) ||
      !checked::mul(desc.element_count, head_bytes)) {
    return segmented_scan_plan_detail::Reject(
        desc, element_bytes, head_bytes, block_count, 0u, 0u,
        "compute_segmented_scan_temp_overflow");
  }
  const u64 temp_value_bytes = desc.element_count * element_bytes;
  const u64 temp_head_bytes = desc.element_count * head_bytes;
  if (!checked::add(temp_value_bytes, temp_head_bytes)) {
    return segmented_scan_plan_detail::Reject(
        desc, element_bytes, head_bytes, block_count, temp_value_bytes,
        temp_head_bytes, "compute_segmented_scan_temp_overflow");
  }
  return SegmentedScanPlan{
      .op = desc.op,
      .element = desc.element,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .head_bytes = head_bytes,
      .block_size = desc.block_size,
      .block_count = block_count,
      .pass_count = block_count > 1u ? 2u : 1u,
      .temp_value_bytes = temp_value_bytes,
      .temp_head_bytes = temp_head_bytes,
      .temp_bytes = temp_value_bytes + temp_head_bytes,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
SegmentedScanPlanMatchesDesc(const SegmentedScanDesc &desc,
                             const SegmentedScanPlan &plan) noexcept {
  const SegmentedScanPlan expected = PlanSegmentedScan(desc);
  return expected.ok && plan.ok && plan.op == expected.op &&
         plan.element == expected.element &&
         plan.element_count == expected.element_count &&
         plan.element_bytes == expected.element_bytes &&
         plan.head_bytes == expected.head_bytes &&
         plan.block_size == expected.block_size &&
         plan.block_count == expected.block_count &&
         plan.pass_count == expected.pass_count &&
         plan.temp_value_bytes == expected.temp_value_bytes &&
         plan.temp_head_bytes == expected.temp_head_bytes &&
         plan.temp_bytes == expected.temp_bytes;
}

} // namespace rund::kernel
