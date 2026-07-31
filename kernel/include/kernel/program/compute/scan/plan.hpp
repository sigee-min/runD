#pragma once

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/scan/model.hpp>

namespace rund::kernel {
namespace scan_plan_detail {

[[nodiscard]] constexpr bool KnownOp(const ScanOp op) noexcept {
  return op == ScanOp::ExclusiveSum || op == ScanOp::InclusiveSum;
}

[[nodiscard]] constexpr u64 ElementBytes(const ScanElement element) noexcept {
  switch (element) {
  case ScanElement::U32:
    return 4u;
  case ScanElement::U64:
    return 8u;
  }
  return 0u;
}

[[nodiscard]] constexpr ScanPlan
Reject(const ScanDesc &desc, const u64 element_bytes, const u64 block_count,
       const u64 temp_bytes, const char *const reason) noexcept {
  return ScanPlan{
      .op = desc.op,
      .element = desc.element,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .block_size = desc.block_size,
      .block_count = block_count,
      .temp_bytes = temp_bytes,
      .count_source = desc.count_source,
      .reason = reason,
  };
}

} // namespace scan_plan_detail

[[nodiscard]] constexpr ScanPlan PlanScan(const ScanDesc &desc) noexcept {
  if (!scan_plan_detail::KnownOp(desc.op)) {
    return scan_plan_detail::Reject(desc, 0u, 0u, 0u,
                                    "compute_scan_op_unsupported");
  }
  if (desc.count_source != ComputeCountSource::Descriptor &&
      ComputeCountBytes(desc.count_source) == 0u) {
    return scan_plan_detail::Reject(desc, 0u, 0u, 0u,
                                    "compute_scan_count_source_unsupported");
  }
  const u64 element_bytes = scan_plan_detail::ElementBytes(desc.element);
  if (element_bytes == 0u) {
    return scan_plan_detail::Reject(desc, 0u, 0u, 0u,
                                    "compute_scan_element_unsupported");
  }
  if (desc.element_count == 0u) {
    return scan_plan_detail::Reject(desc, element_bytes, 0u, 0u,
                                    "compute_scan_count_zero");
  }
  if (desc.block_size == 0u) {
    return scan_plan_detail::Reject(desc, element_bytes, 0u, 0u,
                                    "compute_scan_block_invalid");
  }

  const u64 block_count = checked::ceil(desc.element_count, desc.block_size);
  if (!checked::mul(desc.element_count, element_bytes)) {
    return scan_plan_detail::Reject(desc, element_bytes, block_count, 0u,
                                    "compute_scan_temp_overflow");
  }
  const u64 temp_bytes = desc.element_count * element_bytes;
  return ScanPlan{
      .op = desc.op,
      .element = desc.element,
      .element_count = desc.element_count,
      .element_bytes = element_bytes,
      .block_size = desc.block_size,
      .block_count = block_count,
      .pass_count = block_count > 1u ? 2u : 1u,
      .temp_bytes = temp_bytes,
      .count_source = desc.count_source,
      .ok = true,
      .reason = "ok",
  };
}

[[nodiscard]] constexpr bool
ScanPlanMatchesDesc(const ScanDesc &desc, const ScanPlan &plan) noexcept {
  const ScanPlan expected = PlanScan(desc);
  return expected.ok && plan.ok && plan.op == expected.op &&
         plan.element == expected.element &&
         plan.element_count == expected.element_count &&
         plan.element_bytes == expected.element_bytes &&
         plan.block_size == expected.block_size &&
         plan.block_count == expected.block_count &&
         plan.pass_count == expected.pass_count &&
         plan.temp_bytes == expected.temp_bytes &&
         plan.count_source == expected.count_source;
}

} // namespace rund::kernel
