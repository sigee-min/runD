#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

#include <kernel/program/compute/scan/plan.hpp>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
PlanMetalPartitionScan(MetalAdapter &adapter,
                       const rund::kernel::PartitionPlan &plan,
                       MetalPartitionEncodeResources &raw) {
  raw.scan_desc = rund::kernel::ScanDesc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = plan.element_count,
      .block_size = block::MetalPartition,
  };
  raw.scan_plan = rund::kernel::PlanScan(raw.scan_desc);
  if (!raw.scan_plan.ok || !ScanShapeOk(raw.scan_desc, raw.scan_plan)) {
    SetMetalLastError(adapter, "compute_partition_invalid");
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
