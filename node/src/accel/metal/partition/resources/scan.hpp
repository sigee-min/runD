#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
PlanMetalPartitionScan(MetalAdapter &adapter,
                       const rund::kernel::PartitionPlan &plan,
                       MetalPartitionEncodeResources &raw) {
  raw.scan_desc = MetalPartitionScanDesc(plan);
  raw.scan_plan = MetalPartitionScanPlan(plan);
  const RangePrefixExec scan_execution = MetalPartitionScanExecution(plan);
  if (!raw.scan_plan.ok || !ScanShapeOk(raw.scan_desc, raw.scan_plan) ||
      !MetalScanPrefixMatchesPlan(raw.scan_plan, scan_execution)) {
    SetMetalLastError(adapter, "compute_partition_invalid");
    return rund::AccelCheck{false, "compute_partition_invalid"};
  }
  raw.scan_execution = scan_execution;
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
