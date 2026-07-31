#pragma once

#include <accel/check.hpp>

#include "lookup.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
AcquireMetalPartitionBuffers(MetalAdapter &adapter,
                             const rund::kernel::PartitionPlan &plan,
                             MetalPartitionEncodeResources &raw) {
  const rund::kernel::u64 element_bytes =
      plan.element_count * sizeof(rund::kernel::u32);
  const rund::kernel::u64 totals_bytes =
      raw.scan_plan.block_count * sizeof(rund::kernel::u32);
  raw.false_bits =
      AcquireMetalBuffer(adapter, element_bytes, MetalBufferUsage::Scratch);
  raw.false_offsets =
      AcquireMetalBuffer(adapter, element_bytes, MetalBufferUsage::Scratch);
  raw.false_totals =
      AcquireMetalBuffer(adapter, totals_bytes, MetalBufferUsage::Scratch);
  raw.false_status = AcquireMetalBuffer(
      adapter, sizeof(rund::kernel::u32), MetalBufferUsage::Output);
  if (!MetalPartitionBuffersReady(raw)) {
    SetMetalLastError(adapter, "accel_metal_pipeline_unavailable");
    return rund::AccelCheck{false, "accel_metal_pipeline_unavailable"};
  }
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
