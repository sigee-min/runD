#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

#include <kernel/program/compute/scan/plan.hpp>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
PrepareMetalCompactScan(MetalAdapter &adapter,
                        MetalCompactEncodeResources &resources) {
  const rund::kernel::u64 scan_element_count =
      resources.block_offset_path ? resources.block_count
                                  : resources.plan.element_count;
  resources.scan_desc = rund::kernel::ScanDesc{
      .op = rund::kernel::ScanOp::ExclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = scan_element_count,
      .block_size = block::MetalCompact,
  };
  resources.scan_plan = rund::kernel::PlanScan(resources.scan_desc);
  if (!resources.scan_plan.ok) {
    SetMetalLastError(adapter, resources.scan_plan.reason);
    return rund::AccelCheck{false, resources.scan_plan.reason};
  }
  resources.scan_totals = AcquireMetalBuffer(
      adapter,
      resources.scan_plan.block_count * resources.scan_plan.element_bytes,
      MetalBufferUsage::Scratch);
  resources.scan_status = AcquireMetalBuffer(
      adapter, sizeof(rund::kernel::u32), MetalBufferUsage::Output);
  if (resources.plan.status_bytes != 0u) {
    resources.status = AcquireMetalBuffer(
        adapter, resources.plan.status_bytes, MetalBufferUsage::Output);
  }
  if (resources.scan_totals.buffer != nullptr &&
      resources.scan_status.buffer != nullptr &&
      (resources.plan.status_bytes == 0u ||
       resources.status.buffer != nullptr)) {
    return rund::AccelCheck{true, "ok"};
  }
  SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
  return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
