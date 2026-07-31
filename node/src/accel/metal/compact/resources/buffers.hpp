#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
AcquireMetalCompactPathBuffers(MetalAdapter &adapter,
                               MetalCompactEncodeResources &resources) {
  const rund::kernel::CompactPlan &plan = resources.plan;
  resources.block_offset_path = plan.status_bytes == 0u;
  resources.block_count =
      (plan.element_count + block::MetalCompact - 1u) / block::MetalCompact;
  const rund::kernel::u64 element_scratch_bytes =
      plan.element_count * sizeof(rund::kernel::u32);
  if (resources.block_offset_path) {
    const rund::kernel::u64 block_scratch_bytes =
        resources.block_count * sizeof(rund::kernel::u32);
    resources.block_counts = AcquireMetalBuffer(adapter, block_scratch_bytes,
                                                MetalBufferUsage::Scratch);
    resources.block_offsets = AcquireMetalBuffer(adapter, block_scratch_bytes,
                                                 MetalBufferUsage::Scratch);
    resources.flag_bits = AcquireMetalBuffer(
        adapter, resources.block_count * 32u * sizeof(rund::kernel::u32),
        MetalBufferUsage::Scratch);
    if (resources.block_counts.buffer != nullptr &&
        resources.block_offsets.buffer != nullptr &&
        resources.flag_bits.buffer != nullptr) {
      return rund::AccelCheck{true, "ok"};
    }
  } else {
    resources.offsets = AcquireMetalBuffer(adapter, element_scratch_bytes,
                                           MetalBufferUsage::Scratch);
    if (resources.offsets.buffer != nullptr) {
      return rund::AccelCheck{true, "ok"};
    }
  }
  SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
  return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
