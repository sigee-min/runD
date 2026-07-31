#pragma once

#include <accel/check.hpp>

#include "lookup.hpp"

#include <cstring>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck PrepareMetalSegmentedScanStatusBuffer(
    MetalAdapter &adapter, MetalSegmentedScanEncodeResources &resources) {
  const rund::kernel::u64 value_bytes =
      resources.plan.block_count * resources.plan.element_bytes;
  const rund::kernel::u64 head_bytes =
      resources.plan.block_count * sizeof(rund::kernel::u32);
  resources.offsets =
      AcquireMetalBuffer(adapter, value_bytes, MetalBufferUsage::Scratch);
  resources.first_heads =
      AcquireMetalBuffer(adapter, head_bytes, MetalBufferUsage::Scratch);
  resources.status =
      AcquireMetalBuffer(adapter, head_bytes, MetalBufferUsage::Output);
  if (resources.offsets.buffer == nullptr ||
      resources.first_heads.buffer == nullptr ||
      resources.status.buffer == nullptr) {
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  void *const contents = MetalBufferContents(resources.status);
  if (contents == nullptr) {
    return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
  }
  std::memset(contents, 0, static_cast<std::size_t>(head_bytes));
  return rund::AccelCheck{true, "ok"};
}
#endif

} // namespace rund::node::accel::detail
