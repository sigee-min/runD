#pragma once

#include <accel/check.hpp>

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
namespace {

[[nodiscard]] bool
MetalSortBuffersOk(const MetalSortEncodeResources &resources) {
  if (resources.temp_keys.buffer == nullptr ||
      resources.temp_values.buffer == nullptr ||
      resources.block_counts.buffer == nullptr ||
      resources.block_offsets.buffer == nullptr ||
      resources.bucket_offsets.buffer == nullptr ||
      resources.dispatch_args.buffer == nullptr ||
      resources.status.buffer == nullptr) {
    return false;
  }
  return true;
}

[[nodiscard]] rund::AccelCheck
AcquireMetalSortBuffers(MetalAdapter &adapter,
                        const rund::kernel::u64 block_table_bytes,
                        MetalSortEncodeResources &resources) {
  resources.temp_keys = AcquireMetalBuffer(
      adapter, resources.plan.temp_key_bytes, MetalBufferUsage::Scratch);
  resources.temp_values = AcquireMetalBuffer(
      adapter, resources.plan.temp_value_bytes, MetalBufferUsage::Scratch);
  resources.block_counts =
      AcquireMetalBuffer(adapter, block_table_bytes, MetalBufferUsage::Scratch);
  resources.block_offsets =
      AcquireMetalBuffer(adapter, block_table_bytes, MetalBufferUsage::Scratch);
  const rund::kernel::u64 bucket_bytes =
      resources.plan.bucket_count * sizeof(rund::kernel::u32);
  resources.bucket_offsets =
      AcquireMetalBuffer(adapter, bucket_bytes, MetalBufferUsage::Scratch);
  resources.dispatch_args = AcquireMetalBuffer(
      adapter, 3u * sizeof(rund::kernel::u32), MetalBufferUsage::Output);
  resources.status = AcquireMetalBuffer(
      adapter, sizeof(rund::kernel::u32), MetalBufferUsage::Output);
  if (MetalSortBuffersOk(resources)) {
    return rund::AccelCheck{true, "ok"};
  }
  SetMetalLastError(adapter, "accel_metal_buffer_unavailable");
  return rund::AccelCheck{false, "accel_metal_buffer_unavailable"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
