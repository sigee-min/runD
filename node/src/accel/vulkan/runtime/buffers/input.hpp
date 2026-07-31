#pragma once

#include <rund/counter.hpp>
#include "admit.hpp"

#include <cstddef>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline bool PrepareVulkanStagedInputBuffer(
    VulkanAdapter &adapter, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow &window,
    const rund::kernel::BindingSet &bindings, VulkanWindowBuffers &out) {
  if (out.resident || plan.input_buffer_count == 0u) {
    return true;
  }
  rund::kernel::u64 input_byte_count = 0u;
  if (!StagedInputByteCount(
          bindings, window,
          static_cast<rund::kernel::u64>(adapter.storage_align),
          input_byte_count)) {
    SetVulkanLastError(adapter, "compute_binding_input_stride_invalid");
    return false;
  }
  std::size_t input_size_bytes = 0u;
  if (!ToSize(input_byte_count, input_size_bytes)) {
    SetVulkanLastError(adapter, "compute_dispatch_overflow");
    return false;
  }
  VulkanBuffer buffer{};
  if (!CreateVulkanBuffer(adapter, static_cast<VkDeviceSize>(input_size_bytes),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer)) {
    return false;
  }
  ScopedBuffer input_buffer{adapter, buffer,
                            static_cast<VkDeviceSize>(input_size_bytes)};
  auto *const input_data = static_cast<std::byte *>(input_buffer.buffer.mapped);
  if (input_data == nullptr) {
    SetVulkanLastError(adapter, "accel_vulkan_memory_unavailable");
    return false;
  }
  rund::kernel::u64 input_cursor = 0u;
  rund::kernel::u64 semantic_input_bytes = 0u;
  for (rund::kernel::u64 index = 0u; index < plan.input_buffer_count; ++index) {
    rund::kernel::u64 input_offset = 0u;
    rund::kernel::u64 input_range = 0u;
    rund::kernel::u64 next_cursor = 0u;
    if (!StagedInputRange(bindings.input_buffers[index], window, input_cursor,
                          static_cast<rund::kernel::u64>(adapter.storage_align),
                          input_offset, input_range, next_cursor)) {
      SetVulkanLastError(adapter, "compute_binding_input_stride_invalid");
      return false;
    }
    std::size_t offset_size = 0u;
    std::size_t range_size = 0u;
    if (!ToSize(input_offset, offset_size) ||
        !ToSize(input_range, range_size) || offset_size > input_size_bytes ||
        range_size > input_size_bytes - offset_size ||
        !PackInputBufferRange(bindings.input_buffers[index], bindings, window,
                              input_data + offset_size, range_size,
                              out.staged.bulk())) {
      SetVulkanLastError(adapter, "compute_binding_input_stride_invalid");
      return false;
    }
    if (!rund::kernel::checked::add(semantic_input_bytes, input_range)) {
      SetVulkanLastError(adapter, "compute_dispatch_overflow");
      return false;
    }
    semantic_input_bytes += input_range;
    input_cursor = next_cursor;
  }
  ::rund::detail::counter::Accumulate(adapter.host_to_device_bytes,
                                      semantic_input_bytes);
  out.input = std::move(input_buffer);
  return true;
}
#endif

} // namespace rund::node::accel::detail
