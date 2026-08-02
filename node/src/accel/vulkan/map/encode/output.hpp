#pragma once

#include "../local.hpp"

#include <array>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool
BindVulkanMapOutputs(VulkanAdapter &adapter,
                     const VulkanMapEncodeResources &map,
                     const rund::kernel::ComputeDispatchWindow &window,
                     VkDescriptorBufferInfo *const infos) {
  if (map.prepared == nullptr) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return false;
  }
  for (rund::kernel::u64 index = 0u;
       index < map.prepared->plan.output_buffer_count;
       ++index) {
    VkDeviceSize output_offset = 0u;
    VkDeviceSize output_range = 0u;
    const char *reason = "ok";
    const rund::kernel::ResidentBufferRef *const ref =
        map.bindings.resident_outputs.ref(index);
    if (ref == nullptr ||
        !VulkanMapResidentOutputWindowSpan(map, *ref, window, output_offset,
                                           output_range, reason)) {
      SetVulkanLastError(adapter, reason);
      return false;
    }
    const VulkanResidentBufferResult &output = map.resident.output(index);
    if (output.device_buffer == nullptr) {
      SetVulkanLastError(adapter, "compute_binding_mismatch");
      return false;
    }
    infos[static_cast<std::size_t>(
        map.prepared->plan.input_buffer_count + index + 1u)] =
        VkDescriptorBufferInfo{output.device_buffer->buffer, output_offset,
                               output_range};
  }
  return true;
}

[[nodiscard]] bool EncodeVulkanMapOutputBarrier(
    VkCommandBuffer command, const VulkanMapEncodeResources &map,
    const rund::kernel::ComputeDispatchWindow &window) {
  std::array<VkBufferMemoryBarrier, rund::kernel::kMaxGraphSignatureValues>
      barriers{};
  if (map.prepared == nullptr ||
      map.prepared->plan.output_buffer_count == 0u ||
      map.prepared->plan.output_buffer_count > barriers.size()) {
    SetVulkanLastError(*map.adapter, "compute_binding_mismatch");
    return false;
  }
  for (rund::kernel::u64 index = 0u;
       index < map.prepared->plan.output_buffer_count;
       ++index) {
    VkDeviceSize offset = 0u;
    VkDeviceSize range = 0u;
    const char *reason = "ok";
    const rund::kernel::ResidentBufferRef *const ref =
        map.bindings.resident_outputs.ref(index);
    if (ref == nullptr ||
        !VulkanMapResidentOutputWindowSpan(map, *ref, window, offset, range,
                                           reason) ||
        map.resident.output(index).device_buffer == nullptr) {
      SetVulkanLastError(*map.adapter, reason);
      return false;
    }
    barriers[static_cast<std::size_t>(index)] = VkBufferMemoryBarrier{
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = map.resident.output(index).device_buffer->buffer,
        .offset = offset,
        .size = range,
    };
  }
  vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0u, 0u, nullptr,
                       static_cast<std::uint32_t>(
                           map.prepared->plan.output_buffer_count),
                       barriers.data(), 0u, nullptr);
  return true;
}

} // namespace
#endif

} // namespace rund::node::accel::detail
