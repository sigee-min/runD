#pragma once

#include "../descriptor.hpp"
#include "storage.hpp"
#include "update.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool
WriteVulkanStorageDescriptorSet(VulkanAdapter &adapter,
                                const VkDescriptorSet set,
                                const VulkanBuffer *const *const buffers,
                                const std::uint32_t descriptor_count) {
  if (set == VK_NULL_HANDLE || buffers == nullptr || descriptor_count == 0u) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
  VulkanDescriptorScratch scratch{};
  SelectVulkanDescriptorScratch(adapter, descriptor_count, scratch);

  for (std::uint32_t index = 0u; index < descriptor_count; ++index) {
    const VulkanBuffer *const buffer = buffers[index];
    const VkDescriptorBufferInfo info{
        buffer == nullptr ? VK_NULL_HANDLE : buffer->buffer,
        buffer == nullptr ? 0u : buffer->offset,
        buffer == nullptr ? 0u : buffer->bytes};
    if (!ValidStorage(adapter, info)) {
      SetVulkanLastError(adapter, "compute_resident_bytes_invalid");
      return false;
    }
    scratch.infos[index] = info;
  }

  FillVulkanDescriptorWrites(set, descriptor_count, scratch.infos,
                             scratch.writes);
  SubmitVulkanDescriptorWrites(adapter, descriptor_count, scratch.writes);
  return true;
}

[[nodiscard]] bool
WriteVulkanStorageDescriptorSet(VulkanAdapter &adapter,
                                const VkDescriptorSet set,
                                const VkDescriptorBufferInfo *const infos,
                                const std::uint32_t descriptor_count) {
  if (set == VK_NULL_HANDLE || infos == nullptr || descriptor_count == 0u) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
  for (std::uint32_t index = 0u; index < descriptor_count; ++index) {
    const VkDescriptorBufferInfo &info = infos[index];
    if (!ValidStorage(adapter, info)) {
      SetVulkanLastError(adapter, "compute_resident_bytes_invalid");
      return false;
    }
  }
  VulkanDescriptorScratch scratch{};
  SelectVulkanDescriptorScratch(adapter, descriptor_count, scratch);
  for (std::uint32_t index = 0u; index < descriptor_count; ++index) {
    scratch.infos[index] = infos[index];
  }
  FillVulkanDescriptorWrites(set, descriptor_count, scratch.infos,
                             scratch.writes);
  SubmitVulkanDescriptorWrites(adapter, descriptor_count, scratch.writes);
  return true;
}

[[nodiscard]] bool
WriteVulkanStorageDescriptorSet(VulkanAdapter &adapter,
                                const VkDescriptorSet set,
                                const VulkanStorageBinding *const bindings,
                                const std::uint32_t descriptor_count) {
  if (set == VK_NULL_HANDLE || bindings == nullptr || descriptor_count == 0u) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return false;
  }
  VulkanDescriptorScratch scratch{};
  SelectVulkanDescriptorScratch(adapter, descriptor_count, scratch);

  for (std::uint32_t index = 0u; index < descriptor_count; ++index) {
    const VulkanStorageBinding &binding = bindings[index];
    const VkDescriptorBufferInfo info{
        binding.buffer == nullptr ? VK_NULL_HANDLE : binding.buffer->buffer,
        binding.offset, binding.range};
    const VkDeviceSize local_offset =
        binding.buffer == nullptr || binding.offset < binding.buffer->offset
            ? 0u
            : binding.offset - binding.buffer->offset;
    if (binding.buffer == nullptr || !ValidStorage(adapter, info) ||
        binding.offset < binding.buffer->offset ||
        local_offset > binding.buffer->bytes ||
        binding.range > binding.buffer->bytes - local_offset) {
      SetVulkanLastError(adapter, "compute_resident_bytes_invalid");
      return false;
    }
    scratch.infos[index] = info;
  }

  FillVulkanDescriptorWrites(set, descriptor_count, scratch.infos,
                             scratch.writes);
  SubmitVulkanDescriptorWrites(adapter, descriptor_count, scratch.writes);
  return true;
}

#endif

} // namespace rund::node::accel::detail
