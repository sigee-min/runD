#pragma once

#include <accel/check.hpp>

#include "../../descriptor.hpp"
#include "../../descriptor/storage.hpp"
#include "../../descriptor/update.hpp"
#include "../control.hpp"
#include "../dispatch.hpp"
#include "input.hpp"
#include "output.hpp"

#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck EncodeVulkanMapWindow(
    VulkanAdapter &adapter, const VulkanMapEncodeResources &map,
    const rund::kernel::ComputeDispatchWindow &window,
    const VkDescriptorSet descriptor_set, const std::uint32_t descriptor_count,
    const std::uint32_t window_index, VkCommandBuffer command_buffer) {
  if (descriptor_set == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_descriptor_unavailable"};
  }
  if (window.tile_count > static_cast<rund::kernel::u64>(
                              std::numeric_limits<std::uint32_t>::max())) {
    SetVulkanLastError(adapter, "compute_dispatch_overflow");
    return rund::AccelCheck{false, "compute_dispatch_overflow"};
  }
  VulkanDescriptorScratch scratch{};
  SelectVulkanDescriptorScratch(adapter, descriptor_count, scratch);
  scratch.infos[0] =
      VkDescriptorBufferInfo{map.param.buffer.buffer, 0u, map.param.used_bytes};
  if (!BindVulkanMapInputs(adapter, map, window, scratch.infos)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (!BindVulkanMapOutputs(adapter, map, window, scratch.infos)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (map.controlled()) {
    scratch.infos[descriptor_count - 1u] = VkDescriptorBufferInfo{
        map.control_args.buffer.buffer, 0u, map.control_args.used_bytes};
  }
  if (!WriteVulkanStorageDescriptorSet(adapter, descriptor_set, scratch.infos,
                                       descriptor_count)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (map.controlled()) {
    if (map.control_pipeline == nullptr ||
        map.control_descriptor == VK_NULL_HANDLE ||
        map.control_args.buffer.buffer == VK_NULL_HANDLE) {
      SetVulkanLastError(adapter, "compute_plan_invalid");
      return rund::AccelCheck{false, "compute_plan_invalid"};
    }
    EncodeVulkanMapIndirect(
        command_buffer, *map.prepared->pipeline, descriptor_set, window_index,
        map.iterations, map.control_args.buffer.buffer,
        static_cast<VkDeviceSize>(window_index) * 4u * sizeof(std::uint32_t));
  } else {
    EncodeVulkanMap(command_buffer, *map.prepared->pipeline, descriptor_set,
                    static_cast<std::uint32_t>(window.tile_count),
                    map.iterations);
  }
  return EncodeVulkanMapOutputBarrier(command_buffer, map, window)
             ? rund::AccelCheck{true, "ok"}
             : rund::AccelCheck{false, VulkanLastError(&adapter)};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
