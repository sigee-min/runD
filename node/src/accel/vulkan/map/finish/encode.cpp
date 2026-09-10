#include "../../adapter/error.hpp"

#include "../api.hpp"
#include "../encode/control.hpp"
#include "../encode/window.hpp"
#include "../local.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck EncodeVulkanMap(VulkanAdapter &adapter,
                                 const std::shared_ptr<void> &resources,
                                 void *const command_buffer_raw) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const map = static_cast<VulkanMapEncodeResources *>(resources.get());
  const VkCommandBuffer command_buffer =
      reinterpret_cast<VkCommandBuffer>(command_buffer_raw);
  if (map == nullptr || map->adapter != &adapter || map->prepared == nullptr ||
      command_buffer == VK_NULL_HANDLE || map->prepared->pipeline == nullptr ||
      map->param.buffer.buffer == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  const std::uint32_t descriptor_count =
      static_cast<std::uint32_t>(map->prepared->plan.input_buffer_count +
                                 map->prepared->plan.output_buffer_count + 1u +
                                 (map->controlled() ? 1u : 0u));
  if (map->descriptor_sets.count != map->windows.size()) {
    SetVulkanLastError(adapter, "accel_vulkan_descriptor_unavailable");
    return rund::AccelCheck{false, "accel_vulkan_descriptor_unavailable"};
  }
  if (map->controlled() &&
      !ResetVulkanStatus(command_buffer, map->control_status,
                         2u * sizeof(std::uint32_t))) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (map->mode == VulkanMapMode::Generated &&
      (map->generated_control_pipeline == nullptr ||
       map->generated_control_descriptor == VK_NULL_HANDLE ||
       map->generated_gate_binding.buffer == VK_NULL_HANDLE ||
       map->generated_summary_binding.buffer == VK_NULL_HANDLE)) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  if (!map->prepared->checks.empty()) {
    if (map->check_pipeline == nullptr ||
        map->check_descriptor == VK_NULL_HANDLE) {
      SetVulkanLastError(adapter, "compute_plan_invalid");
      return rund::AccelCheck{false, "compute_plan_invalid"};
    }
    EncodeVulkanMapCheck(command_buffer, *map);
  }
  if (map->controlled()) {
    for (std::size_t index = 0u; index < map->windows.size(); ++index) {
      EncodeVulkanMapControl(command_buffer, *map, map->windows[index],
                             static_cast<std::uint32_t>(index));
    }
  }
  for (std::size_t index = 0u; index < map->windows.size(); ++index) {
    const rund::AccelCheck encoded = EncodeVulkanMapWindow(
        adapter, *map, map->windows[index], map->descriptor_sets.at(index),
        descriptor_count, static_cast<std::uint32_t>(index), command_buffer);
    if (!encoded.ok) {
      return encoded;
    }
  }
  if (map->controlled() &&
      !FinishVulkanStatus(command_buffer, map->control_status, {})) {
    SetVulkanLastError(adapter, "compute_plan_invalid");
    return rund::AccelCheck{false, "compute_plan_invalid"};
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)resources;
  (void)command_buffer_raw;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
