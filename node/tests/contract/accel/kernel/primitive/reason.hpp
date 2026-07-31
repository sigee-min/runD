#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include <string_view>

namespace node_accel_contract::primitive {

[[nodiscard]] inline bool
PickUnavailableReasonIsPrecise(const rund::AccelDevice &pick,
                               const rund::AccelApi api) noexcept {
  if (pick.check.ok) {
    return false;
  }
  const std::string_view reason{pick.check.reason};
  if (api == rund::AccelApi::Metal) {
    return reason == "accel_metal_unavailable" ||
           reason == "accel_metal_device_unavailable" ||
           reason == "accel_metal_queue_unavailable" ||
           reason == "accel_metal_sdk_unavailable";
  }
  if (api == rund::AccelApi::Vulkan) {
    return reason == "accel_vulkan_loader_unavailable" ||
           reason == "accel_vulkan_instance_unavailable" ||
           reason == "accel_vulkan_portability_unavailable" ||
           reason == "accel_vulkan_device_unavailable" ||
           reason == "accel_vulkan_queue_unavailable" ||
           reason == "accel_vulkan_shader_tool_unavailable" ||
           reason == "accel_vulkan_unavailable";
  }
  return false;
}

} // namespace node_accel_contract::primitive
