#pragma once

#include <accel/api.hpp>
#include <accel/device.hpp>

#include "model.hpp"

#include <string_view>

namespace node_accel_contract::fusion {

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
  return reason == "accel_vulkan_loader_unavailable" ||
         reason == "accel_vulkan_instance_unavailable" ||
         reason == "accel_vulkan_portability_unavailable" ||
         reason == "accel_vulkan_device_unavailable" ||
         reason == "accel_vulkan_queue_unavailable" ||
         reason == "accel_vulkan_shader_tool_unavailable" ||
         reason == "accel_vulkan_unavailable";
}

[[nodiscard]] inline rund::AccelPolicy
Policy(const rund::AccelApi api) noexcept {
  return rund::AccelPolicy{
      .preferred = {api, rund::AccelApi::Auto, rund::AccelApi::Auto},
      .preferred_count = 1u,
      .allow_fake = false,
  };
}

} // namespace node_accel_contract::fusion
