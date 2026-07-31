#pragma once

#include <accel/device.hpp>

#include "tile.hpp"

#include <array>
#include <string_view>

namespace node_accel_contract::vulkan {

[[nodiscard]] inline bool
FailureReasonIsSpecific(const rund::AccelDevice &pick) {
  constexpr std::array<std::string_view, 6u> reasons{
      "accel_vulkan_loader_unavailable",
      "accel_vulkan_instance_unavailable",
      "accel_vulkan_portability_unavailable",
      "accel_vulkan_device_unavailable",
      "accel_vulkan_queue_unavailable",
      "accel_vulkan_shader_tool_unavailable",
  };
  if (pick.check.ok) {
    return false;
  }
  for (const std::string_view reason : reasons) {
    if (std::string_view{pick.check.reason} == reason) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] inline bool
FailureReasonIsGenericCatchAll(const rund::AccelDevice &pick) {
  return !pick.check.ok &&
         std::string_view{pick.check.reason} == "accel_vulkan_unavailable";
}

[[nodiscard]] inline bool
FailureReasonIsPrecise(const rund::AccelDevice &pick) {
  return FailureReasonIsSpecific(pick) || FailureReasonIsGenericCatchAll(pick);
}

} // namespace node_accel_contract::vulkan
