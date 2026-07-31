#include <accel/api.hpp>
#include <accel/device.hpp>

#include "local.hpp"

#include <string_view>

namespace node_accel_contract::runtime::window {

[[nodiscard]] bool PickUnavailableReasonIsPrecise(const rund::AccelDevice &pick,
                                                  const rund::AccelApi api) {
  if (pick.check.ok) {
    return true;
  }

  const std::string_view reason{
      pick.check.reason == nullptr ? "" : pick.check.reason};
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

[[nodiscard]] const char *
BackendLastError(const rund::AccelDevice &pick) noexcept {
  if (pick.backend.last_error == nullptr) {
    return "compute_backend_failed";
  }

  const char *const reason = pick.backend.last_error(pick.backend.context);
  return reason == nullptr || reason[0] == '\0' ? "compute_backend_failed"
                                                : reason;
}

} // namespace node_accel_contract::runtime::window
