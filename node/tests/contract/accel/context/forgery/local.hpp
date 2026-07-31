#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/context/buffer/descriptor.hpp>
#include <accel/device.hpp>
#include <accel/kernel/check.hpp>

#include "src/accel/context/internal.hpp"

#include <array>
#include <string_view>

namespace node_accel_contract::context_forgery {

[[nodiscard]] bool CheckReason(const rund::AccelCheck &check,
                               const std::string_view reason) noexcept {
  return !check.ok && std::string_view{check.reason} == reason;
}

[[nodiscard]] bool CheckReason(const rund::AccelKernelCheck &check,
                               const std::string_view reason) noexcept {
  return !check.ok && std::string_view{check.reason} == reason;
}

[[nodiscard]] bool
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

[[nodiscard]] rund::AccelPolicy Policy(const rund::AccelApi api) noexcept {
  return rund::AccelPolicy{
      .preferred = {api, rund::AccelApi::Auto, rund::AccelApi::Auto},
      .preferred_count = 1u,
      .allow_fake = false,
  };
}

[[nodiscard]] rund::AccelBufferDesc
BufferDesc(const rund::BufferUsage usage) noexcept {
  return rund::AccelBufferDesc{
      .scalar_width_bytes = 4u,
      .count = 8u,
      .usage = usage,
  };
}

} // namespace node_accel_contract::context_forgery
