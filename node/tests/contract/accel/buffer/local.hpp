#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/check.hpp>
#include <accel/device.hpp>

#include <kernel/program/compute/dsl.hpp>

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace node_accel_contract::buffer {

[[nodiscard]] inline bool CheckReason(const rund::AccelCheck &check,
                                      const std::string_view reason) noexcept {
  return !check.ok && std::string_view{check.reason} == reason;
}

[[nodiscard]] inline bool
VulkanPickReasonIsPrecise(const rund::AccelDevice &pick) noexcept {
  const std::string_view reason{pick.check.reason};
  return reason == "accel_vulkan_loader_unavailable" ||
         reason == "accel_vulkan_instance_unavailable" ||
         reason == "accel_vulkan_portability_unavailable" ||
         reason == "accel_vulkan_device_unavailable" ||
         reason == "accel_vulkan_queue_unavailable" ||
         reason == "accel_vulkan_shader_tool_unavailable" ||
         reason == "accel_vulkan_unavailable";
}

[[nodiscard]] inline rund::AccelPolicy
Policy(const rund::AccelApi api, const bool allow_fake = false) noexcept {
  return rund::AccelPolicy{
      .preferred = {api, rund::AccelApi::Auto, rund::AccelApi::Auto},
      .preferred_count = 1u,
      .allow_fake = allow_fake,
  };
}

[[nodiscard]] inline rund::BufferDesc BufferDesc() noexcept {
  return rund::BufferDesc{
      .bytes = 16u,
      .usage = rund::BufferUsage::ReadWrite,
      .alignment = 16u,
  };
}

template <typename T>
concept HasElementBytes = requires(T desc) { desc.element_bytes; };

template <typename T>
concept HasStrideBytes = requires(T desc) { desc.stride_bytes; };

template <typename T>
concept HasCount = requires(T desc) { desc.count; };

} // namespace node_accel_contract::buffer

namespace node_accel_contract {

[[nodiscard]] bool PublicBufferApiRejectsInvalidDescriptors();
[[nodiscard]] bool PublicBufferApiRejectsUnavailableBackends();
[[nodiscard]] bool PublicBufferApiExposesMetalResidencyWhenAvailable(
    const rund::AccelDevice &pick);
[[nodiscard]] bool PublicBufferApiRoundTripsAndReportsStatsWhenAvailable(
    const rund::AccelDevice &pick);
[[nodiscard]] bool
PublicBufferApiRejectsRangeAndOwnerFailures(const rund::AccelDevice &pick);

} // namespace node_accel_contract
