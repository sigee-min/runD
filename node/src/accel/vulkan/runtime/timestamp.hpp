#pragma once

#include <accel/runtime.hpp>

#include <rund/counter.hpp>
#include "../adapter/api.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] inline bool
VulkanTimestampAvailable(const VulkanAdapter &adapter) noexcept {
  return adapter.timestamp_query_available &&
         adapter.timestamp_valid_bits != 0u &&
         adapter.timestamp_period_ns > 0.0F;
}

[[nodiscard]] inline VkQueryPool
RecordingVulkanTimestampPool(const VulkanAdapter &adapter) noexcept {
  return adapter.recording_command < adapter.commands.size()
             ? adapter.commands[adapter.recording_command].timestamps
             : VK_NULL_HANDLE;
}

inline void BeginVulkanTimestampSpan(VulkanAdapter &adapter,
                                     const VkCommandBuffer command_buffer) {
  if (!VulkanTimestampAvailable(adapter)) {
    return;
  }
  const VkQueryPool timestamps = RecordingVulkanTimestampPool(adapter);
  if (timestamps == VK_NULL_HANDLE) {
    return;
  }
  vkCmdResetQueryPool(command_buffer, timestamps, 0u, 2u);
  vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      timestamps, 0u);
}

inline void EndVulkanTimestampSpan(VulkanAdapter &adapter,
                                   const VkCommandBuffer command_buffer) {
  if (!VulkanTimestampAvailable(adapter)) {
    return;
  }
  const VkQueryPool timestamps = RecordingVulkanTimestampPool(adapter);
  if (timestamps == VK_NULL_HANDLE) {
    return;
  }
  vkCmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      timestamps, 1u);
}

[[nodiscard]] inline std::uint64_t
VulkanTimestampTicks(std::uint64_t start, std::uint64_t end,
                     const std::uint32_t valid_bits) noexcept {
  if (valid_bits == 0u) {
    return 0u;
  }
  if (valid_bits < 64u) {
    const std::uint64_t mask = (std::uint64_t{1u} << valid_bits) - 1u;
    start &= mask;
    end &= mask;
  }
  if (end >= start) {
    return end - start;
  }
  if (valid_bits >= 64u) {
    return (std::numeric_limits<std::uint64_t>::max() - start) + 1u + end;
  }
  const std::uint64_t modulus = std::uint64_t{1u} << valid_bits;
  return (modulus - start) + end;
}

[[nodiscard]] inline bool
CollectVulkanTimestampSpan(VulkanAdapter &adapter, const std::size_t slot,
                           rund::RuntimeStats *local = nullptr) {
  if (!VulkanTimestampAvailable(adapter) || slot >= adapter.commands.size()) {
    return true;
  }
  const VkQueryPool query_pool = adapter.commands[slot].timestamps;
  if (query_pool == VK_NULL_HANDLE) {
    return true;
  }
  std::uint64_t timestamps[2] = {0u, 0u};
  const VkResult result = vkGetQueryPoolResults(
      adapter.device, query_pool, 0u, 2u, sizeof(timestamps), timestamps,
      sizeof(std::uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  if (result != VK_SUCCESS) {
    SetVulkanLastError(adapter, "accel_vulkan_timestamp_unavailable");
    return false;
  }
  const std::uint64_t ticks = VulkanTimestampTicks(
      timestamps[0], timestamps[1], adapter.timestamp_valid_bits);
  if (ticks == 0u) {
    return true;
  }
  const long double elapsed_ns =
      static_cast<long double>(ticks) *
      static_cast<long double>(adapter.timestamp_period_ns);
  if (elapsed_ns <= 0.0L) {
    return true;
  }
  const auto ns_limit =
      static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
  const std::uint64_t ns = elapsed_ns >= ns_limit
                               ? std::numeric_limits<std::uint64_t>::max()
                               : static_cast<std::uint64_t>(elapsed_ns);
  ::rund::detail::counter::Accumulate(adapter.accel_kernel_ns, ns);
  ::rund::detail::counter::Accumulate(adapter.accel_timestamp_count, 1u);
  adapter.accel_timestamp_source = "vulkan_timestamp_query";
  if (local != nullptr) {
    local->accel_kernel_ns = ns;
    local->accel_timestamp_count = 1u;
    local->accel_timestamp_source = "vulkan_timestamp_query";
  }
  return true;
}

#endif // defined(RUND_NODE_HAVE_VULKAN_SDK)

} // namespace rund::node::accel::detail
