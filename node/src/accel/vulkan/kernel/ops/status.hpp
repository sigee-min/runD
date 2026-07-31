#pragma once

#include "model.hpp"

#include "../../status.hpp"

#include <rund/compute/reason.hpp>

#include <array>
#include <cstdint>
#include <limits>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanPipelineStatusMapping final {
  std::uint32_t raw{};
  rund::compute::Reason reason{rund::compute::Reason::ReasonInvalid};
};

template <std::size_t Count>
[[nodiscard]] inline rund::AccelCheck DescribeVulkanPipelineStatus(
    VulkanStatus &status, const std::uint64_t entry_count,
    const VulkanPipelineStatusRule rule, const std::uint32_t success,
    const std::array<VulkanPipelineStatusMapping, Count> &mapping,
    VulkanPipelineStatusSource &source) noexcept {
  source = {};
  if (status.device.buffer == VK_NULL_HANDLE || entry_count == 0u ||
      entry_count > std::numeric_limits<std::uint32_t>::max() ||
      entry_count > status.device.bytes / sizeof(std::uint32_t) ||
      Count > source.raw_values.size()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  source.raw = &status.device;
  source.count = static_cast<std::uint32_t>(entry_count);
  source.rule = rule;
  source.success = success;
  source.mapping_count = static_cast<std::uint32_t>(Count);
  for (std::size_t index = 0u; index < Count; ++index) {
    source.raw_values[index] = mapping[index].raw;
    source.reasons[index] = static_cast<std::uint32_t>(mapping[index].reason);
  }
  status.pipeline = true;
  return rund::AccelCheck{true, "ok"};
}

#endif

} // namespace rund::node::accel::detail
