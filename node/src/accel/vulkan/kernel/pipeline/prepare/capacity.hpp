#pragma once

#include "../state.hpp"

#include <cstdint>
#include <span>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanPipelineDescriptionCapacity final {
  std::uint64_t step_count{};
  std::uint64_t status_source_count{};
  std::uint64_t status_entry_count{};
  std::uint64_t telemetry_source_count{};
};

[[nodiscard]] rund::AccelCheck DescribeVulkanPipelineCapacity(
    std::span<const BackendBatchEntry> templates,
    VulkanPipelineDescriptionCapacity &capacity) noexcept;

#endif

} // namespace rund::node::accel::detail
