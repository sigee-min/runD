#pragma once

#include "state.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

struct VulkanPipelineWork final {
  std::uint64_t dispatch_count{};
  std::uint64_t workgroup_count{};
  std::uint64_t work_item_count{};
  bool exact{true};
};

struct VulkanMapEncodeResources;

[[nodiscard]] bool AccumulateVulkanWork(const VulkanMapEncodeResources &map,
                                        VulkanPipelineWork &work) noexcept;
[[nodiscard]] bool AccumulateVulkanWork(const VulkanKernelEntry &entry,
                                        VulkanPipelineWork &work) noexcept;
[[nodiscard]] std::uint64_t
VulkanRecurrenceHostBytes(const VulkanPipeline &pipeline) noexcept;

#endif

} // namespace rund::node::accel::detail
