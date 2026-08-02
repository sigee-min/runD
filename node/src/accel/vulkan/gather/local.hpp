#pragma once

#include "../../gather.hpp"
#include "../../gather/model.hpp"
#include "../adapter/api.hpp"
#include "../barrier.hpp"
#include "../buffer/resident/model.hpp"
#include "../collective/pipeline.hpp"
#include "../command.hpp"
#include "../descriptor.hpp"
#include "../status.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
inline constexpr rund::kernel::u32 kGatherBlockSize = 256u;
inline constexpr std::uint32_t kGatherDescriptorCount = 7u;

struct VulkanGatherEncodeResources {
  VulkanAdapter *adapter = nullptr;
  rund::kernel::GatherPlan plan{};
  VulkanCollectivePipeline *control_pipeline = nullptr;
  VulkanCollectivePipeline *gather_pipeline = nullptr;
  VulkanBuffer params{};
  VulkanBuffer indirect{};
  VulkanStatus status{};
  VulkanResidentBufferResult values{};
  VulkanResidentBufferResult indices{};
  VulkanResidentBufferResult logical_count{};
  VulkanResidentBufferResult output{};
  VkDescriptorSet control_descriptor = VK_NULL_HANDLE;
  VkDescriptorSet gather_descriptor = VK_NULL_HANDLE;
};

void DestroyVulkanGatherEncodeResources(void *raw);
[[nodiscard]] std::string
VulkanGatherSource(rund::kernel::GatherElement element, bool control);
[[nodiscard]] bool VulkanGatherSourceBytes(rund::kernel::GatherElement element,
                                           bool control,
                                           std::uint64_t &bytes) noexcept;
[[nodiscard]] VulkanCollectivePipeline *
AcquireGatherPipeline(VulkanAdapter &adapter,
                      const rund::kernel::GatherDesc &desc, bool control);
[[nodiscard]] bool
CreateVulkanGatherDescriptorSet(VulkanAdapter &adapter,
                                VulkanGatherEncodeResources &resources);
#endif

} // namespace rund::node::accel::detail
