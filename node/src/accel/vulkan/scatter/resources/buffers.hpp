#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool CreateVulkanScatterBuffers(
    VulkanAdapter& adapter,
    VulkanScatterEncodeResources& resources,
    const ScatterParams& params_value) {
  return CreateVulkanBuffer(adapter, sizeof(params_value),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                resources.params) &&
         UploadVulkanBuffer(resources.params, &params_value,
                            sizeof(params_value)) &&
         CreateVulkanStatus(adapter, resources.plan.status_bytes,
                            resources.status);
}

}  // namespace
#endif

}  // namespace rund::node::accel::detail
