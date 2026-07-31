#pragma once

#include <accel/check.hpp>

#include "submit.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
rund::AccelCheck SubmitVulkanEncodedResources(
    VulkanAdapter &adapter, const std::shared_ptr<void> &resources,
    const VulkanEncodedResourceFn encode, const VulkanFinishResourceFn finish) {
  if (!EnsureVulkanCommandResources(adapter) || !BeginVulkanCommand(adapter)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  BeginVulkanTimestampSpan(adapter, adapter.command_buffer);
  const rund::AccelCheck encoded = encode(
      adapter, resources, reinterpret_cast<void *>(adapter.command_buffer));
  if (!encoded.ok) {
    CancelVulkanCommand(adapter);
    return encoded;
  }
  EndVulkanTimestampSpan(adapter, adapter.command_buffer);
  if (!SubmitVulkanCommand(adapter)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  return finish(adapter, resources);
}
#endif

} // namespace rund::node::accel::detail
