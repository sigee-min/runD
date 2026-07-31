#pragma once

#include <accel/check.hpp>

#include "../../../runtime/timestamp.hpp"
#include "prepare.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] rund::AccelCheck
SubmitVulkanScanDirect(VulkanAdapter &adapter, VulkanScanDirectState &state) {
  if (!EnsureVulkanCommandResources(adapter) || !BeginVulkanCommand(adapter)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  BeginVulkanTimestampSpan(adapter, adapter.command_buffer);
  const rund::AccelCheck encoded =
      EncodeVulkanScanBuffers(adapter, state.resources,
                              reinterpret_cast<void *>(adapter.command_buffer));
  if (!encoded.ok) {
    CancelVulkanCommand(adapter);
    return encoded;
  }
  EndVulkanTimestampSpan(adapter, adapter.command_buffer);
  if (!SubmitVulkanCommand(adapter)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace
#endif

} // namespace rund::node::accel::detail
