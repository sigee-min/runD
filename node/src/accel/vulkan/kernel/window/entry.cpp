#include "internal.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

rund::AccelCheck PrepareVulkanWindow(
    VulkanAdapter &adapter, const std::span<const BackendBatchEntry> entries,
    const std::uint64_t dispatch_capacity, const std::uint64_t gate_capacity,
    const PreparedPipelineStatusLayout &status, const VulkanBuffer &control,
    VulkanWindowResources &resources) {
  resources = {};
  const rund::AccelCheck admitted =
      PrepareVulkanWindowResources(adapter, entries, dispatch_capacity,
                                   gate_capacity, status, control, resources);
  if (!admitted.ok) {
    return admitted;
  }
  if (resources.state_count == 0u) {
    return rund::AccelCheck{true, "ok"};
  }
  const rund::AccelCheck descriptors =
      PrepareVulkanWindowDescriptors(adapter, control, resources);
  if (!descriptors.ok) {
    DestroyVulkanWindow(resources);
  }
  return descriptors;
}

#endif

} // namespace rund::node::accel::detail
