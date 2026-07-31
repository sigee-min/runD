#include "local.hpp"
#include "resources/binding.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool CreateVulkanGatherDescriptorSet(VulkanAdapter &adapter,
                                     VulkanGatherEncodeResources &resources) {
  if (!AcquireVulkanCollectiveDescriptorSet(
          adapter, *resources.control_pipeline, kGatherDescriptorCount,
          resources.control_descriptor) ||
      !AcquireVulkanCollectiveDescriptorSet(adapter, *resources.gather_pipeline,
                                            kGatherDescriptorCount,
                                            resources.gather_descriptor)) {
    return false;
  }
  const VulkanStorageBinding count =
      resources.plan.count_source ==
              rund::kernel::ComputeCountSource::Descriptor
          ? VulkanStorageBindingFor(resources.params)
          : GatherResidentBinding(resources.logical_count);
  const std::array<VulkanStorageBinding, kGatherDescriptorCount> bindings{
      VulkanStorageBindingFor(resources.params),
      GatherResidentBinding(resources.values),
      GatherResidentBinding(resources.indices),
      GatherResidentBinding(resources.output),
      VulkanStorageBindingFor(resources.status.device),
      count,
      VulkanStorageBindingFor(resources.indirect)};
  return WriteVulkanStorageDescriptorSet(adapter, resources.control_descriptor,
                                         bindings) &&
         WriteVulkanStorageDescriptorSet(adapter, resources.gather_descriptor,
                                         bindings);
}
#endif

} // namespace rund::node::accel::detail
