#include "local.hpp"
#include <rund/counter.hpp>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
bool ExecuteWindow(VulkanAdapter& adapter,
                   VulkanCachedPipeline& pipeline,
                   const rund::kernel::ComputePlan& plan,
                   const rund::kernel::ComputeDispatchWindow& window,
                   const rund::kernel::BindingSet& bindings,
                   const ScopedBuffer& param_buffer,
                   const VulkanResidentBindings* const resident_bindings) {
  if (!EnsureVulkanCommandResources(adapter)) { return false; }
  if (param_buffer.buffer.buffer == VK_NULL_HANDLE) {
    SetVulkanLastError(adapter, "accel_vulkan_buffer_unavailable");
    return false;
  }
  VulkanWindowBuffers buffers{};
  if (!PrepareVulkanWindowBuffers(adapter, plan, window, bindings,
                                  resident_bindings, buffers)) {
    return false;
  }
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  if (!UpdateVulkanWindowDescriptorSet(adapter, pipeline, plan, window,
                                       bindings, param_buffer,
                                       resident_bindings, buffers,
                                       descriptor_set)) {
    return false;
  }
  if (!EncodeSubmitVulkanWindow(adapter, pipeline, window, descriptor_set,
                                buffers)) {
    return false;
  }
  if (!FinishVulkanWindowReadback(adapter, window, bindings, buffers)) {
    return false;
  }
  ::rund::detail::counter::Accumulate(adapter.dispatch_count, 1u);
  return true;
}
#endif

}  // namespace rund::node::accel::detail
