#pragma once

#include "output.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] bool PrepareVulkanWindowBuffers(
    VulkanAdapter& adapter,
    const rund::kernel::ComputePlan& plan,
    const rund::kernel::ComputeDispatchWindow& window,
    const rund::kernel::BindingSet& bindings,
    const VulkanResidentBindings* const resident_bindings,
    VulkanWindowBuffers& out) {
  return AdmitVulkanWindowBuffers(adapter, window, bindings, resident_bindings,
                                  out) &&
         PrepareVulkanStagedInputBuffer(adapter, plan, window, bindings, out) &&
         PrepareVulkanOutputBuffer(adapter, plan, window, out);
}
#endif

}  // namespace rund::node::accel::detail
