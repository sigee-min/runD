#pragma once

#include "../local.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline bool AdmitVulkanWindowBuffers(
    VulkanAdapter& adapter,
    const rund::kernel::ComputeDispatchWindow& window,
    const rund::kernel::BindingSet& bindings,
    const VulkanResidentBindings* const resident_bindings,
    VulkanWindowBuffers& out) {
  out = {};
  out.resident = bindings.has_resident_output();
  if (out.resident) {
    if (resident_bindings == nullptr ||
        resident_bindings->bindings != &bindings ||
        resident_bindings->outputs.size() != bindings.resident_outputs.count) {
      SetVulkanLastError(adapter, "compute_binding_mismatch");
      return false;
    }
    return true;
  }
  out.staged = PlanStagedWindow(bindings, window);
  if (!out.staged.ok()) {
    SetVulkanLastError(adapter, "compute_dispatch_count_mismatch");
    return false;
  }
  return true;
}
#endif

}  // namespace rund::node::accel::detail
