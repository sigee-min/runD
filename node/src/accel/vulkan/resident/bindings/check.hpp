#pragma once

#include "store.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline bool VulkanResidentBindingShapeOk(
    const rund::kernel::ComputePlan& plan,
    const rund::kernel::BindingSet& bindings,
    const char*& reason) {
  if ((plan.input_buffer_count != 0u &&
       (!bindings.resident_inputs.has_refs() ||
        !bindings.resident_inputs.has_handles())) ||
      bindings.resident_inputs.count != plan.input_buffer_count ||
      !bindings.resident_outputs.has_refs() ||
      !bindings.resident_outputs.has_handles() ||
      bindings.resident_outputs.count != plan.output_buffer_count ||
      plan.output_buffer_count == 0u) {
    reason = "compute_binding_mismatch";
    return false;
  }
  reason = "ok";
  return true;
}
#endif

}  // namespace rund::node::accel::detail
