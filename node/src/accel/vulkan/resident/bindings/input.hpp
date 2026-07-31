#pragma once

#include "output.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline bool PrepareVulkanResidentInputs(
    VulkanResidentState &resident, const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings, VulkanResidentBindings &out) {
  if (plan.input_buffer_count >
      static_cast<rund::kernel::u64>(
          std::numeric_limits<std::size_t>::max())) {
    out.reason = "compute_binding_mismatch";
    return false;
  }
  const std::size_t count = static_cast<std::size_t>(plan.input_buffer_count);
  std::vector<VulkanResidentBufferResult> inputs(count);
  if (inputs.size() != count || (count != 0u && inputs.data() == nullptr)) {
    out.reason = "compute_binding_mismatch";
    return false;
  }
  for (std::size_t index = 0u; index < count; ++index) {
    const rund::kernel::ResidentBufferRef *const ref =
        bindings.resident_inputs.ref(index);
    const std::shared_ptr<void> *const handle =
        bindings.resident_inputs.handle(index);
    if (ref == nullptr || handle == nullptr ||
        ref->stride_bytes < ref->element_bytes) {
      out.reason = "compute_resident_stride_invalid";
      return false;
    }
    VulkanResidentBufferResult* const slot = inputs.data() + index;
    const VulkanResidentBufferResult input = ResolveVulkanResidentBuffer(
        resident, *ref, *handle, "compute_resident_id_invalid", true);
    if (!StoreResident(slot, input)) {
      out.reason = "compute_binding_mismatch";
      return false;
    }
    if (!slot->check.ok || slot->device_buffer == nullptr) {
      out.reason = slot->check.reason;
      return false;
    }
  }
  out.inputs = std::move(inputs);
  return true;
}
#endif

}  // namespace rund::node::accel::detail
