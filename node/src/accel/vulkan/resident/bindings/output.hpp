#pragma once

#include "../../buffer/resident/find.hpp"
#include "check.hpp"

#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] inline bool PrepareVulkanResidentOutput(
    VulkanResidentState &resident, const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings, VulkanResidentBindings &out) {
  if (plan.output_buffer_count >
      static_cast<rund::kernel::u64>(
          std::numeric_limits<std::size_t>::max())) {
    out.reason = "compute_binding_mismatch";
    return false;
  }
  const std::size_t count =
      static_cast<std::size_t>(plan.output_buffer_count);
  std::vector<VulkanResidentBufferResult> outputs(count);
  for (std::size_t index = 0u; index < count; ++index) {
    const rund::kernel::ResidentBufferRef *const ref =
        bindings.resident_outputs.ref(index);
    const std::shared_ptr<void> *const handle =
        bindings.resident_outputs.handle(index);
    if (ref == nullptr || handle == nullptr || *handle == nullptr ||
        ref->stride_bytes < ref->element_bytes) {
      out.reason = "compute_resident_stride_invalid";
      return false;
    }
    const VulkanResidentBufferResult result = ResolveVulkanResidentBuffer(
        resident, *ref, *handle, "compute_resident_id_invalid", true);
    if (!StoreResident(&outputs[index], result)) {
      out.reason = "compute_binding_mismatch";
      return false;
    }
    if (!outputs[index].check.ok ||
        outputs[index].device_buffer == nullptr) {
      out.reason = outputs[index].check.reason;
      return false;
    }
  }
  if (outputs.size() != count) {
    out.reason = "compute_resident_stride_invalid";
    return false;
  }
  out.outputs = std::move(outputs);
  return true;
}
#endif

}  // namespace rund::node::accel::detail
