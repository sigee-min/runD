#pragma once

#include "resident.hpp"
#include "staged.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool BindInputDescriptors(
    VulkanAdapter& adapter,
    const rund::kernel::ComputePlan& plan,
    const rund::kernel::ComputeDispatchWindow& window,
    const rund::kernel::BindingSet& bindings,
    const VulkanResidentBindings* const resident_bindings,
    const VulkanWindowBuffers& buffers,
    VkDescriptorBufferInfo* const infos) {
  rund::kernel::u64 input_cursor = 0u;
  for (rund::kernel::u64 index = 0u; index < plan.input_buffer_count; ++index) {
    VkDescriptorBufferInfo& info =
        infos[static_cast<std::size_t>(index + 1u)];
    if (buffers.resident) {
      if (!BindResidentInputDescriptor(adapter, window, bindings,
                                       resident_bindings, index, info)) {
        return false;
      }
    } else if (!BindStagedInputDescriptor(adapter, window, bindings, buffers,
                                          index, input_cursor, info)) {
      return false;
    }
  }
  return true;
}

}  // namespace
#endif

}  // namespace rund::node::accel::detail
