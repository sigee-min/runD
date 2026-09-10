#include "internal.hpp"

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

bool record(VulkanPipeline &pipeline, VulkanResidencySelection &selection,
            const std::span<const std::uint32_t> locals) noexcept {
  if (selection.mode == VulkanResidencyMode::GraphStageGeneratedIndirect) {
    return impl::record_graph(pipeline, selection, locals) &&
           selection.graph_generated_command_count != 0u;
  }
  if (selection.mode == VulkanResidencyMode::GraphStageSequence) {
    return impl::record_sequence(pipeline, selection, locals) &&
           selection.graph_sequence_command_count == 2u;
  }
  return false;
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
