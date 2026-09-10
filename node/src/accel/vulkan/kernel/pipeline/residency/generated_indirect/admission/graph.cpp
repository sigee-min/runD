#include "../internal.hpp"

namespace rund::node::accel::detail::vulkan_generated_indirect_detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

GraphAdmission admit_graph(VulkanPipeline &pipeline) noexcept {
  GraphAdmission result{};
  if (impl::eligible_graph(pipeline, result.graph)) {
    result.candidate =
        VulkanResidencyAdmissionCandidate::GraphStageGeneratedIndirect;
    return result;
  }
  if (impl::eligible_sequence(pipeline, result.sequence)) {
    result.candidate = VulkanResidencyAdmissionCandidate::GraphStageSequence;
  }
  return result;
}

bool matches(const VulkanPipeline &pipeline,
             const VulkanResidencySelection &selection) noexcept {
  if (!vulkan_residency_detail::owns(selection.mode)) {
    return false;
  }
  return selection.mode == VulkanResidencyMode::GraphStageGeneratedIndirect
             ? impl::graph_proof_matches(pipeline, selection)
             : impl::sequence_proof_matches(pipeline, selection);
}

#endif

} // namespace rund::node::accel::detail::vulkan_generated_indirect_detail
