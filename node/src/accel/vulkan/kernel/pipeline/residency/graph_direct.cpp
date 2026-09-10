#include "graph_direct.hpp"
#include "graph_direct/local.hpp"

#include "../prepare/record.hpp"

#include <memory>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

[[nodiscard]] bool
BuildGraphDirectProof(const VulkanPipeline &pipeline,
                      VulkanResidencyGraphStageDirectProof &proof) noexcept {
  proof = {};
  if (pipeline.record == nullptr || pipeline.record->entries.empty() ||
      pipeline.record->entries.size() > PreparedPipelineStepCapacity ||
      pipeline.record->recurrence || pipeline.profile != nullptr ||
      !pipeline.window.routes.empty() || !pipeline.publish.routes.empty() ||
      !pipeline.transducers.empty()) {
    return false;
  }
  for (const VulkanPipelineRecordEntry &record_entry :
       pipeline.record->entries) {
    const auto *const resources =
        record_entry.prepared == nullptr
            ? nullptr
            : static_cast<const VulkanKernelResources *>(
                  record_entry.prepared.get());
    const BackendRun *const run =
        resources == nullptr || resources->program == nullptr
            ? nullptr
            : resources->program->signature;
    const VulkanKernelEntry *const entry =
        resources == nullptr ? nullptr : resources->entry(0u);
    const BoundStep *const bound =
        entry == nullptr || entry->view == nullptr
            ? (run == nullptr || run->steps == nullptr ? nullptr
                                                       : &run->steps[0])
            : &entry->view->step;
    std::uint64_t direct = 0u;
    std::uint64_t indirect = 0u;
    const rund::AccelCheck described =
        resources == nullptr
            ? rund::AccelCheck{false, "accel_kernel_pipeline_invalid"}
            : DescribeVulkanRouteDispatches(*resources, direct, indirect);
    if (resources == nullptr || run == nullptr || resources->size() != 1u ||
        run->step_count != resources->size() || run->steps == nullptr ||
        bound == nullptr || bound->source_binds == nullptr || !described.ok ||
        indirect != 0u ||
        !graph_direct_detail::GraphDirectExecution(*run, *bound,
                                                   *bound->source_binds)) {
      return false;
    }
    const auto &admission = run->execution->admission;
    const std::uint32_t operation =
        static_cast<std::uint32_t>(bound->step->kind());
    if (!proof.valid) {
      proof.kernel_id = admission.kernel_id;
      proof.graph_id_hi = admission.graph_id_hi;
      proof.graph_id_lo = admission.graph_id_lo;
      proof.node_count = admission.node_count;
      proof.operation = operation;
      proof.read_count =
          static_cast<std::uint32_t>(bound->step->artifact.metadata.read_count);
      proof.write_count = static_cast<std::uint32_t>(
          bound->step->artifact.metadata.write_count);
      proof.local_count =
          static_cast<std::uint32_t>(pipeline.record->entries.size());
      proof.direct_dispatch_count = direct;
      proof.valid = true;
    } else if (proof.kernel_id != admission.kernel_id ||
               proof.graph_id_hi != admission.graph_id_hi ||
               proof.graph_id_lo != admission.graph_id_lo ||
               proof.node_count != admission.node_count ||
               proof.operation != operation ||
               proof.read_count != bound->step->artifact.metadata.read_count ||
               proof.write_count !=
                   bound->step->artifact.metadata.write_count ||
               proof.direct_dispatch_count != direct) {
      return false;
    }
  }
  return proof.valid;
}

[[nodiscard]] bool
GraphDirectProofMatches(const VulkanPipeline &pipeline,
                        const VulkanResidencySelection &selection) noexcept {
  if (!selection.graph_direct.valid) {
    return false;
  }
  VulkanResidencyGraphStageDirectProof current{};
  return BuildGraphDirectProof(pipeline, current) &&
         current.kernel_id == selection.graph_direct.kernel_id &&
         current.graph_id_hi == selection.graph_direct.graph_id_hi &&
         current.graph_id_lo == selection.graph_direct.graph_id_lo &&
         current.node_count == selection.graph_direct.node_count &&
         current.operation == selection.graph_direct.operation &&
         current.read_count == selection.graph_direct.read_count &&
         current.write_count == selection.graph_direct.write_count &&
         current.direct_dispatch_count ==
             selection.graph_direct.direct_dispatch_count &&
         current.local_count == selection.graph_direct.local_count;
}

#endif

} // namespace rund::node::accel::detail
