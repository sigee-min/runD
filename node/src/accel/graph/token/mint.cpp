#include <accel/context/buffer/descriptor.hpp>
#include <accel/graph/visibility.hpp>

#include "local.hpp"
#include "reset.hpp"

#include <utility>

namespace rund::node::accel::detail {

KernelTokenMint
MintKernelToken(const ContextAdmission &admission,
                const rund::kernel::GraphCheck &graph_check,
                std::vector<KernelExecutionStep> steps,
                std::vector<std::uint8_t> required_barriers,
                const std::uint64_t removed_dispatch_count,
                std::vector<rund::kernel::BufferRole> graph_roles,
                std::vector<rund::AccelBufferDesc> graph_shapes,
                std::vector<rund::GraphBufferVisibility> graph_visibilities,
                std::vector<std::uint64_t> graph_alias_representatives,
                std::vector<SourceStep> graph_binding_sources,
                std::vector<std::uint64_t> graph_reset_bindings,
                const rund::kernel::ComputeScalar scalar,
                const rund::kernel::ComputeDomain domain,
                const std::uint64_t original_operation_count,
                const std::uint64_t fused_operation_count,
                const std::uint64_t fusion_rejection_count,
                const char *const fusion_reason) {
  const bool fused = fused_operation_count < original_operation_count;
  std::vector<ResetPlan> resets;
  if (steps.empty() || required_barriers.size() != steps.size() ||
      !ValidSourcePartition(steps, original_operation_count) ||
      graph_roles.size() != graph_shapes.size() ||
      graph_roles.size() != graph_visibilities.size() ||
      graph_roles.size() != graph_alias_representatives.size() ||
      graph_roles.size() != graph_binding_sources.size() ||
      original_operation_count == 0u || fused_operation_count == 0u ||
      fused_operation_count > original_operation_count ||
      steps.size() != fused_operation_count ||
      fused != (removed_dispatch_count != 0u) ||
      !PlanResets(steps, graph_roles, graph_visibilities,
                  graph_alias_representatives, graph_binding_sources,
                  graph_reset_bindings, resets)) {
    return {};
  }
  std::shared_ptr<KernelToken> token = MakeKernelToken(KernelToken{
      .context_id = admission.context_id,
      .graph_id_hi = graph_check.graph_id_hi,
      .graph_id_lo = graph_check.graph_id_lo,
      .node_count = graph_check.node_count,
      .api = admission.api,
      .scalar = scalar,
      .domain = domain,
      .frozen_caps = admission.caps,
      .context_owner = admission.owner,
      .graph_roles = std::move(graph_roles),
      .graph_shapes = std::move(graph_shapes),
      .graph_visibilities = std::move(graph_visibilities),
      .graph_alias_representatives = std::move(graph_alias_representatives),
      .resets = std::move(resets),
      .steps = std::move(steps),
      .required_barriers = std::move(required_barriers),
      .removed_dispatch_count = removed_dispatch_count,
      .original_operation_count = original_operation_count,
      .fused_operation_count = fused_operation_count,
      .fusion_rejection_count = fusion_rejection_count,
      .fusion_reason = fusion_reason});
  if (token == nullptr || token->kernel_id == 0u) {
    return {};
  }
  return KernelTokenMint{.owner = std::static_pointer_cast<void>(token),
                         .kernel_id = token->kernel_id};
}

} // namespace rund::node::accel::detail
