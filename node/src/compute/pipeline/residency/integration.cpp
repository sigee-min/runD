#include "integration/local.hpp"

namespace rund::compute::detail {

Status bind_pipeline_residency(const PipelineMemoryPlan &plan,
                               PipelineState &state) noexcept {
  if (plan.residency.pages == nullptr) {
    return state.residency == nullptr ? Status::success()
                                      : Status::fail(Reason::PipelineInvalid);
  }
  if (plan.residency.stage == PipelineResidencyStage::Graph &&
      plan.residency.semantic != PipelineResidencySemantic::None) {
    return bind_graph_semantic_residency(plan, state);
  }
  if (plan.residency.stage == PipelineResidencyStage::Graph) {
    return bind_graph_residency(plan, state);
  }
  return bind_direct_residency(plan, state);
}

} // namespace rund::compute::detail
