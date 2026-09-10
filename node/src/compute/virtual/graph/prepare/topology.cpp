#include "internal.hpp"

#include <algorithm>
#include <new>
#include <utility>

namespace rund::compute::detail::virtual_graph_prepare_detail {

Status plan_graph_topology(GraphPreparationDraft &draft) noexcept {
  // Topology owns the physical footprint. Derive interval colors and the
  // unique physical-class table before turning byte budgets into K; using the
  // legacy one-Input/one-Intermediate/one-Output geometry here would admit a
  // multi-resource Graph whose actual retained owners exceed its config.
  const std::uint64_t frame_limit = std::min(
      {draft.page_count, static_cast<std::uint64_t>(PipelineLeafCapacity),
       static_cast<std::uint64_t>(PipelineIterationCapacity)});
  try {
    auto shape_resources = draft.graph_resources;
    for (residency::TiledGraphResourceInput &resource : shape_resources) {
      resource.remaps.clear();
    }
    draft.topology = residency::PlanResidency(residency::TiledGraphPlanInput{
        .page_count = draft.page_count,
        .requested_frames = frame_limit,
        .max_frames = std::min(PipelineLeafCapacity, PipelineIterationCapacity),
        .prefetch_distance = draft.prefetch_distance,
        .graph_fingerprint_hi = draft.program->graph_info.fingerprint.hi,
        .graph_fingerprint_lo = draft.program->graph_info.fingerprint.lo,
        .resources = std::move(shape_resources),
        .stages = draft.graph_stages,
    });
  } catch (const std::bad_alloc &) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (!draft.topology) {
    return Status::fail(draft.topology.failure == residency::Failure::Infeasible
                            ? Reason::PipelineMemoryBudget
                            : Reason::PipelineCapacity);
  }
  return Status::success();
}

} // namespace rund::compute::detail::virtual_graph_prepare_detail
