#include "internal.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>
#include <utility>

namespace rund::compute::detail::virtual_graph_prepare_detail {

Status project_graph_budget(GraphPreparationDraft &draft) noexcept {
  std::uint64_t per_bank_frame_bytes = 0u;
  std::uint64_t execution_frame_bytes = 0u;
  if (draft.topology.plan.page_bytes() == 0u ||
      !kernel::checked::add(draft.topology.plan.page_bytes(),
                            draft.control_page_bytes, per_bank_frame_bytes) ||
      !kernel::checked::mul(per_bank_frame_bytes, residency::Pool::BankCount,
                            execution_frame_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  std::uint64_t default_device_budget = 0u;
  std::uint64_t default_host_budget = 0u;
  if (!kernel::checked::mul(execution_frame_bytes, 2u, default_device_budget) ||
      !kernel::checked::mul(draft.program->device->backend == Backend::Cpu
                                ? execution_frame_bytes
                                : draft.host_frame_bytes,
                            2u, default_host_budget)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  const std::uint64_t device_budget = draft.config.device_resident_bytes == 0u
                                          ? default_device_budget
                                          : draft.config.device_resident_bytes;
  const std::uint64_t host_budget = draft.config.host_resident_bytes == 0u
                                        ? default_host_budget
                                        : draft.config.host_resident_bytes;
  const std::uint64_t device_requested_frames =
      std::min({draft.page_count, device_budget / execution_frame_bytes,
                static_cast<std::uint64_t>(PipelineLeafCapacity),
                static_cast<std::uint64_t>(PipelineIterationCapacity)});
  const std::uint64_t host_requested_frames =
      std::min({draft.page_count,
                host_budget / (draft.program->device->backend == Backend::Cpu
                                   ? execution_frame_bytes
                                   : draft.host_frame_bytes),
                static_cast<std::uint64_t>(PipelineLeafCapacity),
                static_cast<std::uint64_t>(PipelineIterationCapacity)});
  const std::uint64_t requested_frames =
      draft.program->device->backend == Backend::Cpu
          ? host_requested_frames
          : std::min(device_requested_frames, host_requested_frames);
  if (requested_frames == 0u) {
    return Status::fail(Reason::PipelineMemoryBudget);
  }
  for (const residency::TiledGraphResourceInput &resource :
       draft.graph_resources) {
    if (!resource.remaps.empty() &&
        !valid_graph_map_shape(resource.remaps, requested_frames,
                               draft.page_count)) {
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  draft.planned = residency::PlanResidency(residency::TiledGraphPlanInput{
      .page_count = draft.page_count,
      .requested_frames = requested_frames,
      .max_frames = std::min(PipelineLeafCapacity, PipelineIterationCapacity),
      .prefetch_distance = draft.prefetch_distance,
      .graph_fingerprint_hi = draft.program->graph_info.fingerprint.hi,
      .graph_fingerprint_lo = draft.program->graph_info.fingerprint.lo,
      .resources = std::move(draft.graph_resources),
      .stages = std::move(draft.graph_stages),
  });
  if (!draft.planned) {
    return Status::fail(draft.planned.failure == residency::Failure::Infeasible
                            ? Reason::PipelineMemoryBudget
                            : Reason::PipelineCapacity);
  }
  draft.frames = draft.planned.plan.tiled_graph().frame_capacity();
  draft.host_frames = draft.frames;
  draft.host_output_frames = draft.frames;
  draft.resident_bytes = 0u;
  draft.host_storage_bytes = 0u;
  if (!kernel::checked::mul(draft.frames, execution_frame_bytes,
                            draft.resident_bytes)) {
    return Status::fail(Reason::PipelineCapacity);
  }
  if (draft.program->device->backend != Backend::Cpu) {
    VirtualHostRingCapacities rings{};
    if (!project_virtual_host_ring_capacities(
            draft.page_count, draft.frames, draft.host_input_page_bytes,
            draft.output_page_bytes, host_budget, PipelineIterationCapacity,
            rings)) {
      return Status::fail(Reason::PipelineMemoryBudget);
    }
    draft.host_frames = rings.input;
    draft.host_output_frames = rings.output;
    draft.host_storage_bytes = rings.storage_bytes;
  }
  if (draft.frames == 0u || draft.host_frames < draft.frames ||
      draft.host_output_frames < draft.frames ||
      draft.frames > std::numeric_limits<std::uint32_t>::max() ||
      draft.host_frames > std::numeric_limits<std::uint32_t>::max()) {
    return Status::fail(Reason::PipelineMemoryBudget);
  }
  return Status::success();
}

} // namespace rund::compute::detail::virtual_graph_prepare_detail
