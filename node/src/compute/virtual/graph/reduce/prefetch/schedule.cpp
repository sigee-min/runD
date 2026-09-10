#include "internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::schedule(const std::uint64_t batch,
                                    const bool speculative,
                                    bool &cleanup_failed) noexcept {
  if (graph_.stages().empty()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::span<const residency::TiledGraphStage> stages = graph_.stages();
  for (const residency::TiledGraphPort port : stages.front().ports) {
    const residency::TiledGraphResource *const resource =
        graph_.resource(port.resource);
    std::size_t input_index = 0u;
    if (residency::reads(port.access) && resource != nullptr &&
        resource->kind == residency::GraphResourceKind::ExternalInput &&
        resource->persistence == residency::ResourcePersistence::Backing &&
        find_input(port.resource, input_index)) {
      return schedule_input(batch, input_index, speculative, cleanup_failed);
    }
  }
  return Status::fail(Reason::PipelineInvalid);
}

Status PrefetchController::schedule_input(const std::uint64_t batch,
                                          const std::size_t input_index,
                                          const bool speculative,
                                          bool &cleanup_failed) noexcept {
  if (state_.pipeline->device->backend == Backend::Cpu) {
    return Status::success();
  }
  PrefetchLane &selected = lane(batch);
  if (selected.pending) {
    return selected.batch == batch && selected.input_index == input_index
               ? Status::success()
               : Status::fail(Reason::PipelineBusy);
  }
  const std::size_t selected_index = lane_index(batch);
  if (!pool_.prefetch[selected_index].quiescent()) {
    return Status::fail(Reason::PipelineBusy);
  }

  residency::TiledGraphPort input_port{};
  const residency::PoolPhysicalOwner *input_owner = nullptr;
  Status status = project(selected, batch, input_index, speculative, input_port,
                          input_owner);
  selected.physical_lane = static_cast<std::uint32_t>(selected_index);
  if (status) {
    status = select_missing(selected, *input_owner, speculative);
  }
  if (!status) {
    selected = {};
    return status;
  }
  if (selected.device_only) {
    selected.pending = true;
    return Status::success();
  }
  return issue(selected_index, selected, input_port, cleanup_failed);
}

} // namespace rund::compute::detail::graph_reduce
