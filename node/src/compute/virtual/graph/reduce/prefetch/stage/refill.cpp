#include "../internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::refill(Ticket &ticket,
                                  bool &cleanup_failed) noexcept {
  if (!parallel_supported()) {
    return Status::success();
  }
  for (std::size_t index = 0u; index < lanes_.size(); ++index) {
    PrefetchLane &selected = lanes_[index];
    if (selected.pending || selected.forecast) {
      continue;
    }
    if (!pool_.prefetch[index].quiescent()) {
      return Status::fail(Reason::PipelineBusy);
    }
    WavefrontCoordinate coordinate{};
    std::uint32_t resource = 0u;
    if (!wavefront_.forecast_middle(ticket.batch, coordinate, resource)) {
      break;
    }
    StageScratch scratch{};
    if (!project_stage_scratch(graph_, run_, pool_, ticket, coordinate.stage,
                               static_cast<std::uint32_t>(run_.frame_capacity),
                               scratch)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    PrefetchLane candidate{};
    const residency::PoolPhysicalOwner *owner = nullptr;
    Status status =
        project_stage(candidate, ticket, scratch, coordinate, resource, owner);
    if (!status || owner == nullptr || candidate.input_index >= input_count_) {
      return status ? Status::fail(Reason::PipelineInvalid) : status;
    }
    // Backing concurrency is a resource bound, independent of the logical
    // graph port or physical worker. Never overlap callbacks on one backing.
    for (const PrefetchLane &active : lanes_) {
      if (active.pending && active.input_index < input_count_ &&
          run_.inputs[active.input_index].backing ==
              run_.inputs[candidate.input_index].backing) {
        return Status::success();
      }
    }
    status = select_missing(candidate, *owner, false);
    if (!status) {
      return status;
    }
    if (candidate.device_only) {
      if (!wavefront_.host_ready(ticket.batch, coordinate.stage, resource)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      continue;
    }
    if (!wavefront_.reserve_forecast(coordinate, resource)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    candidate.physical_lane = static_cast<std::uint32_t>(index);
    selected = std::move(candidate);
    status = issue(index, selected,
                   residency::TiledGraphPort{.resource = resource,
                                             .access = residency::Access::Read},
                   cleanup_failed);
    if (!status) {
      const bool cancelled = wavefront_.cancel_forecast(coordinate, resource);
      cleanup_failed = !cancelled || cleanup_failed;
      return cancelled ? status : Status::fail(Reason::PipelineBusy);
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
