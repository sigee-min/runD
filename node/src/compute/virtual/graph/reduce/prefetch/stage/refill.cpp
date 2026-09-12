#include "../internal.hpp"

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::refill(Ticket &ticket,
                                  bool &cleanup_failed) noexcept {
  if (!parallel_supported()) {
    return Status::success();
  }
  if (ticket.host_ready_count > ticket.forecast_resources.size()) {
    return Status::fail(Reason::PipelineInvalid);
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
    std::uint32_t after_stage = 0u;
    PrefetchLane candidate{};
    const residency::PoolPhysicalOwner *owner = nullptr;
    bool found = false;
    // Scan only the fixed middle-stage set. An unavailable backing must not
    // hide a later independent input, and a pinned Ready row must be consumed
    // before another stage can reuse that input's physical Host region.
    while (wavefront_.forecast_middle(ticket.batch, coordinate, resource,
                                      after_stage)) {
      StageScratch scratch = middle_scratch(ticket);
      if (!project_stage_scratch(
              graph_, run_, pool_, ticket, coordinate.stage,
              static_cast<std::uint32_t>(run_.frame_capacity), scratch)) {
        return Status::fail(Reason::PipelineInvalid);
      }
      const Status status = project_stage(candidate, ticket, scratch,
                                          coordinate, resource, owner);
      if (!status || owner == nullptr ||
          candidate.input_index >= input_count_) {
        return status ? Status::fail(Reason::PipelineInvalid) : status;
      }
      const bool callback_live = std::any_of(
          lanes_.begin(), lanes_.end(), [&](const PrefetchLane &active) {
            return active.pending && active.input_index < input_count_ &&
                   run_.inputs[active.input_index].backing ==
                       run_.inputs[candidate.input_index].backing;
          });
      const auto ready_end =
          ticket.forecast_resources.begin() +
          static_cast<std::ptrdiff_t>(ticket.host_ready_count);
      const bool ready_pinned = std::find(ticket.forecast_resources.begin(),
                                          ready_end, resource) != ready_end;
      if (callback_live || ready_pinned) {
        after_stage = coordinate.stage;
        continue;
      }
      found = true;
      break;
    }
    if (!found) {
      break;
    }
    Status status = select_missing(candidate, *owner, false);
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
