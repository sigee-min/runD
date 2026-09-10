#include "../internal.hpp"

#include "../../../../backing.hpp"
#include "../../../../run/backing.hpp"
#include "../../transfer.hpp"

#include <kernel/core/checked.hpp>
#include <rund/counter.hpp>

namespace rund::compute::detail::graph_reduce {

Status PrefetchController::start_pair(Ticket &ticket,
                                      bool &cleanup_failed) noexcept {
  if (!pair_supported() || state_.pipeline->device->backend == Backend::Cpu ||
      ticket.host_ready_count != 0u) {
    return Status::fail(Reason::BackendUnsupported);
  }
  for (std::size_t index = 0u; index < lanes_.size(); ++index) {
    if (lanes_[index].pending || !pool_.prefetch[index].quiescent()) {
      return Status::fail(Reason::PipelineBusy);
    }
  }

  // The entire pair is projected and probed into local lanes before the first
  // Forecast reservation. A topology/cache mismatch therefore leaves both
  // planner masks, workers, and Authority capabilities untouched; the caller
  // can fall back to the established serial Host path.
  std::array<WavefrontCoordinate, LaneCount> coordinates{};
  std::array<std::uint32_t, LaneCount> resources{};
  if (!wavefront_.pair_forecast(ticket.batch, coordinates, resources)) {
    return Status::fail(Reason::BackendUnsupported);
  }

  std::array<PrefetchLane, LaneCount> candidates{};
  std::array<const residency::PoolPhysicalOwner *, LaneCount> owners{};
  for (std::size_t index = 0u; index < LaneCount; ++index) {
    const WavefrontCoordinate coordinate = coordinates[index];
    const std::uint32_t resource = resources[index];
    if (coordinate.batch != ticket.batch || coordinate.stage != index + 1u ||
        coordinate.page_count == 0u || resource == 0u) {
      return Status::fail(Reason::BackendUnsupported);
    }
    StageScratch scratch{};
    if (!project_stage_scratch(graph_, run_, pool_, ticket, coordinate.stage,
                               static_cast<std::uint32_t>(run_.frame_capacity),
                               scratch)) {
      return Status::fail(Reason::BackendUnsupported);
    }
    const residency::TiledGraphResource *const declared =
        graph_.resource(resource);
    const residency::PoolPhysicalOwner *const declared_owner =
        declared == nullptr ? nullptr
                            : pool_.graph_owner(declared->physical_id);
    if (declared_owner == nullptr || declared_owner->arena == nullptr) {
      return Status::fail(Reason::BackendUnsupported);
    }
    const residency::PoolPhysicalOwner *projected_owner = nullptr;
    Status status = project_stage(candidates[index], ticket, scratch,
                                  coordinate, resource, projected_owner);
    if (status) {
      status = select_missing(candidates[index], *declared_owner, false);
    }
    if (!status || projected_owner == nullptr ||
        projected_owner != declared_owner || candidates[index].device_only ||
        candidates[index].count != ticket.count || ticket.count == 0u ||
        candidates[index].input_index >= input_count_ ||
        candidates[index].batch != ticket.batch ||
        candidates[index].stage != coordinate.stage ||
        candidates[index].resource != resource ||
        candidates[index].pages.first_page != ticket.pages.first_page ||
        candidates[index].pages.page_count != ticket.pages.page_count ||
        candidates[index].epoch.ordinal != coordinate.ordinal) {
      return Status::fail(Reason::BackendUnsupported);
    }
    for (std::size_t page = 0u; page < candidates[index].count; ++page) {
      if (candidates[index].sources[page].key.resource != resource) {
        return Status::fail(Reason::BackendUnsupported);
      }
    }
    candidates[index].physical_lane = static_cast<std::uint32_t>(index);
    owners[index] = projected_owner;
  }
  if (candidates[0u].input_index == candidates[1u].input_index ||
      owners[0u] == owners[1u] ||
      candidates[0u].resource == candidates[1u].resource) {
    return Status::fail(Reason::BackendUnsupported);
  }
  for (std::size_t page = 0u; page < ticket.count; ++page) {
    if (candidates[0u].sources[page].key == candidates[1u].sources[page].key) {
      return Status::fail(Reason::BackendUnsupported);
    }
  }

  std::size_t reserved = 0u;
  const auto rollback = [&](const Status status) noexcept {
    bool clean = true;
    for (std::size_t index = 0u; index < reserved; ++index) {
      PrefetchLane &selected = lanes_[index];
      bool released = true;
      if (selected.pending && !selected.device_only) {
        released = retire(index, selected);
        clean = released && clean;
      }
      clean =
          wavefront_.cancel_forecast(coordinates[index], resources[index]) &&
          clean;
      if (released && !selected.forecast) {
        lanes_[index] = {};
      }
    }
    cleanup_failed = !clean || cleanup_failed;
    return clean ? status : Status::fail(Reason::PipelineBusy);
  };

  for (std::size_t index = 0u; index < LaneCount; ++index) {
    if (!wavefront_.reserve_forecast(coordinates[index], resources[index])) {
      return rollback(Status::fail(Reason::BackendUnsupported));
    }
    ++reserved;
    lanes_[index] = std::move(candidates[index]);
    Status status =
        issue(index, lanes_[index],
              residency::TiledGraphPort{.resource = resources[index],
                                        .access = residency::Access::Read},
              cleanup_failed);
    if (!status) {
      return rollback(status);
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail::graph_reduce
