#include "state.hpp"

#include <new>

namespace rund::kernel {
namespace {

[[nodiscard]] WorkspaceReservation
Reservation(const WorkspaceCapacity capacity) noexcept {
  return WorkspaceReservation{
      .schedule_partition_capacity = capacity.schedule_partition_capacity,
      .packet_capacity = capacity.packet_capacity,
      .packet_work_unit_capacity = capacity.packet_work_unit_capacity,
      .ordered_packet_capacity = capacity.ordered_packet_capacity,
      .packet_partition_capacity = capacity.packet_partition_capacity,
      .ordered_packet_scratch_capacity =
          capacity.ordered_packet_scratch_capacity,
      .partition_load_capacity = capacity.partition_load_capacity,
      .partition_count_capacity = capacity.partition_count_capacity,
      .partition_offset_capacity = capacity.partition_offset_capacity,
      .partition_write_offset_capacity =
          capacity.partition_write_offset_capacity,
      .fold_slot_capacity = capacity.fold_slot_capacity,
      .fold_graph_node_capacity = capacity.fold_graph_node_capacity,
      .fold_graph_edge_capacity = capacity.fold_graph_edge_capacity,
      .worker_stats_capacity = capacity.worker_stats_capacity,
  };
}

} // namespace

ComputeTilePrepareResult ComputeTileExecutor::prepare(const u32 count) {
  if (state_ == nullptr) {
    return ComputeTilePrepareResult{
        .reason = "compute_tile_executor_capacity",
        .count = count,
    };
  }
  state_->failures.clear();
  state_->prepared = PreparedEach<1u>{};
  Executor exec = executor(state_->workspace, state_->backend, state_->workers,
                           state_->alignment, state_->tile_policy);
  exec.collect_worker_stats = true;
  if (!exec.valid) {
    state_->reason = exec.reason;
    return ComputeTilePrepareResult{
        .reason = state_->reason,
        .count = count,
        .worker_count = state_->workers,
    };
  }
  state_->prepared = exec.prepare(space(count));
  if (!state_->prepared.valid) {
    state_->reason = state_->prepared.reason;
    return ComputeTilePrepareResult{
        .reason = state_->reason,
        .count = count,
        .worker_count = state_->workers,
    };
  }
  const u32 tiles = state_->prepared.physical_tiling_enabled
                        ? static_cast<u32>(state_->prepared.physical_tile_count)
                        : (count == 0u ? 0u : state_->prepared.partition_count);
  try {
    state_->failures.resize(tiles, nullptr);
    state_->worker_tiles.resize(state_->workers, 0u);
  } catch (...) {
    state_->prepared.valid = false;
    state_->reason = "compute_tile_failure_slots_allocation_failed";
    return ComputeTilePrepareResult{
        .reason = state_->reason,
        .count = count,
        .worker_count = state_->workers,
    };
  }
  state_->reason = "pass";
  return ComputeTilePrepareResult{
      .ok = true,
      .reason = "pass",
      .count = count,
      .worker_count = state_->workers,
      .tile_units = state_->prepared.physical_tiling_enabled
                        ? static_cast<u32>(state_->prepared.physical_tile_units)
                        : count,
      .tile_count = tiles,
  };
}

ComputeTileExecutor ComputeTileExecutor::make_run() const {
  ComputeTileExecutor run{};
  if (!prepared()) {
    return run;
  }
  try {
    run.state_ = std::make_unique<State>();
    run.state_->backend = state_->backend;
    run.state_->workers = state_->workers;
    run.state_->tile_policy = state_->tile_policy;
    run.state_->alignment = state_->alignment;
    run.state_->workspace = state_->workspace;
    if (!ReserveWorkspace(
            run.state_->workspace,
            Reservation(GetWorkspaceCapacity(state_->workspace)))) {
      run.state_.reset();
      return run;
    }
    run.state_->prepared = state_->prepared;
    run.state_->prepared.exec.workspace = &run.state_->workspace;
    run.state_->failures.resize(state_->failures.size(), nullptr);
    run.state_->worker_tiles.resize(state_->worker_tiles.size(), 0u);
    run.state_->async_context =
        std::make_unique<compute_tile_detail::Context>();
    run.state_->reason = state_->reason;
  } catch (...) {
    run.state_.reset();
  }
  return run;
}

} // namespace rund::kernel
