#include "state.hpp"

#include <kernel/program/compute/retention.hpp>

#include <new>

namespace rund::kernel {
namespace {

[[nodiscard]] u64 WorkspaceBytes(const Workspace &workspace) noexcept {
  using compute_retained_detail::Add;
  using compute_retained_detail::VectorCapacityBytes;
  u64 bytes = VectorCapacityBytes(workspace.schedule.partitions);
  bytes = Add(bytes, VectorCapacityBytes(workspace.fold_slots.values));
  bytes = Add(bytes,
              VectorCapacityBytes(workspace.fold_graph.partition_fold_slots));
  bytes = Add(bytes, VectorCapacityBytes(workspace.fold_graph.nodes));
  bytes = Add(bytes, VectorCapacityBytes(workspace.fold_graph.reduction_edges));
  bytes = Add(bytes, VectorCapacityBytes(workspace.packet_work_units));
  bytes = Add(bytes, VectorCapacityBytes(workspace.ordered_packet_indices));
  bytes = Add(bytes, VectorCapacityBytes(workspace.packet_partition_indices));
  bytes = Add(bytes, VectorCapacityBytes(workspace.ordered_packet_scratch));
  bytes = Add(bytes, VectorCapacityBytes(workspace.partition_loads));
  bytes = Add(bytes, VectorCapacityBytes(workspace.partition_counts));
  bytes = Add(bytes, VectorCapacityBytes(workspace.partition_offsets));
  bytes = Add(bytes, VectorCapacityBytes(workspace.partition_write_offsets));
  bytes = Add(
      bytes, VectorCapacityBytes(workspace.worker_stats_partitions_per_worker));
  bytes =
      Add(bytes, VectorCapacityBytes(workspace.worker_stats_start_offset_ns));
  bytes = Add(bytes, VectorCapacityBytes(workspace.worker_stats_elapsed_ns));
  return Add(bytes, VectorCapacityBytes(workspace.worker_stats_tail_wait_ns));
}

} // namespace

ComputeTileExecutor::ComputeTileExecutor() noexcept = default;

ComputeTileExecutor::ComputeTileExecutor(
    const WorkerBackend worker_backend, const u32 workers,
    const KernelProgramPhysicalTilePolicy tile_policy,
    const Alignment alignment) noexcept
    : state_(new (std::nothrow) State{}) {
  if (state_ == nullptr) {
    return;
  }
  state_->backend = worker_backend;
  state_->workers = workers;
  state_->tile_policy = tile_policy;
  state_->alignment = alignment;
  const Executor exec = executor(state_->workspace, worker_backend, workers,
                                 alignment, tile_policy);
  state_->reason = exec.reason;
}

ComputeTileExecutor::ComputeTileExecutor(ComputeTileExecutor &&) noexcept =
    default;

ComputeTileExecutor &
ComputeTileExecutor::operator=(ComputeTileExecutor &&) noexcept = default;

ComputeTileExecutor::~ComputeTileExecutor() = default;

bool ComputeTileExecutor::prepared() const noexcept {
  return state_ != nullptr && state_->prepared.valid;
}

u32 ComputeTileExecutor::count() const noexcept {
  return !prepared() || state_->prepared.units > std::numeric_limits<u32>::max()
             ? 0u
             : static_cast<u32>(state_->prepared.units);
}

u32 ComputeTileExecutor::tile_count() const noexcept {
  return state_ == nullptr ? 0u : static_cast<u32>(state_->failures.size());
}

ComputeTileRetainedMemory
ComputeTileExecutor::retained_memory() const noexcept {
  if (state_ == nullptr) {
    return {};
  }
  ComputeTileRetainedMemory memory{
      .state_bytes = sizeof(State),
      .workspace_bytes = WorkspaceBytes(state_->workspace),
      .failure_slot_bytes =
          compute_retained_detail::VectorCapacityBytes(state_->failures),
      .worker_tile_bytes =
          compute_retained_detail::VectorCapacityBytes(state_->worker_tiles),
      .async_context_bytes = state_->async_context == nullptr
                                 ? 0u
                                 : sizeof(compute_tile_detail::Context),
  };
  memory.total_bytes =
      compute_retained_detail::Add(memory.state_bytes, memory.workspace_bytes);
  memory.total_bytes = compute_retained_detail::Add(memory.total_bytes,
                                                    memory.failure_slot_bytes);
  memory.total_bytes = compute_retained_detail::Add(memory.total_bytes,
                                                    memory.worker_tile_bytes);
  memory.total_bytes = compute_retained_detail::Add(memory.total_bytes,
                                                    memory.async_context_bytes);
  return memory;
}

} // namespace rund::kernel
