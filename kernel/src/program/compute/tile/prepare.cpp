#include "state.hpp"

#include <kernel/program/build.hpp>

#include <algorithm>
#include <limits>
#include <memory>

namespace rund::kernel {
namespace {

struct ActiveRunShape final {
  u32 tile_count = 0u;
  u32 last_tile_units = 0u;
  bool ok = false;
};

[[nodiscard]] ActiveRunShape
PlanActiveRunShape(const compute_tile_detail::PlanState &plan,
                   const u32 active_count) noexcept {
  if (active_count == 0u) {
    return ActiveRunShape{.ok = true};
  }
  if (plan.prepared.physical_tiling_enabled) {
    const u64 units = plan.prepared.physical_tile_units;
    if (units == 0u) {
      return {};
    }
    const u64 tiles = (static_cast<u64>(active_count) - 1u) / units + 1u;
    if (tiles > std::numeric_limits<u32>::max()) {
      return {};
    }
    const u64 last = static_cast<u64>(active_count) - (tiles - 1u) * units;
    return ActiveRunShape{
        .tile_count = static_cast<u32>(tiles),
        .last_tile_units = static_cast<u32>(last),
        .ok = true,
    };
  }

  const ScheduleView schedule = plan.workspace.program.schedule;
  u32 tiles = 0u;
  u32 last = 0u;
  for (u32 index = 0u; index < schedule.partition_count; ++index) {
    const Partition &partition = schedule.partitions[index];
    if (partition.begin >= active_count) {
      continue;
    }
    const u32 end = std::min(partition.end, active_count);
    if (partition.begin < end) {
      ++tiles;
      last = end - partition.begin;
    }
  }
  return ActiveRunShape{
      .tile_count = tiles,
      .last_tile_units = last,
      .ok = tiles != 0u,
  };
}

} // namespace

ComputeTilePrepareResult ComputeTileExecutor::prepare(const u32 count) {
  plan_.reset();
  if (!configured_ || storage_ != nullptr) {
    reason_ = storage_ == nullptr ? "compute_tile_executor_capacity"
                                  : "compute_tile_plan_bound";
    return ComputeTilePrepareResult{
        .reason = reason_,
        .count = count,
        .worker_count = configured_workers_,
    };
  }

  std::shared_ptr<compute_tile_detail::PlanState> candidate{};
  try {
    candidate = std::make_shared<compute_tile_detail::PlanState>();
  } catch (...) {
    reason_ = "compute_tile_executor_capacity";
    return ComputeTilePrepareResult{
        .reason = reason_,
        .count = count,
        .worker_count = configured_workers_,
    };
  }
  candidate->backend = configured_backend_;
  candidate->workers = configured_workers_;
  candidate->tile_policy = configured_tile_policy_;
  candidate->alignment = configured_alignment_;

  Executor exec =
      executor(candidate->workspace, candidate->backend, candidate->workers,
               candidate->alignment, candidate->tile_policy);
  exec.collect_worker_stats = true;
  if (!exec.valid) {
    reason_ = exec.reason;
    return ComputeTilePrepareResult{
        .reason = reason_,
        .count = count,
        .worker_count = candidate->workers,
    };
  }
  candidate->prepared = exec.prepare(space(count));
  if (!candidate->prepared.valid) {
    reason_ = candidate->prepared.reason;
    return ComputeTilePrepareResult{
        .reason = reason_,
        .count = count,
        .worker_count = candidate->workers,
    };
  }

  candidate->tile_count =
      candidate->prepared.physical_tiling_enabled
          ? static_cast<u32>(candidate->prepared.physical_tile_count)
          : (count == 0u ? 0u : candidate->prepared.partition_count);
  candidate->storage =
      PlanComputeTileRunStorage(candidate->tile_count, candidate->workers);
  if (!candidate->storage.ok) {
    reason_ = candidate->storage.reason;
    return ComputeTilePrepareResult{
        .reason = reason_,
        .count = count,
        .worker_count = candidate->workers,
    };
  }

  candidate->reason = "pass";
  plan_ = std::move(candidate);
  reason_ = "pass";
  return ComputeTilePrepareResult{
      .ok = true,
      .reason = "pass",
      .count = count,
      .worker_count = plan_->workers,
      .tile_units = plan_->prepared.physical_tiling_enabled
                        ? static_cast<u32>(plan_->prepared.physical_tile_units)
                        : count,
      .tile_count = plan_->tile_count,
  };
}

ComputeTileExecutor ComputeTileRunPlan::bind(
    const ComputeTileRunStorageView storage_view) const noexcept {
  return bind(storage_view, count());
}

ComputeTileExecutor
ComputeTileRunPlan::bind(const ComputeTileRunStorageView storage_view,
                         const u32 active_count) const noexcept {
  ComputeTileExecutor run{};
  if (!prepared()) {
    run.reason_ = "compute_tile_run_memory_not_prepared";
    return run;
  }
  if (active_count > count()) {
    run.reason_ = "compute_tile_run_active_count_invalid";
    return run;
  }
  const ActiveRunShape active = PlanActiveRunShape(*state_, active_count);
  if (!active.ok) {
    run.reason_ = "compute_tile_run_active_count_invalid";
    return run;
  }
  if (!compute_tile_detail::StorageViewSatisfies(storage_view,
                                                 state_->storage)) {
    run.reason_ = storage_view.state == nullptr
                      ? "compute_tile_run_storage_missing"
                      : "compute_tile_run_storage_capacity";
    return run;
  }

  ComputeTileRunStorage &storage = *storage_view.state;
  std::uint8_t expected =
      static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle);
  if (!storage.phase_.compare_exchange_strong(
          expected,
          static_cast<std::uint8_t>(compute_tile_detail::Phase::Binding),
          std::memory_order_acq_rel, std::memory_order_acquire)) {
    run.reason_ = "compute_tile_run_busy";
    return run;
  }

  const u64 current = storage.generation_.load(std::memory_order_relaxed);
  if (current == std::numeric_limits<u64>::max()) {
    storage.phase_.store(
        static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle),
        std::memory_order_release);
    run.reason_ = "compute_tile_run_generation_overflow";
    return run;
  }

  ResetWorkspace(storage.workspace_);
  storage.workspace_.program = state_->workspace.program;
  storage.workspace_.program.exec_kernel.physical_tile_count =
      active.tile_count;
  storage.prepared_ = state_->prepared;
  storage.prepared_.exec.workspace = &storage.workspace_;
  storage.prepared_.index_space = space(active_count);
  storage.prepared_.units = active_count;
  storage.prepared_.partition_count = active.tile_count;
  if (storage.prepared_.physical_tiling_enabled) {
    storage.prepared_.physical_tile_count = active.tile_count;
  }
  storage.plan_ = state_;
  storage.failures_ = storage_view.failure_slots.data();
  storage.failure_capacity_ = storage_view.failure_slots.size();
  storage.failure_count_ = state_->storage.failure_slot_capacity;
  storage.worker_tiles_ = storage_view.worker_tiles.data();
  storage.worker_tile_capacity_ = storage_view.worker_tiles.size();
  storage.worker_count_ = state_->storage.worker_capacity;
  storage.active_count_ = active_count;
  storage.active_tile_count_ = active.tile_count;
  storage.active_last_tile_units_ = active.last_tile_units;
  storage.worker_stats_partitions_ =
      storage_view.worker_stats_partitions.data();
  storage.worker_stats_partition_capacity_ =
      storage_view.worker_stats_partitions.size();
  storage.worker_stats_start_offset_ns_ =
      storage_view.worker_stats_start_offset_ns.data();
  storage.worker_stats_start_capacity_ =
      storage_view.worker_stats_start_offset_ns.size();
  storage.worker_stats_elapsed_ns_ =
      storage_view.worker_stats_elapsed_ns.data();
  storage.worker_stats_elapsed_capacity_ =
      storage_view.worker_stats_elapsed_ns.size();
  storage.worker_stats_tail_wait_ns_ =
      storage_view.worker_stats_tail_wait_ns.data();
  storage.worker_stats_tail_capacity_ =
      storage_view.worker_stats_tail_wait_ns.size();
  storage.callback_context_ = nullptr;
  storage.callback_ = nullptr;
  storage.ready_context_ = nullptr;
  storage.ready_ = nullptr;
  storage.async_ok_ = false;
  storage.submission_.remaining.store(0u, std::memory_order_relaxed);
  storage.submission_.failed.store(false, std::memory_order_relaxed);
  storage.first_failure_.store(kNoComputeTileFailure,
                               std::memory_order_relaxed);
  storage.completed_.store(0u, std::memory_order_relaxed);
  std::fill_n(storage.failures_, storage.failure_count_, nullptr);
  std::fill_n(storage.worker_tiles_, storage.worker_count_, 0u);
  std::fill_n(storage.worker_stats_partitions_, storage.worker_count_, 0u);
  std::fill_n(storage.worker_stats_start_offset_ns_, storage.worker_count_, 0u);
  std::fill_n(storage.worker_stats_elapsed_ns_, storage.worker_count_, 0u);
  std::fill_n(storage.worker_stats_tail_wait_ns_, storage.worker_count_, 0u);

  const u64 generation = current + 1u;
  storage.generation_.store(generation, std::memory_order_release);
  storage.phase_.store(
      static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle),
      std::memory_order_release);

  run.plan_ = state_;
  run.storage_ = &storage;
  run.bound_generation_ = generation;
  run.reason_ = "pass";
  return run;
}

ComputeTileExecutor ComputeTileRunPlan::make_run() const {
  ComputeTileExecutor run{};
  if (!prepared()) {
    run.reason_ = "compute_tile_run_memory_not_prepared";
    return run;
  }
  std::unique_ptr<compute_tile_detail::OwnedRunStorage> owned{};
  try {
    owned = std::make_unique<compute_tile_detail::OwnedRunStorage>();
  } catch (...) {
    run.reason_ = "compute_tile_run_storage_allocation_failed";
    return run;
  }
  if (!owned->allocate(state_->storage)) {
    run.reason_ = "compute_tile_run_storage_allocation_failed";
    return run;
  }
  run = bind(owned->view());
  if (!run.has_run_storage()) {
    return run;
  }
  run.owned_storage_ = std::move(owned);
  return run;
}

ComputeTileExecutor ComputeTileExecutor::make_run() const {
  return run_plan().make_run();
}

} // namespace rund::kernel
