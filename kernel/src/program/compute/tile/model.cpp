#include "state.hpp"

#include <kernel/program/compute/retention.hpp>

#include "../../../schedule/workspace/local.hpp"

#include <algorithm>
#include <limits>
#include <new>

namespace rund::kernel {
namespace {

[[nodiscard]] u64 RequestedBytes(const std::size_t count,
                                 const u64 width) noexcept {
  constexpr u64 maximum = std::numeric_limits<u64>::max();
  if constexpr (sizeof(std::size_t) > sizeof(u64)) {
    if (count > maximum) {
      return maximum;
    }
  }
  const u64 capacity = static_cast<u64>(count);
  return width != 0u && capacity > maximum / width ? maximum : capacity * width;
}

[[nodiscard]] u64 WorkerStatsBytes(const std::size_t workers) noexcept {
  constexpr u64 width = sizeof(u32) + 3u * sizeof(u64);
  return RequestedBytes(workers, width);
}

[[nodiscard]] bool AddRequestedBytes(const std::size_t count, const u64 width,
                                     u64 &total) noexcept {
  const u64 bytes = RequestedBytes(count, width);
  if (bytes > std::numeric_limits<u64>::max() - total) {
    total = std::numeric_limits<u64>::max();
    return false;
  }
  total += bytes;
  return true;
}

[[nodiscard]] bool
SameStorageMemory(const ComputeTileRetainedMemory &left,
                  const ComputeTileRetainedMemory &right) noexcept {
  return left.state_bytes == right.state_bytes &&
         left.workspace_bytes == right.workspace_bytes &&
         left.failure_slot_bytes == right.failure_slot_bytes &&
         left.worker_tile_bytes == right.worker_tile_bytes &&
         left.async_context_bytes == right.async_context_bytes &&
         left.total_bytes == right.total_bytes;
}

} // namespace

ComputeTileRunStoragePlan
PlanComputeTileRunStorage(const u32 failure_slot_capacity,
                          const u32 worker_capacity) noexcept {
  const ComputeTileRunMemoryPlan retained =
      PlanComputeTileRunMemory(ComputeTileRetainedMemory{
          .state_bytes = sizeof(ComputeTileRunStorage),
          .workspace_bytes = WorkerStatsBytes(worker_capacity),
          .failure_slot_bytes =
              RequestedBytes(failure_slot_capacity, sizeof(const char *)),
          .worker_tile_bytes = RequestedBytes(worker_capacity, sizeof(u32)),
      });
  return ComputeTileRunStoragePlan{
      .failure_slot_capacity = failure_slot_capacity,
      .worker_capacity = worker_capacity,
      .retained = retained,
      .ok = retained.ok,
      .reason = retained.reason,
  };
}

ComputeTileRunStoragePlan MergeComputeTileRunStoragePlans(
    const ComputeTileRunStoragePlan left,
    const ComputeTileRunStoragePlan right) noexcept {
  if (!left.ok || !right.ok) {
    return ComputeTileRunStoragePlan{
        .reason = "compute_tile_run_storage_plan_invalid",
    };
  }
  return PlanComputeTileRunStorage(
      std::max(left.failure_slot_capacity, right.failure_slot_capacity),
      std::max(left.worker_capacity, right.worker_capacity));
}

ComputeTileRunMemoryPlan
MeasureComputeTileRunStorage(const ComputeTileRunStorageView storage) noexcept {
  if (storage.state == nullptr) {
    return ComputeTileRunMemoryPlan{
        .reason = "compute_tile_run_storage_missing",
    };
  }
  u64 workspace_bytes = 0u;
  const bool workspace_valid =
      AddRequestedBytes(storage.worker_stats_partitions.size(), sizeof(u32),
                        workspace_bytes) &&
      AddRequestedBytes(storage.worker_stats_start_offset_ns.size(),
                        sizeof(u64), workspace_bytes) &&
      AddRequestedBytes(storage.worker_stats_elapsed_ns.size(), sizeof(u64),
                        workspace_bytes) &&
      AddRequestedBytes(storage.worker_stats_tail_wait_ns.size(), sizeof(u64),
                        workspace_bytes);
  return PlanComputeTileRunMemory(ComputeTileRetainedMemory{
      .state_bytes = sizeof(ComputeTileRunStorage),
      .workspace_bytes =
          workspace_valid ? workspace_bytes : std::numeric_limits<u64>::max(),
      .failure_slot_bytes =
          RequestedBytes(storage.failure_slots.size(), sizeof(const char *)),
      .worker_tile_bytes =
          RequestedBytes(storage.worker_tiles.size(), sizeof(u32)),
  });
}

bool ComputeTileRunStorage::busy() const noexcept {
  return phase_.load(std::memory_order_acquire) !=
         static_cast<std::uint8_t>(compute_tile_detail::Phase::Idle);
}

u64 ComputeTileRunStorage::generation() const noexcept {
  return generation_.load(std::memory_order_acquire);
}

namespace compute_tile_detail {

bool OwnedRunStorage::allocate(const ComputeTileRunStoragePlan plan) noexcept {
  if (!plan.ok) {
    return false;
  }
  try {
    if (plan.failure_slot_capacity != 0u) {
      failure_slots =
          std::make_unique<const char *[]>(plan.failure_slot_capacity);
    }
    if (plan.worker_capacity != 0u) {
      worker_tiles = std::make_unique<u32[]>(plan.worker_capacity);
      worker_stats_partitions = std::make_unique<u32[]>(plan.worker_capacity);
      worker_stats_start_offset_ns =
          std::make_unique<u64[]>(plan.worker_capacity);
      worker_stats_elapsed_ns = std::make_unique<u64[]>(plan.worker_capacity);
      worker_stats_tail_wait_ns = std::make_unique<u64[]>(plan.worker_capacity);
    }
  } catch (...) {
    return false;
  }
  failure_slot_count = plan.failure_slot_capacity;
  worker_count = plan.worker_capacity;
  return SameStorageMemory(MeasureComputeTileRunStorage(view()).memory,
                           plan.retained.memory);
}

ComputeTileRunStorageView OwnedRunStorage::view() noexcept {
  return ComputeTileRunStorageView{
      .state = &state,
      .failure_slots =
          std::span<const char *>(failure_slots.get(), failure_slot_count),
      .worker_tiles = std::span<u32>(worker_tiles.get(), worker_count),
      .worker_stats_partitions =
          std::span<u32>(worker_stats_partitions.get(), worker_count),
      .worker_stats_start_offset_ns =
          std::span<u64>(worker_stats_start_offset_ns.get(), worker_count),
      .worker_stats_elapsed_ns =
          std::span<u64>(worker_stats_elapsed_ns.get(), worker_count),
      .worker_stats_tail_wait_ns =
          std::span<u64>(worker_stats_tail_wait_ns.get(), worker_count),
  };
}

bool StorageViewSatisfies(const ComputeTileRunStorageView storage,
                          const ComputeTileRunStoragePlan plan) noexcept {
  return plan.ok && storage.state != nullptr &&
         storage.failure_slots.size() >= plan.failure_slot_capacity &&
         storage.worker_tiles.size() >= plan.worker_capacity &&
         storage.worker_stats_partitions.size() >= plan.worker_capacity &&
         storage.worker_stats_start_offset_ns.size() >= plan.worker_capacity &&
         storage.worker_stats_elapsed_ns.size() >= plan.worker_capacity &&
         storage.worker_stats_tail_wait_ns.size() >= plan.worker_capacity;
}

ComputeTileRetainedMemory MeasurePlanMemory(const PlanState &plan) noexcept {
  return PlanComputeTileRunMemory(
             ComputeTileRetainedMemory{
                 .state_bytes = sizeof(PlanState),
                 .workspace_bytes =
                     workspace_detail::WorkspaceRetainedBytes(plan.workspace),
             })
      .memory;
}

} // namespace compute_tile_detail

ComputeTileRunPlan::ComputeTileRunPlan(
    std::shared_ptr<const compute_tile_detail::PlanState> state) noexcept
    : state_(std::move(state)) {}

bool ComputeTileRunPlan::prepared() const noexcept {
  return state_ != nullptr && state_->prepared.valid && state_->storage.ok;
}

u32 ComputeTileRunPlan::count() const noexcept {
  return !prepared() || state_->prepared.units > std::numeric_limits<u32>::max()
             ? 0u
             : static_cast<u32>(state_->prepared.units);
}

u32 ComputeTileRunPlan::tile_count() const noexcept {
  return prepared() ? state_->tile_count : 0u;
}

ComputeTileRunStoragePlan ComputeTileRunPlan::storage_plan() const noexcept {
  return prepared()
             ? state_->storage
             : ComputeTileRunStoragePlan{
                   .retained =
                       ComputeTileRunMemoryPlan{
                           .reason = "compute_tile_run_memory_not_prepared",
                       },
                   .reason = "compute_tile_run_memory_not_prepared",
               };
}

ComputeTileExecutor::ComputeTileExecutor() noexcept = default;

ComputeTileExecutor::ComputeTileExecutor(
    const WorkerBackend worker_backend, const u32 workers,
    const KernelProgramPhysicalTilePolicy tile_policy,
    const Alignment alignment) noexcept
    : configured_backend_(worker_backend), configured_workers_(workers),
      configured_tile_policy_(tile_policy), configured_alignment_(alignment),
      configured_(true) {
  Workspace validation{};
  reason_ =
      executor(validation, worker_backend, workers, alignment, tile_policy)
          .reason;
}

ComputeTileExecutor::ComputeTileExecutor(ComputeTileExecutor &&) noexcept =
    default;

ComputeTileExecutor &
ComputeTileExecutor::operator=(ComputeTileExecutor &&) noexcept = default;

ComputeTileExecutor::~ComputeTileExecutor() = default;

ComputeTileRunPlan ComputeTileExecutor::run_plan() const noexcept {
  return ComputeTileRunPlan{plan_};
}

ComputeTileExecutor ComputeTileExecutor::bind_run(
    const ComputeTileRunStorageView storage) const noexcept {
  return run_plan().bind(storage);
}

ComputeTileExecutor
ComputeTileExecutor::bind_run(const ComputeTileRunStorageView storage,
                              const u32 active_count) const noexcept {
  return run_plan().bind(storage, active_count);
}

bool ComputeTileExecutor::prepared() const noexcept {
  if (plan_ == nullptr || !plan_->prepared.valid || !plan_->storage.ok) {
    return false;
  }
  return storage_ == nullptr ||
         (bound_generation_ != 0u &&
          storage_->generation_.load(std::memory_order_acquire) ==
              bound_generation_);
}

bool ComputeTileExecutor::has_run_storage() const noexcept {
  return storage_ != nullptr && prepared();
}

bool ComputeTileExecutor::borrowed_run_storage() const noexcept {
  return has_run_storage() && owned_storage_ == nullptr;
}

u32 ComputeTileExecutor::count() const noexcept {
  if (!prepared()) {
    return 0u;
  }
  if (storage_ != nullptr) {
    return storage_->active_count_;
  }
  return plan_->prepared.units > std::numeric_limits<u32>::max()
             ? 0u
             : static_cast<u32>(plan_->prepared.units);
}

u32 ComputeTileExecutor::tile_count() const noexcept {
  return !prepared() ? 0u
                     : (storage_ == nullptr ? plan_->tile_count
                                            : storage_->active_tile_count_);
}

ComputeTileRetainedMemory
ComputeTileExecutor::retained_memory() const noexcept {
  if (plan_ == nullptr) {
    return {};
  }
  if (storage_ == nullptr) {
    return compute_tile_detail::MeasurePlanMemory(*plan_);
  }
  if (!prepared()) {
    return {};
  }
  return MeasureComputeTileRunStorage(
             ComputeTileRunStorageView{
                 .state = storage_,
                 .failure_slots = std::span<const char *>(
                     storage_->failures_, storage_->failure_capacity_),
                 .worker_tiles = std::span<u32>(
                     storage_->worker_tiles_, storage_->worker_tile_capacity_),
                 .worker_stats_partitions =
                     std::span<u32>(storage_->worker_stats_partitions_,
                                    storage_->worker_stats_partition_capacity_),
                 .worker_stats_start_offset_ns =
                     std::span<u64>(storage_->worker_stats_start_offset_ns_,
                                    storage_->worker_stats_start_capacity_),
                 .worker_stats_elapsed_ns =
                     std::span<u64>(storage_->worker_stats_elapsed_ns_,
                                    storage_->worker_stats_elapsed_capacity_),
                 .worker_stats_tail_wait_ns =
                     std::span<u64>(storage_->worker_stats_tail_wait_ns_,
                                    storage_->worker_stats_tail_capacity_),
             })
      .memory;
}

} // namespace rund::kernel
