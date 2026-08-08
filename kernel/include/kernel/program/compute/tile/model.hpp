#pragma once

#include <kernel/dispatch/worker/backend.hpp>
#include <kernel/program/executor/model.hpp>
#include <kernel/program/executor/prepare/state.hpp>
#include <kernel/program/skeleton/model.hpp>
#include <kernel/program/tile.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>

namespace rund::kernel {

inline constexpr u32 kNoComputeTileFailure = std::numeric_limits<u32>::max();

struct ComputeTile {
  u32 index = 0u;
  u32 worker_index = 0u;
  u32 begin = 0u;
  u32 end = 0u;

  [[nodiscard]] u32 size() const noexcept {
    return end > begin ? end - begin : 0u;
  }
};

struct ComputeTileCallbackResult {
  bool ok = true;
  const char *reason = "pass";
};

struct ComputeTilePrepareResult {
  bool ok = false;
  const char *reason = "compute_tile_not_prepared";
  u32 count = 0u;
  u32 worker_count = 0u;
  u32 tile_units = 0u;
  u32 tile_count = 0u;
};

struct ComputeTileRunResult {
  bool ok = false;
  const char *reason = "compute_tile_not_run";
  u32 first_failure_tile = kNoComputeTileFailure;
  u32 completed_tile_count = 0u;
  u32 worker_count = 0u;
  u32 participating_workers = 0u;
  u32 worker_tile_count = 0u;
  u32 min_tiles_per_worker = 0u;
  u32 max_tiles_per_worker = 0u;
  u32 backend_dispatch_count = 0u;
  u32 last_tile_units = 0u;
};

struct ComputeTileSubmitResult {
  bool ok = false;
  const char *reason = "compute_tile_not_submitted";
};

struct ComputeTileRetainedMemory {
  u64 state_bytes = 0u;
  u64 workspace_bytes = 0u;
  u64 failure_slot_bytes = 0u;
  u64 worker_tile_bytes = 0u;
  u64 async_context_bytes = 0u;
  u64 total_bytes = 0u;
};

struct ComputeTileRunMemoryPlan {
  ComputeTileRetainedMemory memory{};
  bool ok = false;
  const char *reason = "compute_tile_run_memory_not_planned";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

[[nodiscard]] constexpr ComputeTileRunMemoryPlan
PlanComputeTileRunMemory(ComputeTileRetainedMemory memory) noexcept {
  constexpr u64 maximum = std::numeric_limits<u64>::max();
  memory.total_bytes = 0u;
  const auto add = [&](const u64 bytes) constexpr {
    if (bytes > maximum - memory.total_bytes) {
      memory.total_bytes = maximum;
      return false;
    }
    memory.total_bytes += bytes;
    return true;
  };
  if (!add(memory.state_bytes) || !add(memory.workspace_bytes) ||
      !add(memory.failure_slot_bytes) || !add(memory.worker_tile_bytes) ||
      !add(memory.async_context_bytes)) {
    return ComputeTileRunMemoryPlan{
        .memory = memory,
        .reason = "compute_tile_run_memory_overflow",
    };
  }
  return ComputeTileRunMemoryPlan{
      .memory = memory,
      .ok = true,
      .reason = "pass",
  };
}

struct ComputeTileRunStoragePlan {
  u32 failure_slot_capacity = 0u;
  u32 worker_capacity = 0u;
  ComputeTileRunMemoryPlan retained{};
  bool ok = false;
  const char *reason = "compute_tile_run_storage_not_planned";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

using ComputeTileCallback =
    ComputeTileCallbackResult (*)(const void *context, const ComputeTile &tile);
using ComputeTileReady = void (*)(void *context) noexcept;

class ComputeTileExecutor;
class ComputeTileRunPlan;
class ComputeTileRunStorage;

namespace compute_tile_detail {

struct PlanState;
struct OwnedRunStorage;

void Begin(ComputeTileRunStorage &storage, const void *callback_context,
           ComputeTileCallback callback) noexcept;
void Invoke(void *context, const Partition &partition) noexcept;
void InvokeWorker(void *context, const Partition &partition) noexcept;
[[nodiscard]] ComputeTileRunResult
Project(const ComputeTileRunStorage &storage, bool ok, const char *reason,
        u32 worker_count, u32 dispatch_count,
        bool report_tail_on_failure) noexcept;

} // namespace compute_tile_detail

// Fixed-address control state for one active tile run. The caller may place
// this typed object in a sealed arena and supply the variable extents through
// ComputeTileRunStorageView. It is deliberately non-copyable and non-movable:
// worker callbacks retain its address until synchronous return or finish().
class ComputeTileRunStorage final {
public:
  ComputeTileRunStorage() noexcept = default;
  ComputeTileRunStorage(const ComputeTileRunStorage &) = delete;
  ComputeTileRunStorage &operator=(const ComputeTileRunStorage &) = delete;
  ComputeTileRunStorage(ComputeTileRunStorage &&) = delete;
  ComputeTileRunStorage &operator=(ComputeTileRunStorage &&) = delete;
  ~ComputeTileRunStorage() = default;

  [[nodiscard]] bool busy() const noexcept;
  [[nodiscard]] u64 generation() const noexcept;

private:
  friend class ComputeTileExecutor;
  friend class ComputeTileRunPlan;
  friend void compute_tile_detail::Begin(ComputeTileRunStorage &, const void *,
                                         ComputeTileCallback) noexcept;
  friend void compute_tile_detail::Invoke(void *, const Partition &) noexcept;
  friend void compute_tile_detail::InvokeWorker(void *,
                                                const Partition &) noexcept;
  friend ComputeTileRunResult
  compute_tile_detail::Project(const ComputeTileRunStorage &, bool,
                               const char *, u32, u32, bool) noexcept;

  Workspace workspace_{};
  PreparedEach<1u> prepared_{};
  std::shared_ptr<const compute_tile_detail::PlanState> plan_{};
  WorkerSubmission submission_{};
  const void *callback_context_ = nullptr;
  ComputeTileCallback callback_ = nullptr;
  const char **failures_ = nullptr;
  std::size_t failure_capacity_ = 0u;
  u32 failure_count_ = 0u;
  u32 *worker_tiles_ = nullptr;
  std::size_t worker_tile_capacity_ = 0u;
  u32 worker_count_ = 0u;
  u32 active_count_ = 0u;
  u32 active_tile_count_ = 0u;
  u32 active_last_tile_units_ = 0u;
  u32 *worker_stats_partitions_ = nullptr;
  std::size_t worker_stats_partition_capacity_ = 0u;
  u64 *worker_stats_start_offset_ns_ = nullptr;
  std::size_t worker_stats_start_capacity_ = 0u;
  u64 *worker_stats_elapsed_ns_ = nullptr;
  std::size_t worker_stats_elapsed_capacity_ = 0u;
  u64 *worker_stats_tail_wait_ns_ = nullptr;
  std::size_t worker_stats_tail_capacity_ = 0u;
  std::atomic<u32> first_failure_{kNoComputeTileFailure};
  std::atomic<u32> completed_{0u};
  std::atomic<u64> generation_{0u};
  std::atomic<std::uint8_t> phase_{0u};
  void *ready_context_ = nullptr;
  ComputeTileReady ready_ = nullptr;
  bool async_ok_ = false;
};

// Typed, caller-owned variable storage for ComputeTileRunStorage. Every span
// must remain valid until the bound executor is idle and no ready callback can
// still observe it. A larger max-envelope view may be rebound serially to any
// plan whose storage_plan() fits component by component.
struct ComputeTileRunStorageView {
  ComputeTileRunStorage *state = nullptr;
  std::span<const char *> failure_slots{};
  std::span<u32> worker_tiles{};
  std::span<u32> worker_stats_partitions{};
  std::span<u64> worker_stats_start_offset_ns{};
  std::span<u64> worker_stats_elapsed_ns{};
  std::span<u64> worker_stats_tail_wait_ns{};
};

[[nodiscard]] ComputeTileRunStoragePlan
PlanComputeTileRunStorage(u32 failure_slot_capacity,
                          u32 worker_capacity) noexcept;
[[nodiscard]] ComputeTileRunStoragePlan
MergeComputeTileRunStoragePlans(ComputeTileRunStoragePlan left,
                                ComputeTileRunStoragePlan right) noexcept;
[[nodiscard]] ComputeTileRunMemoryPlan
MeasureComputeTileRunStorage(ComputeTileRunStorageView storage) noexcept;

// Immutable, lifetime-safe authority published by a successful prepare().
// Re-preparing the source executor publishes a new state; existing handles and
// already-bound runs continue to reference their original frozen plan.
class ComputeTileRunPlan final {
public:
  ComputeTileRunPlan() noexcept = default;

  [[nodiscard]] bool prepared() const noexcept;
  [[nodiscard]] u32 count() const noexcept;
  [[nodiscard]] u32 tile_count() const noexcept;
  [[nodiscard]] ComputeTileRunStoragePlan storage_plan() const noexcept;
  [[nodiscard]] ComputeTileExecutor
  bind(ComputeTileRunStorageView storage) const noexcept;
  [[nodiscard]] ComputeTileExecutor bind(ComputeTileRunStorageView storage,
                                         u32 active_count) const noexcept;
  [[nodiscard]] ComputeTileExecutor make_run() const;

private:
  friend class ComputeTileExecutor;
  explicit ComputeTileRunPlan(
      std::shared_ptr<const compute_tile_detail::PlanState> state) noexcept;

  std::shared_ptr<const compute_tile_detail::PlanState> state_{};
};

class ComputeTileExecutor {
public:
  ComputeTileExecutor() noexcept;
  explicit ComputeTileExecutor(
      WorkerBackend worker_backend, u32 workers,
      KernelProgramPhysicalTilePolicy physical_tile_policy =
          KernelProgramPhysicalTilePolicy{},
      Alignment boundary_alignment = Alignment{}) noexcept;
  ComputeTileExecutor(const ComputeTileExecutor &) = delete;
  ComputeTileExecutor &operator=(const ComputeTileExecutor &) = delete;
  ComputeTileExecutor(ComputeTileExecutor &&other) noexcept;
  ComputeTileExecutor &operator=(ComputeTileExecutor &&other) noexcept;
  ~ComputeTileExecutor();

  [[nodiscard]] ComputeTilePrepareResult prepare(u32 count);
  [[nodiscard]] ComputeTileRunPlan run_plan() const noexcept;
  [[nodiscard]] ComputeTileExecutor
  bind_run(ComputeTileRunStorageView storage) const noexcept;
  [[nodiscard]] ComputeTileExecutor bind_run(ComputeTileRunStorageView storage,
                                             u32 active_count) const noexcept;
  [[nodiscard]] ComputeTileExecutor make_run() const;

  template <typename Callback>
  [[nodiscard]] ComputeTileRunResult run(Callback &&callback);

  template <typename Callback>
  [[nodiscard]] ComputeTileRunResult run_with(WorkerBackend worker_backend,
                                              Callback &&callback);

  [[nodiscard]] bool prepared() const noexcept;
  [[nodiscard]] bool has_run_storage() const noexcept;
  [[nodiscard]] bool borrowed_run_storage() const noexcept;
  [[nodiscard]] u32 count() const noexcept;
  [[nodiscard]] u32 tile_count() const noexcept;
  [[nodiscard]] ComputeTileRetainedMemory retained_memory() const noexcept;

  [[nodiscard]] ComputeTileSubmitResult
  submit_with_erased(WorkerBackend worker_backend, const void *callback_context,
                     ComputeTileCallback callback, void *ready_context,
                     ComputeTileReady ready) noexcept;
  [[nodiscard]] ComputeTileRunResult finish() noexcept;

private:
  friend class ComputeTileRunPlan;

  [[nodiscard]] ComputeTileRunResult run_erased(const void *callback_context,
                                                ComputeTileCallback callback);
  [[nodiscard]] ComputeTileRunResult
  run_with_erased(WorkerBackend worker_backend, const void *callback_context,
                  ComputeTileCallback callback);

  static void CompleteSubmission(void *context, bool ok) noexcept;

  WorkerBackend configured_backend_{};
  u32 configured_workers_ = 0u;
  KernelProgramPhysicalTilePolicy configured_tile_policy_{};
  Alignment configured_alignment_{};
  std::shared_ptr<const compute_tile_detail::PlanState> plan_{};
  ComputeTileRunStorage *storage_ = nullptr;
  std::unique_ptr<compute_tile_detail::OwnedRunStorage> owned_storage_{};
  u64 bound_generation_ = 0u;
  const char *reason_ = "compute_tile_executor_not_validated";
  bool configured_ = false;
};

} // namespace rund::kernel
