#pragma once

#include <kernel/dispatch/worker/backend.hpp>
#include <kernel/program/skeleton/model.hpp>
#include <kernel/program/tile.hpp>

#include <limits>
#include <memory>

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

using ComputeTileCallback =
    ComputeTileCallbackResult (*)(const void *context, const ComputeTile &tile);
using ComputeTileReady = void (*)(void *context) noexcept;

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
  [[nodiscard]] ComputeTileExecutor make_run() const;

  template <typename Callback>
  [[nodiscard]] ComputeTileRunResult run(Callback &&callback);

  template <typename Callback>
  [[nodiscard]] ComputeTileRunResult run_with(WorkerBackend worker_backend,
                                              Callback &&callback);

  [[nodiscard]] bool prepared() const noexcept;
  [[nodiscard]] u32 count() const noexcept;
  [[nodiscard]] u32 tile_count() const noexcept;
  [[nodiscard]] ComputeTileRetainedMemory retained_memory() const noexcept;

  [[nodiscard]] ComputeTileSubmitResult
  submit_with_erased(WorkerBackend worker_backend, const void *callback_context,
                     ComputeTileCallback callback, void *ready_context,
                     ComputeTileReady ready) noexcept;
  [[nodiscard]] ComputeTileRunResult finish() noexcept;

private:
  struct State;

  [[nodiscard]] ComputeTileRunResult run_erased(const void *callback_context,
                                                ComputeTileCallback callback);
  [[nodiscard]] ComputeTileRunResult
  run_with_erased(WorkerBackend worker_backend, const void *callback_context,
                  ComputeTileCallback callback);

  static void CompleteSubmission(void *context, bool ok) noexcept;

  std::unique_ptr<State> state_{};
};

} // namespace rund::kernel
