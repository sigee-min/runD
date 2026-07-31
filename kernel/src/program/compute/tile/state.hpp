#pragma once

#include <kernel/dispatch/kernel.hpp>
#include <kernel/program/compute/tile/model.hpp>
#include <kernel/program/executor.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::kernel {

namespace compute_tile_detail {

enum class Mode : std::uint8_t {
  Sync,
  Async,
};

struct Backend final {
  WorkerBackend value{};
  bool ok = false;
  const char *reason = "compute_tile_backend_invalid";
};

struct Context final {
  const void *callback_context = nullptr;
  ComputeTileCallback callback = nullptr;
  const PreparedEach<1u> *prepared = nullptr;
  const char **failures = nullptr;
  u32 failure_count = 0u;
  u32 *worker_tiles = nullptr;
  u32 worker_count = 0u;
  std::atomic<u32> first_failure{kNoComputeTileFailure};
  std::atomic<u32> completed{0u};
};

struct Completion final {
  bool ok = false;
  const char *reason = "compute_tile_backend_failed";
  u32 worker_count = 0u;
  u32 dispatch_count = 0u;
  bool report_tail_on_failure = false;
};

[[nodiscard]] Backend Select(WorkerBackend backend, u32 workers,
                             Mode mode) noexcept;

void Begin(Context &context, const void *callback_context,
           ComputeTileCallback callback, const PreparedEach<1u> &prepared,
           std::vector<const char *> &failures,
           std::vector<u32> &worker_tiles) noexcept;

void Invoke(void *context, const Partition &partition) noexcept;
void InvokeWorker(void *context, const Partition &partition) noexcept;

[[nodiscard]] ComputeTileRunResult Project(const Context &context,
                                           const Completion &completion,
                                           u32 count, u32 tile_count,
                                           u32 tile_units) noexcept;

} // namespace compute_tile_detail

struct ComputeTileExecutor::State {
  WorkerBackend backend{};
  u32 workers = 0u;
  KernelProgramPhysicalTilePolicy tile_policy{};
  Alignment alignment{};
  Workspace workspace{};
  PreparedEach<1u> prepared{};
  std::vector<const char *> failures{};
  std::vector<u32> worker_tiles{};
  std::unique_ptr<compute_tile_detail::Context> async_context{};
  WorkerSubmission submission{};
  std::atomic<std::uint8_t> async_phase{0u};
  void *ready_context = nullptr;
  ComputeTileReady ready = nullptr;
  bool async_ok = false;
  const char *reason = "compute_tile_executor_not_validated";
};

} // namespace rund::kernel
