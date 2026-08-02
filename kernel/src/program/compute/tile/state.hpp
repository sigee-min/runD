#pragma once

#include <kernel/dispatch/kernel.hpp>
#include <kernel/program/compute/tile/model.hpp>
#include <kernel/program/executor.hpp>

#include <cstdint>
#include <memory>

namespace rund::kernel::compute_tile_detail {

enum class Mode : std::uint8_t {
  Sync,
  Async,
};

enum class Phase : std::uint8_t {
  Idle = 0u,
  Sync = 1u,
  Async = 2u,
  Ready = 3u,
  Binding = 4u,
};

struct Backend final {
  WorkerBackend value{};
  bool ok = false;
  const char *reason = "compute_tile_backend_invalid";
};

struct PlanState final {
  WorkerBackend backend{};
  u32 workers = 0u;
  KernelProgramPhysicalTilePolicy tile_policy{};
  Alignment alignment{};
  Workspace workspace{};
  PreparedEach<1u> prepared{};
  ComputeTileRunStoragePlan storage{};
  u32 tile_count = 0u;
  const char *reason = "compute_tile_not_prepared";
};

// Exact standalone owner used only by make_run(). The execution algorithm sees
// the same typed view that an external sealed arena supplies; there is no
// second owning Workspace clone or vector-backed run representation.
struct OwnedRunStorage final {
  ComputeTileRunStorage state{};
  std::unique_ptr<const char *[]> failure_slots{};
  std::unique_ptr<u32[]> worker_tiles{};
  std::unique_ptr<u32[]> worker_stats_partitions{};
  std::unique_ptr<u64[]> worker_stats_start_offset_ns{};
  std::unique_ptr<u64[]> worker_stats_elapsed_ns{};
  std::unique_ptr<u64[]> worker_stats_tail_wait_ns{};
  u32 failure_slot_count = 0u;
  u32 worker_count = 0u;

  [[nodiscard]] bool allocate(ComputeTileRunStoragePlan plan) noexcept;
  [[nodiscard]] ComputeTileRunStorageView view() noexcept;
};

[[nodiscard]] Backend Select(WorkerBackend backend, u32 workers,
                             Mode mode) noexcept;

[[nodiscard]] ComputeTileRetainedMemory
MeasurePlanMemory(const PlanState &plan) noexcept;
[[nodiscard]] ComputeTileRunStorageView
BoundStorageView(ComputeTileRunStorage &storage) noexcept;
[[nodiscard]] bool
StorageViewSatisfies(ComputeTileRunStorageView storage,
                     ComputeTileRunStoragePlan plan) noexcept;

} // namespace rund::kernel::compute_tile_detail
