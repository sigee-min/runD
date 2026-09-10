#pragma once

#include "contract/support/allocation.hpp"
#include "contract/support/backend/factory.hpp"
#include "test/assert.hpp"

#include <kernel/program/compute/tile/run.hpp>
#include <kernel/program/executor.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace program_compute_contract::tile_contract {

using rund::kernel::ComputeTile;
using rund::kernel::ComputeTileCallbackResult;
using rund::kernel::ComputeTileExecutor;
using rund::kernel::ComputeTileRetainedMemory;
using rund::kernel::ComputeTileRunStorage;
using rund::kernel::ComputeTileRunStorageView;
using rund::kernel::PlanComputeTileRunMemory;
using rund::kernel::u32;
using rund::kernel::u64;

inline constexpr u32 Workers = 4u;
inline constexpr u32 TileUnits = 32u;
inline constexpr u32 TargetTilesPerWorker = 4u;
inline constexpr u32 Boundary = Workers * TargetTilesPerWorker * TileUnits;

template <std::size_t FailureSlots, std::size_t WorkerCount>
struct ExternalRunStorage final {
  ComputeTileRunStorage state{};
  std::array<const char *, FailureSlots> failures{};
  std::array<u32, WorkerCount> worker_tiles{};
  std::array<u32, WorkerCount> worker_stats_partitions{};
  std::array<u64, WorkerCount> worker_stats_start_offset_ns{};
  std::array<u64, WorkerCount> worker_stats_elapsed_ns{};
  std::array<u64, WorkerCount> worker_stats_tail_wait_ns{};

  [[nodiscard]] ComputeTileRunStorageView view() noexcept {
    return ComputeTileRunStorageView{
        .state = &state,
        .failure_slots = failures,
        .worker_tiles = worker_tiles,
        .worker_stats_partitions = worker_stats_partitions,
        .worker_stats_start_offset_ns = worker_stats_start_offset_ns,
        .worker_stats_elapsed_ns = worker_stats_elapsed_ns,
        .worker_stats_tail_wait_ns = worker_stats_tail_wait_ns,
    };
  }
};

[[nodiscard]] u64 TotalBytes(ComputeTileRetainedMemory) noexcept;
[[nodiscard]] bool SameMemory(ComputeTileRetainedMemory,
                              ComputeTileRetainedMemory) noexcept;
[[nodiscard]] ComputeTileExecutor MakeExecutor(rund::kernel::WorkerBackend,
                                               u32 workers);

int CheckPositiveDispatch();
int CheckBoundaries();
int CheckExplicitBackend();
int CheckConstCallback();
int CheckIndependentRunState();
int CheckBorrowedStorage();
int CheckBoundedBind();
int CheckRetainedMemory();

} // namespace program_compute_contract::tile_contract
