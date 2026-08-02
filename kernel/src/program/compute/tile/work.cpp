#include "state.hpp"

#include <algorithm>

namespace rund::kernel::compute_tile_detail {
namespace {

void PublishLowerFailure(std::atomic<u32> &first, const u32 tile) noexcept {
  u32 current = first.load(std::memory_order_acquire);
  while (tile < current &&
         !first.compare_exchange_weak(current, tile, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
  }
}

} // namespace

void Begin(ComputeTileRunStorage &storage, const void *const callback_context,
           const ComputeTileCallback callback) noexcept {
  storage.callback_context_ = callback_context;
  storage.callback_ = callback;
  std::fill_n(storage.worker_tiles_, storage.worker_count_, 0u);
  storage.first_failure_.store(kNoComputeTileFailure,
                               std::memory_order_release);
  storage.completed_.store(0u, std::memory_order_release);
}

void Invoke(void *const raw, const Partition &partition) noexcept {
  auto *const storage = static_cast<ComputeTileRunStorage *>(raw);
  if (storage == nullptr || storage->callback_ == nullptr ||
      storage->failures_ == nullptr || storage->worker_tiles_ == nullptr ||
      !storage->prepared_.valid) {
    return;
  }
  const PreparedEach<1u> &prepared = storage->prepared_;
  const u32 tile =
      prepared.physical_tiling_enabled
          ? partition.begin / static_cast<u32>(prepared.physical_tile_units)
          : partition.worker_index;
  const u32 end = std::min(partition.end, storage->active_count_);
  if (partition.begin >= end || tile >= storage->active_tile_count_ ||
      tile >= storage->failure_count_ ||
      tile > storage->first_failure_.load(std::memory_order_acquire)) {
    return;
  }

  ComputeTileCallbackResult result{};
  try {
    result = storage->callback_(storage->callback_context_,
                                ComputeTile{
                                    .index = tile,
                                    .worker_index = partition.worker_index,
                                    .begin = partition.begin,
                                    .end = end,
                                });
  } catch (...) {
    result = ComputeTileCallbackResult{
        .ok = false,
        .reason = "compute_tile_callback_failed",
    };
  }
  storage->completed_.fetch_add(1u, std::memory_order_relaxed);
  if (partition.worker_index < storage->worker_count_) {
    ++storage->worker_tiles_[partition.worker_index];
  }
  if (result.ok) {
    return;
  }

  storage->failures_[tile] =
      result.reason != nullptr ? result.reason : "compute_tile_callback_failed";
  PublishLowerFailure(storage->first_failure_, tile);
}

void InvokeWorker(void *const raw, const Partition &worker_partition) noexcept {
  auto *const storage = static_cast<ComputeTileRunStorage *>(raw);
  if (storage == nullptr || !storage->prepared_.valid) {
    return;
  }
  const PreparedEach<1u> &prepared = storage->prepared_;
  if (!prepared.physical_tiling_enabled) {
    Invoke(raw, worker_partition);
    return;
  }
  const u32 units = static_cast<u32>(prepared.physical_tile_units);
  const u32 tiles = storage->active_tile_count_;
  if (units == 0u || prepared.exec.workers == 0u) {
    return;
  }
  for (u32 tile = worker_partition.worker_index; tile < tiles;
       tile += prepared.exec.workers) {
    const u64 begin64 = static_cast<u64>(tile) * units;
    const u64 end64 = begin64 + units;
    const u32 begin = static_cast<u32>(std::min<u64>(begin64, prepared.units));
    const u32 end =
        static_cast<u32>(std::min<u64>(end64, storage->active_count_));
    if (begin < end) {
      Invoke(raw, Partition{.worker_index = worker_partition.worker_index,
                            .begin = begin,
                            .end = end});
    }
  }
}

} // namespace rund::kernel::compute_tile_detail
