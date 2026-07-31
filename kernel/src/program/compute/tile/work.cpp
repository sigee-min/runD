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

void Begin(Context &context, const void *const callback_context,
           const ComputeTileCallback callback, const PreparedEach<1u> &prepared,
           std::vector<const char *> &failures,
           std::vector<u32> &worker_tiles) noexcept {
  context.callback_context = callback_context;
  context.callback = callback;
  context.prepared = &prepared;
  context.failures = failures.data();
  context.failure_count = static_cast<u32>(failures.size());
  context.worker_tiles = worker_tiles.data();
  context.worker_count = static_cast<u32>(worker_tiles.size());
  std::fill(worker_tiles.begin(), worker_tiles.end(), 0u);
  context.first_failure.store(kNoComputeTileFailure, std::memory_order_release);
  context.completed.store(0u, std::memory_order_release);
}

void Invoke(void *const raw, const Partition &partition) noexcept {
  auto *const context = static_cast<Context *>(raw);
  if (context == nullptr || context->callback == nullptr ||
      context->prepared == nullptr || context->failures == nullptr ||
      context->worker_tiles == nullptr) {
    return;
  }
  const PreparedEach<1u> &prepared = *context->prepared;
  const u32 tile =
      prepared.physical_tiling_enabled
          ? partition.begin / static_cast<u32>(prepared.physical_tile_units)
          : partition.worker_index;
  if (tile >= context->failure_count ||
      tile > context->first_failure.load(std::memory_order_acquire)) {
    return;
  }

  ComputeTileCallbackResult result{};
  try {
    result = context->callback(context->callback_context,
                               ComputeTile{
                                   .index = tile,
                                   .worker_index = partition.worker_index,
                                   .begin = partition.begin,
                                   .end = partition.end,
                               });
  } catch (...) {
    result = ComputeTileCallbackResult{
        .ok = false,
        .reason = "compute_tile_callback_failed",
    };
  }
  context->completed.fetch_add(1u, std::memory_order_relaxed);
  if (partition.worker_index < context->worker_count) {
    ++context->worker_tiles[partition.worker_index];
  }
  if (result.ok) {
    return;
  }

  context->failures[tile] =
      result.reason != nullptr ? result.reason : "compute_tile_callback_failed";
  PublishLowerFailure(context->first_failure, tile);
}

void InvokeWorker(void *const raw, const Partition &worker_partition) noexcept {
  auto *const context = static_cast<Context *>(raw);
  if (context == nullptr || context->prepared == nullptr) {
    return;
  }
  const PreparedEach<1u> &prepared = *context->prepared;
  if (!prepared.physical_tiling_enabled) {
    Invoke(raw, worker_partition);
    return;
  }
  const u32 units = static_cast<u32>(prepared.physical_tile_units);
  const u32 tiles = static_cast<u32>(prepared.physical_tile_count);
  if (units == 0u || prepared.exec.workers == 0u) {
    return;
  }
  for (u32 tile = worker_partition.worker_index; tile < tiles;
       tile += prepared.exec.workers) {
    const u64 begin64 = static_cast<u64>(tile) * units;
    const u64 end64 = begin64 + units;
    const u32 begin = static_cast<u32>(std::min<u64>(begin64, prepared.units));
    const u32 end = static_cast<u32>(std::min<u64>(end64, prepared.units));
    if (begin < end) {
      Invoke(raw, Partition{.worker_index = worker_partition.worker_index,
                            .begin = begin,
                            .end = end});
    }
  }
}

} // namespace rund::kernel::compute_tile_detail
