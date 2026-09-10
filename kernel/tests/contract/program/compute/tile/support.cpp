#include "local.hpp"

#include <limits>

namespace program_compute_contract::tile_contract {
namespace {

[[nodiscard]] constexpr u64 AddBytes(const u64 left, const u64 right) noexcept {
  return right > std::numeric_limits<u64>::max() - left
             ? std::numeric_limits<u64>::max()
             : left + right;
}

} // namespace

u64 TotalBytes(const ComputeTileRetainedMemory memory) noexcept {
  u64 total = AddBytes(memory.state_bytes, memory.workspace_bytes);
  total = AddBytes(total, memory.failure_slot_bytes);
  total = AddBytes(total, memory.worker_tile_bytes);
  return AddBytes(total, memory.async_context_bytes);
}

bool SameMemory(const ComputeTileRetainedMemory left,
                const ComputeTileRetainedMemory right) noexcept {
  return left.state_bytes == right.state_bytes &&
         left.workspace_bytes == right.workspace_bytes &&
         left.failure_slot_bytes == right.failure_slot_bytes &&
         left.worker_tile_bytes == right.worker_tile_bytes &&
         left.async_context_bytes == right.async_context_bytes &&
         left.total_bytes == right.total_bytes;
}

ComputeTileExecutor MakeExecutor(const rund::kernel::WorkerBackend backend,
                                 const u32 workers) {
  return ComputeTileExecutor{
      backend, workers,
      rund::kernel::physical_tiles(TargetTilesPerWorker, 1u, TileUnits)};
}

} // namespace program_compute_contract::tile_contract
