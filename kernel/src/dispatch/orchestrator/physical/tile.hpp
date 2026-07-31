#pragma once

#include "../local.hpp"

namespace rund::kernel::orchestrator_detail {

struct PhysicalTileDispatchAdapter {
  void* context = nullptr;
  DispatchFn dispatch = nullptr;
  FailureSignal* failure_signal = nullptr;
  u32 packet_count = 0u;
  u32 execution_width = 1u;
  u32 physical_tile_units = 0u;
  u32 physical_tile_count = 0u;
};

[[nodiscard]] inline bool UseStripedPhysicalTiles(const KernelProgram& program) {
  return program.exec_kernel.physical_tiling_enabled &&
         program.exec_kernel.physical_tile_assignment == PhysicalTileAssignment::StripedStatic &&
         program.exec_kernel.physical_tile_units > 0u &&
         program.exec_kernel.physical_tile_count > 0u;
}

inline void InvokeStripedPhysicalTiles(void* const raw_context,
                                       const Partition& worker_partition) {
  auto* const adapter = static_cast<PhysicalTileDispatchAdapter*>(raw_context);
  if (adapter == nullptr || adapter->dispatch == nullptr ||
      adapter->physical_tile_units == 0u || adapter->execution_width == 0u) {
    return;
  }
  const u32 worker = worker_partition.worker_index;
  for (u32 tile = worker; tile < adapter->physical_tile_count; tile += adapter->execution_width) {
    if (adapter->failure_signal != nullptr && HasFailure(*adapter->failure_signal)) {
      return;
    }
    const u64 begin64 = static_cast<u64>(tile) * adapter->physical_tile_units;
    const u64 end64 = begin64 + adapter->physical_tile_units;
    const u32 begin = begin64 > adapter->packet_count
                          ? adapter->packet_count
                          : static_cast<u32>(begin64);
    const u32 end = end64 > adapter->packet_count
                        ? adapter->packet_count
                        : static_cast<u32>(end64);
    if (begin >= end) {
      continue;
    }
    adapter->dispatch(adapter->context,
                      Partition{
                          .worker_index = worker,
                          .begin = begin,
                          .end = end,
                      });
  }
}

inline void AttachPhysicalTileTelemetry(Workspace& workspace) {
  const KernelProgramTilePlan& tile_plan = workspace.program.exec_kernel;
  if (!tile_plan.physical_tiling_enabled || tile_plan.physical_tile_count == 0u ||
      tile_plan.schedule.execution_width == 0u) {
    return;
  }
  Telemetry& telemetry = workspace.telemetry;
  const u32 workers = tile_plan.schedule.execution_width;
  const u32 min_tiles = tile_plan.physical_tile_count / workers;
  const u32 extra_tiles = tile_plan.physical_tile_count % workers;
  const u32 max_tiles = min_tiles + (extra_tiles == 0u ? 0u : 1u);
  telemetry.worker_count = workers;
  telemetry.worker_tile_count = tile_plan.physical_tile_count;
  telemetry.min_tiles_per_worker = min_tiles;
  telemetry.max_tiles_per_worker = max_tiles;
  telemetry.tile_imbalance_milli =
      min_tiles == 0u ? 0u : static_cast<u32>(((max_tiles - min_tiles) * 1000u) / min_tiles);
  telemetry.static_tile_map_used = tile_plan.static_tile_map;
  telemetry.global_claim_sync_elided = tile_plan.global_claim_sync_elided;
  telemetry.backend_dispatch_count = tile_plan.backend_dispatch_count;
}

} // namespace rund::kernel::orchestrator_detail
