#pragma once

#include <kernel/schedule/planner/view.hpp>

namespace rund::kernel {

enum class PhysicalTileAssignment : u32 {
  None = 0u,
  ContiguousWorkerBlocks = 1u,
  StripedStatic = 2u,
};

inline constexpr u32 kDefaultPhysicalTileTargetTilesPerWorker = 3u;
inline constexpr u32 kDefaultPhysicalTileMinUnits = 1u;
inline constexpr u32 kDefaultPhysicalTileMaxUnits = 0u;

struct KernelProgramPhysicalTilePolicy {
  bool enabled = true;
  u32 target_tiles_per_worker = kDefaultPhysicalTileTargetTilesPerWorker;
  u32 min_tile_units = kDefaultPhysicalTileMinUnits;
  // Zero means no maximum cap. Non-zero values smaller than min_tile_units
  // disable the physical tile plan because the policy is contradictory. The
  // selected tile size must also satisfy this cap after schedule alignment.
  u32 max_tile_units = kDefaultPhysicalTileMaxUnits;
};

[[nodiscard]] constexpr KernelProgramPhysicalTilePolicy physical_tiles(
    const u32 target_tiles_per_worker = kDefaultPhysicalTileTargetTilesPerWorker,
    const u32 min_tile_units = kDefaultPhysicalTileMinUnits,
    const u32 max_tile_units = kDefaultPhysicalTileMaxUnits) noexcept {
  return KernelProgramPhysicalTilePolicy{
      .enabled = true,
      .target_tiles_per_worker = target_tiles_per_worker,
      .min_tile_units = min_tile_units,
      .max_tile_units = max_tile_units,
  };
}

[[nodiscard]] constexpr KernelProgramPhysicalTilePolicy physical_tiles_per_worker(
    const u32 target_tiles_per_worker,
    const u32 max_tile_units = kDefaultPhysicalTileMaxUnits) noexcept {
  return physical_tiles(target_tiles_per_worker,
                        kDefaultPhysicalTileMinUnits,
                        max_tile_units);
}

[[nodiscard]] constexpr KernelProgramPhysicalTilePolicy no_physical_tiles() noexcept {
  return KernelProgramPhysicalTilePolicy{.enabled = false};
}

struct KernelProgramTilePlan {
  ScheduleView schedule{};
  bool static_tile_map = false;
  bool global_claim_sync_elided = false;
  bool over_partitioned_tiles = false;
  u32 tile_count = 0u;
  u32 worker_tile_count = 0u;
  u32 local_reduce_slot_count = 0u;
  u32 backend_dispatch_count = 1u;
  bool physical_tiling_enabled = false;
  u32 physical_tile_units = 0u;
  u32 physical_tile_count = 0u;
  PhysicalTileAssignment physical_tile_assignment = PhysicalTileAssignment::None;
};

} // namespace rund::kernel
