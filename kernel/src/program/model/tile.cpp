#include "tile.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::kernel::program_detail {
namespace {

bool HasStaticTileMap(const ScheduleView schedule,
                      const WorkerBackendCapabilities &capabilities) {
  return schedule.intent == PartitionIntent::StaticWidth &&
         capabilities.supports_static_tile_map;
}

u64 RoundUpToAlignment(const u64 value, const u32 alignment) {
  const u64 units = alignment == 0u ? 1u : static_cast<u64>(alignment);
  const u64 remainder = value % units;
  return remainder == 0u ? value : value + (units - remainder);
}

u64 RoundDownToAlignment(const u64 value, const u32 alignment) {
  const u64 units = alignment == 0u ? 1u : static_cast<u64>(alignment);
  return value - (value % units);
}

u32 SaturateU32(const u64 value) {
  return value > static_cast<u64>(std::numeric_limits<u32>::max())
             ? std::numeric_limits<u32>::max()
             : static_cast<u32>(value);
}

bool ValidPhysicalTilePolicy(const KernelProgramPhysicalTilePolicy &policy) {
  return policy.enabled && policy.target_tiles_per_worker > 0u &&
         policy.min_tile_units > 0u &&
         (policy.max_tile_units == 0u ||
          policy.max_tile_units >= policy.min_tile_units);
}

} // namespace

KernelProgramTilePlan BuildKernelProgramTilePlan(
    const ScheduleView schedule, const WorkerBackendCapabilities &capabilities,
    const KernelProgramPhysicalTilePolicy &physical_tile_policy) {
  const bool static_tiles = HasStaticTileMap(schedule, capabilities);
  const bool eligible_for_internal_tiling =
      ValidPhysicalTilePolicy(physical_tile_policy) && static_tiles &&
      capabilities.supports_claim_free_static_tiles &&
      schedule.placement == PlacementPolicy::Uniform &&
      schedule.ordered_packet_indices == nullptr &&
      schedule.packet_count > 0u && schedule.execution_width > 1u;
  const u64 target_tile_count =
      static_cast<u64>(schedule.execution_width) *
      static_cast<u64>(physical_tile_policy.target_tiles_per_worker);
  const u64 ideal_tile_units =
      checked::ceil(schedule.packet_count, target_tile_count);
  const bool physical_tile_units_large_enough =
      ideal_tile_units >= physical_tile_policy.min_tile_units;
  u64 selected_tile_units = ideal_tile_units;
  if (physical_tile_policy.max_tile_units > 0u &&
      selected_tile_units > physical_tile_policy.max_tile_units) {
    selected_tile_units = physical_tile_policy.max_tile_units;
  }
  u64 aligned_tile_units =
      RoundUpToAlignment(selected_tile_units, schedule.alignment_packets);
  if (physical_tile_policy.max_tile_units > 0u &&
      aligned_tile_units > physical_tile_policy.max_tile_units) {
    aligned_tile_units = RoundDownToAlignment(
        physical_tile_policy.max_tile_units, schedule.alignment_packets);
  }
  const bool physical_tile_units_satisfy_policy =
      aligned_tile_units >= physical_tile_policy.min_tile_units &&
      (physical_tile_policy.max_tile_units == 0u ||
       aligned_tile_units <= physical_tile_policy.max_tile_units);
  const u32 physical_tile_units = SaturateU32(aligned_tile_units);
  const u32 physical_tile_count =
      SaturateU32(checked::ceil(schedule.packet_count, physical_tile_units));
  const bool physical_tiling_enabled =
      eligible_for_internal_tiling && physical_tile_units_large_enough &&
      physical_tile_units_satisfy_policy && physical_tile_units > 0u &&
      physical_tile_count >= schedule.execution_width;
  return KernelProgramTilePlan{
      .schedule = schedule,
      .static_tile_map = static_tiles,
      .global_claim_sync_elided =
          static_tiles && capabilities.supports_claim_free_static_tiles,
      .over_partitioned_tiles =
          schedule.partition_count > schedule.execution_width,
      .tile_count = physical_tiling_enabled ? physical_tile_count
                                            : schedule.partition_count,
      .worker_tile_count = physical_tiling_enabled ? physical_tile_count
                                                   : schedule.worker_slot_count,
      .local_reduce_slot_count = schedule.fold_slot_count,
      .backend_dispatch_count = static_tiles ? 1u : 0u,
      .physical_tiling_enabled = physical_tiling_enabled,
      .physical_tile_units = physical_tiling_enabled ? physical_tile_units : 0u,
      .physical_tile_count = physical_tiling_enabled ? physical_tile_count : 0u,
      .physical_tile_assignment = physical_tiling_enabled
                                      ? PhysicalTileAssignment::StripedStatic
                                      : PhysicalTileAssignment::None,
  };
}

} // namespace rund::kernel::program_detail
