#pragma once

#include <kernel/core/model.hpp>

namespace rund::kernel {

enum class TilePhaseOrder : u32 {
  None = 0u,
  TileIndexAscending = 1u,
};

struct TilePhaseCapacityRequirement {
  u64 scratch_bytes_per_tile = 0u;
  u64 scratch_alignment = 1u;
  u64 output_shards = 0u;
  u64 queue_slots = 0u;
  u64 task_slots = 0u;
};

struct TilePhasePreparedCapacity {
  u64 scratch_bytes = 0u;
  u64 output_shards = 0u;
  u64 queue_slots = 0u;
  u64 task_slots = 0u;
};

struct TilePhaseDescription {
  u64 phase_id = 0u;
  u64 tile_count = 0u;
  TilePhaseOrder order = TilePhaseOrder::TileIndexAscending;
  TilePhaseCapacityRequirement capacity{};
};

struct TilePhaseAdmissionResult {
  bool ok = false;
  const char* reason = "tile_phase_invalid";
  TilePhasePreparedCapacity required{};
};

struct TilePhaseTile {
  bool ok = false;
  const char* reason = "tile_phase_invalid";
  u64 phase_id = 0u;
  u64 tile_index = 0u;
};

[[nodiscard]] constexpr bool TilePhaseIsPowerOfTwo(const u64 value) noexcept {
  return value != 0u && (value & (value - 1u)) == 0u;
}

[[nodiscard]] constexpr bool TilePhaseAlignmentInvalid(
    const TilePhaseDescription& phase) noexcept {
  return !TilePhaseIsPowerOfTwo(phase.capacity.scratch_alignment);
}

[[nodiscard]] constexpr bool TilePhaseAlignOverflows(
    const TilePhaseDescription& phase) noexcept {
  const u64 alignment = phase.capacity.scratch_alignment;
  return alignment != 0u &&
         phase.capacity.scratch_bytes_per_tile > (~u64{0u} - (alignment - 1u));
}

[[nodiscard]] constexpr u64 TilePhaseScratchStride(
    const TilePhaseDescription& phase) noexcept {
  const u64 alignment = phase.capacity.scratch_alignment;
  if (!TilePhaseIsPowerOfTwo(alignment) || TilePhaseAlignOverflows(phase)) {
    return ~u64{0u};
  }
  const u64 bytes = phase.capacity.scratch_bytes_per_tile;
  return (bytes + (alignment - 1u)) & ~(alignment - 1u);
}

[[nodiscard]] constexpr bool TilePhaseScratchOverflows(
    const TilePhaseDescription& phase) noexcept {
  if (TilePhaseAlignmentInvalid(phase) || TilePhaseAlignOverflows(phase)) {
    return true;
  }
  const u64 stride = TilePhaseScratchStride(phase);
  return stride != 0u && phase.tile_count > (~u64{0u} / stride);
}

[[nodiscard]] constexpr TilePhasePreparedCapacity TilePhaseRequiredCapacity(
    const TilePhaseDescription& phase) noexcept {
  if (TilePhaseScratchOverflows(phase)) {
    return TilePhasePreparedCapacity{
        .scratch_bytes = ~u64{0u},
        .output_shards = phase.capacity.output_shards,
        .queue_slots = phase.capacity.queue_slots,
        .task_slots = phase.capacity.task_slots,
    };
  }
  return TilePhasePreparedCapacity{
      .scratch_bytes = TilePhaseScratchStride(phase) * phase.tile_count,
      .output_shards = phase.capacity.output_shards,
      .queue_slots = phase.capacity.queue_slots,
      .task_slots = phase.capacity.task_slots,
  };
}

[[nodiscard]] constexpr TilePhaseAdmissionResult ValidateTilePhaseDescription(
    const TilePhaseDescription& phase) noexcept {
  if (phase.phase_id == 0u) {
    return TilePhaseAdmissionResult{.reason = "tile_phase_id_invalid"};
  }
  if (phase.tile_count == 0u) {
    return TilePhaseAdmissionResult{.reason = "tile_phase_tile_count_invalid"};
  }
  if (phase.order != TilePhaseOrder::TileIndexAscending) {
    return TilePhaseAdmissionResult{.reason = "tile_phase_order_invalid"};
  }
  if (TilePhaseAlignmentInvalid(phase)) {
    return TilePhaseAdmissionResult{.reason = "tile_phase_alignment_invalid"};
  }
  if (TilePhaseScratchOverflows(phase)) {
    return TilePhaseAdmissionResult{.reason = "tile_phase_capacity_overflow"};
  }
  return TilePhaseAdmissionResult{
      .ok = true,
      .reason = "ok",
      .required = TilePhaseRequiredCapacity(phase),
  };
}

[[nodiscard]] constexpr TilePhaseAdmissionResult AdmitTilePhase(
    const TilePhaseDescription& phase,
    const TilePhasePreparedCapacity& available) noexcept {
  const TilePhaseAdmissionResult validation =
      ValidateTilePhaseDescription(phase);
  if (!validation.ok) {
    return validation;
  }
  const TilePhasePreparedCapacity required = validation.required;
  if (available.scratch_bytes < required.scratch_bytes ||
      available.output_shards < required.output_shards ||
      available.queue_slots < required.queue_slots ||
      available.task_slots < required.task_slots) {
    return TilePhaseAdmissionResult{
        .reason = "tile_phase_capacity_exceeded",
        .required = required,
    };
  }
  return validation;
}

[[nodiscard]] constexpr TilePhaseTile TilePhaseTileAt(
    const TilePhaseDescription& phase, const u64 tile_index) noexcept {
  const TilePhaseAdmissionResult validation =
      ValidateTilePhaseDescription(phase);
  if (!validation.ok) {
    return TilePhaseTile{.reason = validation.reason};
  }
  if (tile_index >= phase.tile_count) {
    return TilePhaseTile{.reason = "tile_phase_tile_index_out_of_range"};
  }
  return TilePhaseTile{
      .ok = true,
      .reason = "ok",
      .phase_id = phase.phase_id,
      .tile_index = tile_index,
  };
}

}  // namespace rund::kernel
