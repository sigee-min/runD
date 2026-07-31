#pragma once

#include <kernel/program/executor/model.hpp>

namespace rund::kernel {

[[nodiscard]] constexpr SerialPolicy seq() noexcept {
  return SerialPolicy{};
}

[[nodiscard]] inline ParallelPolicy par(
    const u32 workers,
    const Alignment boundary_alignment = Alignment{},
    const KernelProgramPhysicalTilePolicy physical_tile_policy =
        KernelProgramPhysicalTilePolicy{}) noexcept {
  if (workers == 0u) {
    return ParallelPolicy{
        .workers = workers,
        .boundary_alignment = boundary_alignment,
        .physical_tile_policy = physical_tile_policy,
        .valid = false,
        .reason = "parallel_zero_workers",
    };
  }
  if (const char* const reason =
          skeleton_detail::ValidateAlignmentReason(boundary_alignment);
      reason != nullptr) {
    return ParallelPolicy{
        .workers = workers,
        .boundary_alignment = boundary_alignment,
        .physical_tile_policy = physical_tile_policy,
        .valid = false,
        .reason = reason,
    };
  }
  return ParallelPolicy{
      .workers = workers,
      .boundary_alignment = boundary_alignment,
      .physical_tile_policy = physical_tile_policy,
  };
}

[[nodiscard]] inline ParallelPolicy par(
    const Alignment boundary_alignment,
    const KernelProgramPhysicalTilePolicy physical_tile_policy =
        KernelProgramPhysicalTilePolicy{}) noexcept {
  if (const char* const reason =
          skeleton_detail::ValidateAlignmentReason(boundary_alignment);
      reason != nullptr) {
    return ParallelPolicy{
        .workers = 0u,
        .use_runtime_default_workers = true,
        .boundary_alignment = boundary_alignment,
        .physical_tile_policy = physical_tile_policy,
        .valid = false,
        .reason = reason,
    };
  }
  return ParallelPolicy{
      .workers = 0u,
      .use_runtime_default_workers = true,
      .boundary_alignment = boundary_alignment,
      .physical_tile_policy = physical_tile_policy,
  };
}

[[nodiscard]] inline ParallelPolicy par() noexcept {
  return ParallelPolicy{
      .workers = 0u,
      .use_runtime_default_workers = true,
  };
}

[[nodiscard]] inline ParallelPolicy par(
    const KernelProgramPhysicalTilePolicy physical_tile_policy) noexcept {
  return ParallelPolicy{
      .workers = 0u,
      .use_runtime_default_workers = true,
      .physical_tile_policy = physical_tile_policy,
  };
}

} // namespace rund::kernel
