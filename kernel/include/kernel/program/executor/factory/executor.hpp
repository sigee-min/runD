#pragma once

#include <kernel/program/executor/model.hpp>

namespace rund::kernel {

[[nodiscard]] inline Executor executor(
    Workspace& workspace,
    const WorkerBackend worker_backend,
    const u32 workers,
    const Alignment boundary_alignment = Alignment{},
    const KernelProgramPhysicalTilePolicy physical_tile_policy =
        KernelProgramPhysicalTilePolicy{}) noexcept {
  if (workers == 0u) {
    return Executor{
        .workspace = &workspace,
        .worker_backend = worker_backend,
        .workers = workers,
        .boundary_alignment = boundary_alignment,
        .physical_tile_policy = physical_tile_policy,
        .reason = "executor_zero_workers",
    };
  }
  if (!worker_backend) {
    return Executor{
        .workspace = &workspace,
        .worker_backend = worker_backend,
        .workers = workers,
        .boundary_alignment = boundary_alignment,
        .physical_tile_policy = physical_tile_policy,
        .reason = "executor_backend_invalid",
    };
  }
  if (const char* const reason =
          skeleton_detail::ValidateAlignmentReason(boundary_alignment);
      reason != nullptr) {
    return Executor{
        .workspace = &workspace,
        .worker_backend = worker_backend,
        .workers = workers,
        .boundary_alignment = boundary_alignment,
        .physical_tile_policy = physical_tile_policy,
        .reason = reason,
    };
  }
  return Executor{
      .workspace = &workspace,
      .worker_backend = worker_backend,
      .workers = workers,
      .boundary_alignment = boundary_alignment,
      .physical_tile_policy = physical_tile_policy,
      .valid = true,
      .reason = "pass",
  };
}

} // namespace rund::kernel
