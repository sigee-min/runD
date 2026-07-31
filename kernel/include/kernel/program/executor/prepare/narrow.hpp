#pragma once

#include <kernel/program/build.hpp>
#include <kernel/program/executor/prepare/run.hpp>

#include <limits>

namespace rund::kernel {

template <std::size_t Rank>
[[nodiscard]] inline PreparedEach<Rank> prepare_each(const Executor& exec,
                                                     const Space<Rank>& index_space) {
  if (!exec.valid) {
    return PreparedEach<Rank>{.exec = exec, .index_space = index_space, .reason = exec.reason};
  }
  if (exec.workspace == nullptr) {
    return PreparedEach<Rank>{
        .exec = exec,
        .index_space = index_space,
        .reason = "executor_workspace_missing",
    };
  }
  if (!exec.worker_backend) {
    return PreparedEach<Rank>{
        .exec = exec,
        .index_space = index_space,
        .reason = "executor_backend_invalid",
    };
  }
  if (const char* const reason =
          skeleton_detail::ValidatePartitionSpaceReason(index_space);
      reason != nullptr) {
    return PreparedEach<Rank>{.exec = exec, .index_space = index_space, .reason = reason};
  }
  if (const char* const reason =
          skeleton_detail::ValidateAlignmentReason(exec.boundary_alignment);
      reason != nullptr) {
    return PreparedEach<Rank>{.exec = exec, .index_space = index_space, .reason = reason};
  }

  const u64 units = skeleton_detail::UnitCount(index_space);
  if (units == 0u) {
    return PreparedEach<Rank>{
        .exec = exec,
        .index_space = index_space,
        .units = units,
        .valid = true,
        .reason = "pass",
    };
  }
  if (units > std::numeric_limits<u32>::max()) {
    return PreparedEach<Rank>{
        .exec = exec,
        .index_space = index_space,
        .units = units,
        .reason = "skeleton_partition_space_exceeds_u32",
    };
  }

  const u32 packet_count = static_cast<u32>(units);
  const KernelProgramCompileRequest compile_request{
      .schedule = ScheduleCompileRequest{
          .packet_count = packet_count,
          .execution_width = exec.workers,
          .intent = PartitionIntent::StaticWidth,
          .placement = PlacementPolicy::Uniform,
          .allocation = AllocationPolicy::NoGrowth,
          .preferred_alignment_packets = exec.boundary_alignment.units,
          .locality_bucket_packets = 1u,
          .alignment_group_packets = exec.boundary_alignment.units,
      },
      .worker_backend = exec.worker_backend,
      .require_no_allocation = true,
      .collect_worker_stats = exec.collect_worker_stats,
      .require_dispatch_backend = true,
      .physical_tile_policy = exec.physical_tile_policy,
      .fold_operation = FoldOperation::FixedBinaryTreeHash,
  };

  ResetWorkspace(*exec.workspace);
  if (!ReserveWorkspace(*exec.workspace,
                        KernelProgramWorkspaceReservation(compile_request))) {
    return PreparedEach<Rank>{
        .exec = exec,
        .index_space = index_space,
        .units = units,
        .reason = "executor_workspace_reserve_failed",
    };
  }
  const KernelProgramBuild build =
      CompileKernelProgram(*exec.workspace, compile_request);
  if (!build.ok) {
    return PreparedEach<Rank>{
        .exec = exec,
        .index_space = index_space,
        .units = units,
        .reason = build.reason,
    };
  }

  return PreparedEach<Rank>{
      .exec = exec,
      .index_space = index_space,
      .units = units,
      .program_generation = build.program.generation,
      .partition_count = build.program.schedule.partition_count,
      .physical_tiling_enabled = build.program.exec_kernel.physical_tiling_enabled,
      .physical_tile_units = build.program.exec_kernel.physical_tile_units,
      .physical_tile_count = build.program.exec_kernel.physical_tile_count,
      .valid = true,
      .reason = "pass",
  };
}

} // namespace rund::kernel
