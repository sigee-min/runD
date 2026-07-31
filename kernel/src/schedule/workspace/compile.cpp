#include "compile/local.hpp"

#include <kernel/internal/workspace/schedule.hpp>

#include <chrono>

namespace rund::kernel::internal {

PartitionBuild CompileWorkspaceSchedule(Workspace& workspace,
                                        const ScheduleCompileRequest& request) {
  const auto compile_start = std::chrono::steady_clock::now();
  const PartitionProjection projection = ProjectSchedule(request);
  if (!projection.ok) {
    PartitionBuild build =
        workspace_compile_detail::BuildProjectedFailure(workspace, request, projection, projection.reason);
    workspace_detail::RecordCompileOutcome(workspace, request, build, {}, compile_start);
    return build;
  }

  if (!workspace_detail::PacketHintsValid(request)) {
    PartitionBuild build =
        workspace_compile_detail::BuildProjectedFailure(workspace, request, projection, "invalid_packet_hints");
    workspace_detail::RecordCompileOutcome(workspace, request, build, {}, compile_start);
    return build;
  }

  if (request.allocation == AllocationPolicy::NoGrowth) {
    if (workspace.schedule.partitions.capacity() < projection.partition_count) {
      PartitionBuild build = workspace_compile_detail::BuildProjectedFailure(
          workspace,
          request,
          projection,
          "schedule_partition_capacity_exceeded");
      workspace_detail::RecordCompileOutcome(workspace, request, build, {}, compile_start);
      return build;
    }
    if (request.placement == PlacementPolicy::WeightedStable &&
        !workspace_detail::HasWeightedNoGrowthCapacity(workspace, projection)) {
      PartitionBuild build =
          workspace_detail::FailWeightedNoGrowthSchedule(workspace, projection, "weighted_workspace_capacity_exceeded");
      workspace_detail::RecordCompileOutcome(workspace, request, build, {}, compile_start);
      return build;
    }
  }

  workspace.ordered_packet_indices.clear();
  PartitionBuild build{};
  if (request.placement == PlacementPolicy::WeightedStable) {
    build = workspace_compile_detail::CompileWeightedSchedule(workspace, request, projection);
  } else {
    build = workspace_compile_detail::CompileStandardSchedule(workspace, request, projection);
  }
  if (build.ok && request.placement == PlacementPolicy::WeightedStable) {
    workspace_compile_detail::StoreWeightedScheduleState(workspace, request, build);
  }
  build.no_allocation = request.allocation == AllocationPolicy::NoGrowth;
  workspace.last_failure_reason = build.reason;
  const std::span<const u64> resolved_work_units =
      build.ok && request.placement == PlacementPolicy::WeightedStable
          ? workspace_detail::ViewResolvedWeightedWorkUnits(workspace, request.packet_count)
          : std::span<const u64>{};
  workspace_detail::RecordCompileOutcome(workspace, request, build, resolved_work_units, compile_start);
  return build;
}

} // namespace rund::kernel::internal
