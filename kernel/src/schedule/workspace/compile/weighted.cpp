#include "local.hpp"

namespace rund::kernel::workspace_compile_detail {

PartitionBuild CompileWeightedSchedule(Workspace& workspace,
                                       const ScheduleCompileRequest& request,
                                       const PartitionProjection& projection) {
  if (request.allocation == AllocationPolicy::NoGrowth) {
    return workspace_detail::CompileWeightedNoGrowthSchedule(workspace, request, projection);
  }
  const workspace_detail::WeightedWorkSourcePlan source_plan =
      workspace_detail::ClassifyWeightedWorkSource(workspace, request);
  if (source_plan.kind == workspace_detail::WeightedWorkSourceKind::Invalid) {
    return workspace_detail::FailWeightedNoGrowthSchedule(workspace, projection, source_plan.reason);
  }
  if (!ReserveWorkspace(workspace, ScheduleWorkspaceReservation(request))) {
    return workspace_detail::FailWeightedNoGrowthSchedule(workspace, projection, "workspace_reserve_failed");
  }
  PartitionBuild build =
      workspace_detail::CompileWeightedNoGrowthSchedule(workspace, request, projection);
  workspace.schedule.no_allocation = false;
  build.no_allocation = false;
  return build;
}

void StoreWeightedScheduleState(Workspace& workspace,
                                const ScheduleCompileRequest& request,
                                const PartitionBuild& build) {
  workspace.schedule.packet_count = request.packet_count;
  workspace.schedule.execution_width = request.execution_width;
  workspace.schedule.intent = request.intent;
  workspace.schedule.placement = request.placement;
  workspace.schedule.alignment_packets = build.alignment_packets;
  workspace.schedule.packets_per_partition_max = build.packets_per_partition_max;
  workspace.schedule.worker_slot_count = build.worker_slot_count;
  workspace.schedule.fold_slot_count = build.fold_slot_count;
  workspace.schedule.useful_width = build.useful_width;
  workspace.schedule.no_allocation = request.allocation == AllocationPolicy::NoGrowth;
}

} // namespace rund::kernel::workspace_compile_detail
