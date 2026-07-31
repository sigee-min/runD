#include "weighted/local.hpp"

namespace rund::kernel::workspace_detail {

PartitionBuild CompileWeightedNoGrowthSchedule(Workspace& workspace,
                                               const ScheduleCompileRequest& request,
                                               const PartitionProjection& projection) {
  if (!PacketHintsValid(request)) {
    return FailWeightedNoGrowthSchedule(workspace, projection, "invalid_packet_hints");
  }
  if (!HasWorkUnits(request)) {
    return FailWeightedNoGrowthSchedule(workspace, projection, "invalid_packet_work_units");
  }
  const WeightedWorkSourcePlan source_plan = ClassifyWeightedWorkSource(workspace, request);
  if (source_plan.kind == WeightedWorkSourceKind::Invalid) {
    return FailWeightedNoGrowthSchedule(workspace, projection, source_plan.reason);
  }
  if (!HasWeightedNoGrowthCapacity(workspace, projection)) {
    return FailWeightedNoGrowthSchedule(workspace, projection, "weighted_workspace_capacity_exceeded");
  }

  PrepareWeightedNoGrowthBuffers(workspace, request, projection);
  if (!MaterializeWeightedWorkUnits(workspace, request, source_plan)) {
    return FailWeightedNoGrowthSchedule(workspace, projection, "invalid_packet_work_units");
  }
  const std::span<const u64> resolved_work_units =
      ViewResolvedWeightedWorkUnits(workspace, request.packet_count);
  if (resolved_work_units.size() != static_cast<std::size_t>(request.packet_count)) {
    return FailWeightedNoGrowthSchedule(workspace, projection, "invalid_packet_work_units");
  }
  AssignWeightedPacketPartitions(workspace, request, resolved_work_units);
  if (!ScatterWeightedPacketOrder(workspace, request, projection)) {
    return FailWeightedNoGrowthSchedule(workspace, projection, "partition_coverage_mismatch");
  }

  u32 max_packets = 0u;
  if (!BuildWeightedSchedulePartitions(workspace, request, projection, max_packets)) {
    return FailWeightedNoGrowthSchedule(workspace, projection, "partition_coverage_mismatch");
  }
  StoreWeightedNoGrowthScheduleState(workspace, request, projection, max_packets);
  workspace.last_failure_reason = "pass";
  return BuildWeightedNoGrowthSuccess(projection, max_packets);
}

} // namespace rund::kernel::workspace_detail
