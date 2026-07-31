#include "local.hpp"

namespace rund::kernel::workspace_detail {

void StoreWeightedNoGrowthScheduleState(Workspace& workspace,
                                        const ScheduleCompileRequest& request,
                                        const PartitionProjection& projection,
                                        const u32 max_packets) {
  workspace.schedule.packet_count = request.packet_count;
  workspace.schedule.execution_width = request.execution_width;
  workspace.schedule.intent = request.intent;
  workspace.schedule.placement = request.placement;
  workspace.schedule.alignment_packets = 1u;
  workspace.schedule.packets_per_partition_max = max_packets;
  workspace.schedule.worker_slot_count = projection.worker_slot_count;
  workspace.schedule.fold_slot_count = projection.fold_slot_count;
  workspace.schedule.useful_width = projection.useful_width;
  workspace.schedule.no_allocation = true;
}

PartitionBuild BuildWeightedNoGrowthSuccess(const PartitionProjection& projection,
                                            const u32 max_packets) {
  return PartitionBuild{
      .ok = true,
      .reason = "pass",
      .partition_count = projection.partition_count,
      .alignment_packets = 1u,
      .packets_per_partition_max = max_packets,
      .worker_slot_count = projection.worker_slot_count,
      .fold_slot_count = projection.fold_slot_count,
      .useful_width = projection.useful_width,
      .placement = projection.placement,
      .no_allocation = true,
  };
}

} // namespace rund::kernel::workspace_detail
