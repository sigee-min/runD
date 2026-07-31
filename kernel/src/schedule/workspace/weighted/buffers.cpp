#include "local.hpp"

namespace rund::kernel::workspace_detail {

void PrepareWeightedNoGrowthBuffers(Workspace& workspace,
                                    const ScheduleCompileRequest& request,
                                    const PartitionProjection& projection) {
  workspace.schedule.partitions.clear();
  workspace.packet_work_units.resize(request.packet_count);
  workspace.ordered_packet_indices.resize(request.packet_count);
  workspace.packet_partition_indices.resize(request.packet_count);
  workspace.ordered_packet_scratch.resize(request.packet_count);
  workspace.partition_loads.resize(projection.partition_count);
  workspace.partition_counts.resize(projection.partition_count);
  workspace.partition_offsets.resize(projection.partition_count);
  workspace.partition_write_offsets.resize(projection.partition_count);
}

} // namespace rund::kernel::workspace_detail
