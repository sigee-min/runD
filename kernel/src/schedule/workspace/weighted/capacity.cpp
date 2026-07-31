#include "../local.hpp"

namespace rund::kernel::workspace_detail {

bool HasWeightedNoGrowthCapacity(const Workspace& workspace,
                                 const PartitionProjection& projection) {
  const std::size_t packet_count = projection.packet_count;
  const std::size_t partition_count = projection.partition_count;
  return workspace.schedule.partitions.capacity() >= partition_count &&
         workspace.packet_work_units.capacity() >= packet_count &&
         workspace.ordered_packet_indices.capacity() >= packet_count &&
         workspace.packet_partition_indices.capacity() >= packet_count &&
         workspace.ordered_packet_scratch.capacity() >= packet_count &&
         workspace.partition_loads.capacity() >= partition_count &&
         workspace.partition_counts.capacity() >= partition_count &&
         workspace.partition_offsets.capacity() >= partition_count &&
         workspace.partition_write_offsets.capacity() >= partition_count;
}

PartitionBuild FailWeightedNoGrowthSchedule(Workspace& workspace,
                                            const PartitionProjection& projection,
                                            const char* reason) {
  workspace.last_failure_reason = reason;
  return PartitionBuild{
      .ok = false,
      .reason = workspace.last_failure_reason,
      .partition_count = projection.partition_count,
      .alignment_packets = projection.alignment_packets,
      .worker_slot_count = projection.worker_slot_count,
      .fold_slot_count = projection.fold_slot_count,
      .useful_width = projection.useful_width,
      .placement = projection.placement,
      .no_allocation = true,
  };
}

} // namespace rund::kernel::workspace_detail
