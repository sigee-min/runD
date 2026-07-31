#include "local.hpp"
#include "../local.hpp"

namespace rund::kernel::internal {

PartitionBuild BuildPartitions(
    std::vector<Partition>& out_partitions,
    const PartitionRequest& request) {
  out_partitions.clear();
  const PartitionProjection projection = ProjectPartitions(request);
  if (!projection.ok) {
    return schedule::planner::FailProjectedPartitionBuild(projection, projection.reason);
  }
  const u32 partition_units = schedule::planner::ResolvePartitionUnits(request, projection.alignment_packets);
  uniform_detail::BuildUnitPartitions(out_partitions,
                                      partition_units,
                                      projection.partition_count,
                                      projection.alignment_packets,
                                      AllocationPolicy::AllowGrowth);
  u32 max_packets = 0u;
  if (!uniform_detail::ValidatePartitionCoverage(out_partitions.data(),
                                                 static_cast<u32>(out_partitions.size()),
                                                 request.packet_count,
                                                 max_packets)) {
    out_partitions.clear();
    return schedule::planner::FailPartitionBuild("partition_coverage_mismatch");
  }
  return PartitionBuild{
      .ok = true,
      .reason = "pass",
      .partition_count = static_cast<u32>(out_partitions.size()),
      .alignment_packets = projection.alignment_packets,
      .packets_per_partition_max = max_packets,
      .worker_slot_count = projection.worker_slot_count,
      .fold_slot_count = projection.fold_slot_count,
      .useful_width = projection.useful_width,
      .placement = projection.placement,
  };
}

} // namespace rund::kernel::internal
