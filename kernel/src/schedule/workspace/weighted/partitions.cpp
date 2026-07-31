#include "local.hpp"

#include <algorithm>

namespace rund::kernel::workspace_detail {

bool BuildWeightedSchedulePartitions(Workspace& workspace,
                                     const ScheduleCompileRequest& request,
                                     const PartitionProjection& projection,
                                     u32& out_max_packets) {
  workspace.schedule.partitions.resize(projection.partition_count);
  out_max_packets = 0u;
  for (u32 partition = 0u; partition < projection.partition_count; ++partition) {
    const u32 begin = workspace.partition_offsets[partition];
    const u32 end = begin + workspace.partition_counts[partition];
    if (begin >= end || end > request.packet_count) {
      workspace.schedule.partitions.clear();
      return false;
    }
    out_max_packets = std::max<u32>(out_max_packets, end - begin);
    workspace.schedule.partitions[partition] = Partition{
        .worker_index = partition,
        .begin = begin,
        .end = end,
    };
  }
  return true;
}

} // namespace rund::kernel::workspace_detail
